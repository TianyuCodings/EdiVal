"""
Batch image editing using Qwen/Qwen-Image-Edit, mirroring the Flux Kontext pipeline structure.

Reads oai_instruction_generation_output.csv and input_images_resize_512.zip,
then generates multipass (sequential) and singlepass (progressive) edited images.
"""

import os
import ast
import zipfile
import tempfile
from typing import List

import pandas as pd
from PIL import Image
import torch
import torch.multiprocessing as mp

from diffusers import QwenImageEditPipeline


MODEL_ID = "Qwen/Qwen-Image-Edit"
TORCH_DTYPE = torch.bfloat16


class QwenImageEditGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print(f"Initializing Qwen-Image-Edit on device: {device}")
        try:
            self.pipe = QwenImageEditPipeline.from_pretrained(MODEL_ID).to(TORCH_DTYPE).to(device)
            print("pipeline loaded")
            # dtype and device
            # try:
            #     self.pipe.to(TORCH_DTYPE)
            # except Exception:
            #     pass
            # self.pipe.to(device)
            # progress bar visibility (None -> show)
            if hasattr(self.pipe, "set_progress_bar_config"):
                self.pipe.set_progress_bar_config(disable=None)
            print("Qwen-Image-Edit pipeline initialized successfully!")
        except Exception as e:
            print(f"Error initializing Qwen-Image-Edit: {e}")
            raise

    @torch.inference_mode()
    def generate_single_edit(
        self,
        instruction: str,
        current_image: Image.Image,
        seed: int | None = None,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
        num_inference_steps: int = 50,
    ) -> Image.Image:
        if seed is not None:
            gen = torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        else:
            gen = None

        inputs = {
            "image": current_image,
            "prompt": instruction,
            "generator": gen,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }
        out = self.pipe(**inputs)
        return out.images[0]


def worker_process(gpu_id, task_queue, result_queue, zip_file, output_dir, zip_files):
    try:
        # Bind to device
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
            print(f"🚀 Worker starting on GPU {gpu_id}")
        else:
            device = "cpu"
            print("🚀 Worker starting on CPU")

        result_queue.put(f"🚀 Worker {gpu_id} started successfully")

        try:
            print(f"📥 Loading Qwen-Image-Edit model on {device}...")
            generator = QwenImageEditGenerator(device=device)
            result_queue.put(f"✅ Model loaded on {device}")

            tasks_processed = 0
            while True:
                try:
                    task = task_queue.get(timeout=6000)
                    if task is None:
                        result_queue.put(
                            f"🛑 Worker {gpu_id} received stop signal after processing {tasks_processed} tasks"
                        )
                        break
                    res = process_single_image_on_gpu(task, zip_file, output_dir, generator, zip_files)
                    result_queue.put(res)
                    tasks_processed += 1
                except Exception as e:
                    if "Empty" in str(e) or "timeout" in str(e).lower():
                        result_queue.put(
                            f"⏰ Worker {gpu_id} timeout waiting for tasks (processed {tasks_processed})"
                        )
                        break
                    else:
                        import traceback
                        result_queue.put(f"❌ Worker error on GPU {gpu_id}: {e}\n{traceback.format_exc()}")

        except Exception as e:
            import traceback
            result_queue.put(f"❌ Failed to load model on GPU {gpu_id}: {e}\n{traceback.format_exc()}")
    except Exception as e:
        import traceback
        result_queue.put(f"❌ Worker {gpu_id} failed to start: {e}\n{traceback.format_exc()}")


def process_single_image_on_gpu(row_data, zip_file_path, output_dir, generator: QwenImageEditGenerator, zip_files: List[str]):
    try:
        image_index = row_data["image_index"]
        instructions_str = row_data["instructions"]

        # Accept .jpg or .png
        image_filename = None
        for ext in ("jpg", "png"):
            candidate = f"{image_index}_input_raw.{ext}"
            if candidate in zip_files:
                image_filename = candidate
                break
        if image_filename is None:
            return f"❌ Image not found for index {image_index}"

        # Extract to temp
        with zipfile.ZipFile(zip_file_path, "r") as zf:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(zf.read(image_filename))
                temp_image_path = tmp.name

        # Parse instructions (list or single string)
        try:
            instructions = ast.literal_eval(instructions_str)
            if not isinstance(instructions, list):
                instructions = [instructions_str]
        except Exception:
            instructions = [instructions_str]

        base = f"{image_index}_input_raw"
        multipass_dir = os.path.join(output_dir, "multipass")
        singlepass_dir = os.path.join(output_dir, "singlepass")
        os.makedirs(multipass_dir, exist_ok=True)
        os.makedirs(singlepass_dir, exist_ok=True)

        # Save source to both dirs (PNG, 512x512)
        with Image.open(temp_image_path) as img:
            img_rgb = img.convert("RGB").resize((512, 512), Image.LANCZOS)
            img_rgb.save(os.path.join(multipass_dir, f"{base}.png"), "PNG")
            img_rgb.save(os.path.join(singlepass_dir, f"{base}.png"), "PNG")

        results = []

        # 1) Multipass: apply each instruction sequentially
        try:
            current = Image.open(temp_image_path).convert("RGB")
            for i, instr in enumerate(instructions):
                out_name = f"{base}_turn_{i+1}.png"
                out_path = os.path.join(multipass_dir, out_name)

                edited = generator.generate_single_edit(
                    instruction=instr,
                    current_image=current,
                    seed=1234,
                )
                if edited:
                    resized = edited.resize((512, 512), Image.LANCZOS)
                    resized.save(out_path)
                    current = resized
                    results.append(f"✅ Multipass Turn {i+1} completed: multipass/{out_name}")
                else:
                    results.append(f"❌ Failed multipass at turn {i+1}")
                    break
        except Exception as e:
            results.append(f"❌ Error during multipass generation: {e}")

        # 2) Singlepass: progressively concatenate instructions up to turn i
        try:
            for i in range(len(instructions)):
                progressive = ". ".join(instructions[: i + 1])
                out_name = f"{base}_turn_{i+1}.png"
                out_path = os.path.join(singlepass_dir, out_name)

                ref_img = Image.open(temp_image_path).convert("RGB")
                edited = generator.generate_single_edit(
                    instruction=progressive,
                    current_image=ref_img,
                    seed=1234,
                )
                if edited:
                    resized = edited.resize((512, 512), Image.LANCZOS)
                    resized.save(out_path)
                    results.append(f"✅ Singlepass Turn {i+1} completed: singlepass/{out_name}")
                else:
                    results.append(f"❌ Failed singlepass at turn {i+1}")
        except Exception as e:
            results.append(f"❌ Error during singlepass generation: {e}")

        # cleanup
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)

        dev_info = f"GPU {generator.device}" if torch.cuda.is_available() else "CPU"
        return f"[{dev_info}] Image {image_index}: " + "; ".join(results)
    except Exception as e:
        return f"❌ Error processing image {row_data.get('image_index', 'unknown')}: {e}"


def process_batch_generation(num_gpus: int | None = None):
    csv_file = "oai_instruction_generation_output.csv"
    zip_file = "input_images_resize_512.zip"
    output_dir = "baseline_generations/QWEN_generation"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Original CSV shape: {df.shape}")
    df_turn3 = df[df["turns"] == 3].copy()
    print(f"Filtered for turn 3: {df_turn3.shape[0]} rows")
    if df_turn3.empty:
        print("No rows found with turns = 3")
        return

    avail = torch.cuda.device_count()
    if avail == 0:
        print("No GPUs available, using CPU")
        device_ids = [None]
        num_gpus = 0
    else:
        num_gpus = min(num_gpus or avail, avail)
        device_ids = list(range(num_gpus))
        print(f"Using {num_gpus} GPUs: {device_ids}")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    with zipfile.ZipFile(zip_file, "r") as zf:
        zip_files = zf.namelist()
        print(f"Found {len(zip_files)} files in zip")

    if num_gpus > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        print("\n" + "=" * 60)
        print(f"Starting multi-GPU processing with {num_gpus} workers...")
        print(f"Processing {len(df_turn3)} images...")

        task_queue = mp.Queue()
        result_queue = mp.Queue()
        for _, row in df_turn3.iterrows():
            task_queue.put(row.to_dict())
        for _ in range(num_gpus):
            task_queue.put(None)

        procs = []
        for gid in device_ids:
            p = mp.Process(
                target=worker_process,
                args=(gid, task_queue, result_queue, zip_file, output_dir, zip_files),
            )
            p.start()
            procs.append(p)

        processed = 0
        total = len(df_turn3)
        total_msgs = 0
        print("Waiting for workers to initialize...")
        workers_loaded = 0
        while workers_loaded < num_gpus:
            try:
                msg = result_queue.get(timeout=6000)
                print(msg)
                if "Model loaded" in msg:
                    workers_loaded += 1
                total_msgs += 1
            except Exception as e:
                print(f"Timeout waiting for worker initialization: {e}")
                break

        while processed < total and total_msgs < (total + num_gpus * 3):
            try:
                msg = result_queue.get(timeout=6000)
                print(msg)
                if "Image" in msg and (": ✅" in msg or ": ❌" in msg or "completed" in msg):
                    processed += 1
                    print(f"Progress: {processed}/{total}")
                total_msgs += 1
            except Exception as e:
                print(f"Timeout waiting for results: {e}")
                break

        for p in procs:
            p.join(timeout=6000)
            if p.is_alive():
                p.terminate()
        print("All workers finished.")
    else:
        # Single device/CPU path
        print("Single-device processing...")
        res_queue = []
        gen = QwenImageEditGenerator(device=device_ids[0] if device_ids[0] is not None else "cpu")
        for _, row in df_turn3.iterrows():
            msg = process_single_image_on_gpu(row.to_dict(), zip_file, output_dir, gen, zip_files)
            print(msg)
            res_queue.append(msg)


if __name__ == "__main__":
    # Optional: allow overriding number of GPUs via env NUM_GPUS
    env_ngpus = os.environ.get("NUM_GPUS")
    n = int(env_ngpus) if env_ngpus and env_ngpus.isdigit() else None
    process_batch_generation(num_gpus=n)

