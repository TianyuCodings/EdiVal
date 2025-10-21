#!/usr/bin/env python
"""
Self-contained CLI for running Qwen/Qwen-Image-Edit generations.

Key features:
- Ships with a default `QwenImageEditGenerator` implementation.
- Supports custom editor classes via `--editor-class module:ClassName`.
- Provides multiprocessing workers and single-device execution paths.

Generations are written to:

your_generations/
  multipass/
  singlepass/
"""

from __future__ import annotations

import argparse
import ast
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Type

import pandas as pd
import torch
import torch.multiprocessing as mp
from diffusers import QwenImageEditPipeline
from PIL import Image


MODEL_ID = "Qwen/Qwen-Image-Edit"
TORCH_DTYPE = torch.bfloat16
DEFAULT_SEED = 1234
DEFAULT_IMAGE_SIZE = 512


class QwenImageEditGenerator:
    """Default editor backed by `diffusers.QwenImageEditPipeline`."""

    def __init__(self, device: str = "cuda", model_id: str = MODEL_ID):
        self.device = device
        self.model_id = model_id
        print(f"Initializing {model_id} on device: {device}")
        self.pipe = QwenImageEditPipeline.from_pretrained(model_id).to(TORCH_DTYPE).to(device)
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=None)
        print(f"{model_id} pipeline ready on {device}")

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
            generator = torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        else:
            generator = None

        outputs = self.pipe(
            image=current_image,
            prompt=instruction,
            generator=generator,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
        )
        return outputs.images[0]


def process_single_image_on_gpu(
    row_data,
    zip_file_path: str,
    output_dir: str,
    generator: QwenImageEditGenerator,
    zip_files: List[str],
    seed: int = DEFAULT_SEED,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> str:
    """Run multipass and singlepass generations for a single CSV row."""
    try:
        image_index = row_data["image_index"]
        instructions_str = row_data["instructions"]

        image_filename = None
        for ext in ("jpg", "png"):
            candidate = f"{image_index}_input_raw.{ext}"
            if candidate in zip_files:
                image_filename = candidate
                break
        if image_filename is None:
            return f"❌ Image not found for index {image_index}"

        with zipfile.ZipFile(zip_file_path, "r") as zf:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(zf.read(image_filename))
                temp_image_path = tmp.name

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

        with Image.open(temp_image_path) as img:
            source = img.convert("RGB").resize((image_size, image_size), Image.LANCZOS)
            source.save(os.path.join(multipass_dir, f"{base}.png"), "PNG")
            source.save(os.path.join(singlepass_dir, f"{base}.png"), "PNG")

        results = []

        # Multipass: apply instructions sequentially.
        try:
            current = Image.open(temp_image_path).convert("RGB")
            for turn, instruction in enumerate(instructions, start=1):
                out_name = f"{base}_turn_{turn}.png"
                out_path = os.path.join(multipass_dir, out_name)
                edited = generator.generate_single_edit(
                    instruction=instruction,
                    current_image=current,
                    seed=seed,
                )
                resized = edited.resize((image_size, image_size), Image.LANCZOS)
                resized.save(out_path)
                current = resized
                results.append(f"✅ Multipass Turn {turn} completed: multipass/{out_name}")
        except Exception as exc:
            results.append(f"❌ Error during multipass generation: {exc}")

        # Singlepass: accumulate prompts up to each turn.
        try:
            for turn in range(len(instructions)):
                progressive_prompt = ". ".join(instructions[: turn + 1])
                out_name = f"{base}_turn_{turn+1}.png"
                out_path = os.path.join(singlepass_dir, out_name)
                ref_img = Image.open(temp_image_path).convert("RGB")
                edited = generator.generate_single_edit(
                    instruction=progressive_prompt,
                    current_image=ref_img,
                    seed=seed,
                )
                resized = edited.resize((image_size, image_size), Image.LANCZOS)
                resized.save(out_path)
                results.append(f"✅ Singlepass Turn {turn+1} completed: singlepass/{out_name}")
        except Exception as exc:
            results.append(f"❌ Error during singlepass generation: {exc}")

        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)

        dev_info = f"GPU {generator.device}" if torch.cuda.is_available() else "CPU"
        return f"[{dev_info}] Image {image_index}: " + "; ".join(results)
    except Exception as exc:
        return f"❌ Error processing image {row_data.get('image_index', 'unknown')}: {exc}"


def worker_process(
    gpu_id,
    task_queue,
    result_queue,
    zip_file,
    output_dir,
    zip_files,
    editor_cls: Type[QwenImageEditGenerator],
):
    """Spawned worker that handles a subset of the images."""
    try:
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
            print(f"🚀 Worker starting on GPU {gpu_id}")
        else:
            device = "cpu"
            print("🚀 Worker starting on CPU")

        result_queue.put(f"🚀 Worker {gpu_id} started successfully")
        generator = editor_cls(device=device)
        result_queue.put(f"✅ Model loaded on {device}")

        tasks_processed = 0
        while True:
            task = task_queue.get(timeout=6000)
            if task is None:
                result_queue.put(
                    f"🛑 Worker {gpu_id} received stop signal after processing {tasks_processed} tasks"
                )
                break
            res = process_single_image_on_gpu(task, zip_file, output_dir, generator, zip_files)
            result_queue.put(res)
            tasks_processed += 1
    except Exception as exc:
        import traceback

        result_queue.put(
            f"❌ Worker {gpu_id if gpu_id is not None else 'CPU'} error: {exc}\n{traceback.format_exc()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch image editing with Qwen/Qwen-Image-Edit using the EdiVal data format."
    )
    parser.add_argument(
        "--csv",
        default="oai_instruction_generation_output.csv",
        help="Path to the instruction CSV file.",
    )
    parser.add_argument(
        "--zip",
        default="input_images_resize_512.zip",
        help="Path to the zipped input images (resized to 512).",
    )
    parser.add_argument(
        "--output-dir",
        default="your_generations",
        help="Destination folder for generated images.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use. Default: detect automatically.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows/images to process.",
    )
    parser.add_argument(
        "--editor-class",
        default=None,
        help=(
            "Optional dotted path to a custom editor class "
            "(e.g. 'scripts.my_editor:MyGenerator'). The class should accept a "
            "`device` argument and expose `generate_single_edit`."
        ),
    )
    return parser.parse_args()


def ensure_structure(base_dir: Path) -> None:
    """Ensure multipass/singlepass directories exist under the base dir."""
    for sub in ("multipass", "singlepass"):
        (base_dir / sub).mkdir(parents=True, exist_ok=True)


def resolve_editor_class(path: str | None) -> Type[QwenImageEditGenerator]:
    """Resolve the editor class, importing custom implementations on demand."""
    if not path:
        return QwenImageEditGenerator

    if ":" in path:
        module_name, class_name = path.split(":", 1)
    elif "." in path:
        module_name, class_name = path.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid editor class path '{path}'. Use 'module:ClassName'.")

    import importlib

    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' does not define '{class_name}'") from exc

    if not hasattr(cls, "generate_single_edit"):
        raise TypeError(
            f"Custom editor '{class_name}' must implement a 'generate_single_edit' method."
        )

    return cls


def run_generation(
    csv_path: str,
    zip_path: str,
    output_dir: str,
    num_gpus: int | None,
    limit: int | None,
    editor_cls: Type[QwenImageEditGenerator],
) -> None:
    output_root = Path(output_dir)
    ensure_structure(output_root)

    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    df_turn3 = df[df["turns"] == 3].copy()
    if limit is not None:
        df_turn3 = df_turn3.head(limit)
    print(f"Rows with turns == 3: {len(df_turn3)}")
    if df_turn3.empty:
        print("No rows found to process. Exiting.")
        return

    avail = torch.cuda.device_count()
    if avail == 0:
        print("No GPUs detected, running on CPU.")
        device_ids: List[int | None] = [None]
    else:
        resolved = min(num_gpus or avail, avail)
        device_ids = list(range(resolved))
        print(f"Using {resolved} GPU(s): {device_ids}")
        for idx in device_ids:
            name = torch.cuda.get_device_name(idx)
            mem = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            print(f"  GPU {idx}: {name} ({mem:.1f} GB)")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_files = zf.namelist()
        print(f"Found {len(zip_files)} files in zip archive.")

    if len(device_ids) > 1 and device_ids[0] is not None:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        print("=" * 60)
        print(f"Starting generation with {len(device_ids)} worker(s)")
        print(f"Output directory: {output_root.resolve()}")

        task_queue: mp.Queue = mp.Queue()
        result_queue: mp.Queue = mp.Queue()
        for _, row in df_turn3.iterrows():
            task_queue.put(row.to_dict())
        for _ in device_ids:
            task_queue.put(None)

        procs: List[mp.Process] = []
        for gid in device_ids:
            proc = mp.Process(
                target=worker_process,
                args=(gid, task_queue, result_queue, zip_path, str(output_root), zip_files, editor_cls),
            )
            proc.start()
            procs.append(proc)

        processed = 0
        total = len(df_turn3)
        try:
            while processed < total:
                msg = result_queue.get()
                print(msg)
                if msg and "Image" in msg:
                    processed += 1
                    print(f"Progress: {processed}/{total}")
        finally:
            for proc in procs:
                proc.join()
        print("All workers finished.")
    else:
        device = "cpu" if device_ids[0] is None else f"cuda:{device_ids[0]}"
        print(f"Single-device processing on {device}")
        generator = editor_cls(device=device)
        processed = 0
        for _, row in df_turn3.iterrows():
            msg = process_single_image_on_gpu(row.to_dict(), zip_path, str(output_root), generator, zip_files)
            print(msg)
            processed += 1
            if processed % 10 == 0 or processed == len(df_turn3):
                print(f"Progress: {processed}/{len(df_turn3)}")

    print("Generation complete.")


def main() -> None:
    args = parse_args()
    editor_cls = resolve_editor_class(args.editor_class)
    run_generation(
        csv_path=args.csv,
        zip_path=args.zip,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        limit=args.limit,
        editor_cls=editor_cls,
    )


if __name__ == "__main__":
    main()
