# Flux Kontext Batch Image Editing Implementation
# This implementation adapts the InstructPix2Pix batch processing structure to use Flux Kontext model
# for sequential image editing based on CSV instructions

import pandas as pd
import zipfile
import os
import ast
from PIL import Image
import tempfile
import torch
import torch.multiprocessing as mp
import argparse
import sys
import json
import shutil
from typing import List, Dict, Any

# Import the Flux Kontext dependencies
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

# Flux Kontext Configuration
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
TORCH_DTYPE = torch.bfloat16

class FluxKontextGenerator:
    def __init__(self, device="cuda"):
        """Initialize the Flux Kontext pipeline."""
        self.device = device
        print(f"Initializing Flux Kontext on device: {device}")
        
        try:
            # Load the model
            self.pipe = FluxKontextPipeline.from_pretrained(
                MODEL_ID, 
                torch_dtype=TORCH_DTYPE
            )
            self.pipe.to(device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            
            # NOTE: Disabling model CPU offload for multi-GPU processing
            # enable_model_cpu_offload() can interfere with multi-GPU parallelism
            # by moving model components to CPU, causing inference to happen on fewer GPUs
            # if hasattr(self.pipe, "enable_model_cpu_offload") and device != "cpu":
            #     self.pipe.enable_model_cpu_offload()
                
            print("Flux Kontext pipeline initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing Flux Kontext: {e}")
            raise

    def generate_single_edit(self, instruction: str, current_image: Image.Image, 
                           guidance_scale: float = 2.5,
                           seed: int = None) -> Image.Image:
        """Generate a single edited image based on instruction using Flux Kontext."""
        try:
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Generate the edited image
            result = self.pipe(
                image=current_image,
                prompt=instruction,
                guidance_scale=guidance_scale
            )
            
            # Get the generated image
            edited_image = result.images[0]
            
            return edited_image
            
        except Exception as e:
            print(f"Error in image generation: {e}")
            raise

def worker_process(gpu_id, task_queue, result_queue, zip_file, output_dir, zip_files):
    """Worker process that runs on a specific GPU"""
    
    try:
        # Set the GPU for this process
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
            print(f"🚀 Worker starting on GPU {gpu_id}")
        else:
            device = "cpu"
            print(f"🚀 Worker starting on CPU")
        
        result_queue.put(f"🚀 Worker {gpu_id} started successfully")
        
        try:
            # Load model on this specific GPU
            print(f"📥 Loading Flux Kontext model on GPU {gpu_id}...")
            generator = FluxKontextGenerator(device=device)
            result_queue.put(f"✅ Model loaded on GPU {gpu_id}")
            
            # Process tasks from the queue
            tasks_processed = 0
            while True:
                try:
                    task = task_queue.get(timeout=300)
                    if task is None:  # Poison pill to stop worker
                        result_queue.put(f"🛑 Worker {gpu_id} received stop signal after processing {tasks_processed} tasks")
                        break
                    
                    row_data = task
                    result = process_single_image_on_gpu(
                        row_data, zip_file, output_dir, generator, zip_files
                    )
                    result_queue.put(result)
                    tasks_processed += 1
                    
                except Exception as e:
                    if "Empty" in str(e) or "timeout" in str(e).lower():
                        result_queue.put(f"⏰ Worker {gpu_id} timeout waiting for tasks (processed {tasks_processed})")
                        break
                    else:
                        result_queue.put(f"❌ Worker error on GPU {gpu_id}: {e}")
                        import traceback
                        result_queue.put(f"❌ Traceback: {traceback.format_exc()}")
        
        except Exception as e:
            result_queue.put(f"❌ Failed to load model on GPU {gpu_id}: {e}")
            import traceback
            result_queue.put(f"❌ Model loading traceback: {traceback.format_exc()}")
    
    except Exception as e:
        result_queue.put(f"❌ Worker {gpu_id} failed to start: {e}")
        import traceback
        result_queue.put(f"❌ Worker startup traceback: {traceback.format_exc()}")
    
    print(f"🔚 Worker on GPU {gpu_id} finished")

def process_single_image_on_gpu(row_data, zip_file_path, output_dir, generator, zip_files):
    """Process a single image using the pre-loaded Flux Kontext generator with both multipass and singlepass approaches"""
    
    try:
        image_index = row_data['image_index']
        instructions_str = row_data['instructions']
        
        # Try both .png and .jpg extensions
        image_filename_jpg = f"{image_index}_input_raw.jpg"
        image_filename_png = f"{image_index}_input_raw.png"
        
        image_filename = None
        if image_filename_jpg in zip_files:
            image_filename = image_filename_jpg
        elif image_filename_png in zip_files:
            image_filename = image_filename_png
        
        if image_filename is None:
            return f"❌ Image not found for index {image_index}"
        
        # Extract image to temporary location
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(zip_ref.read(image_filename))
                temp_image_path = tmp_file.name
        
        # Parse instructions
        try:
            instructions = ast.literal_eval(instructions_str)
            if not isinstance(instructions, list):
                instructions = [instructions_str]
        except:
            instructions = [instructions_str]
        
        base_name = f"{image_index}_input_raw"
        
        # Create subdirectories
        multipass_dir = os.path.join(output_dir, "multipass")
        singlepass_dir = os.path.join(output_dir, "singlepass")
        os.makedirs(multipass_dir, exist_ok=True)
        os.makedirs(singlepass_dir, exist_ok=True)
        
        # Save source image in both subdirectories
        src_output_path_multipass = os.path.join(multipass_dir, f"{base_name}.png")
        src_output_path_singlepass = os.path.join(singlepass_dir, f"{base_name}.png")
        with Image.open(temp_image_path) as img:
            # Keep image in its original format for Flux
            img_rgb = img.convert("RGB")
            
            img_rgb.save(src_output_path_multipass, "PNG")
            img_rgb.save(src_output_path_singlepass, "PNG")
        
        results = []
        
        # 1. Sequential editing (multipass) - each instruction applied to result of previous
        try:
            # Start with the source image in memory
            current_ref_image = Image.open(temp_image_path).convert("RGB")
            
            for i, instruction in enumerate(instructions):
                if i == 0:
                    output_filename = f"{base_name}_turn_1.png"
                elif i == 1:
                    output_filename = f"{base_name}_turn_2.png"
                else:
                    output_filename = f"{base_name}_turn_{i+1}.png"
                
                output_path = os.path.join(multipass_dir, output_filename)
                
                # Generate edited image using Flux Kontext
                edited_image = generator.generate_single_edit(
                    instruction=instruction,
                    current_image=current_ref_image,
                    guidance_scale=2.5,
                    seed=1234
                )
                
                if edited_image:
                    # Resize to 512x512 before saving
                    resized_image = edited_image.resize((512, 512), Image.LANCZOS)
                    resized_image.save(output_path)
                    # Keep the resized result in memory for next iteration
                    current_ref_image = resized_image
                    results.append(f"✅ Multipass Turn {i+1} completed: multipass/{output_filename}")
                else:
                    results.append(f"❌ Failed to generate multipass image for instruction: {instruction}")
                    break
        
        except Exception as e:
            results.append(f"❌ Error during multipass generation: {e}")
        
        # 2. Single-pass editing - generate 3 turns using progressive instruction sets
        try:
            for i in range(len(instructions)):
                # Use instructions[:i+1] for turn i+1
                progressive_prompt = ". ".join(instructions[:i+1])
                turn_output_path = os.path.join(singlepass_dir, f"{base_name}_turn_{i+1}.png")
                
                # Always use original source image
                ref_image = Image.open(temp_image_path).convert("RGB")
                
                # Generate edited image using Flux Kontext
                edited_image = generator.generate_single_edit(
                    instruction=progressive_prompt,
                    current_image=ref_image,
                    guidance_scale=2.5,
                    seed=1234
                )
                
                if edited_image:
                    # Resize to 512x512 before saving
                    resized_image = edited_image.resize((512, 512), Image.LANCZOS)
                    resized_image.save(turn_output_path)
                    results.append(f"✅ Singlepass Turn {i+1} completed: singlepass/{base_name}_turn_{i+1}.png")
                else:
                    results.append(f"❌ Failed to generate singlepass turn {i+1} image")
                
        except Exception as e:
            results.append(f"❌ Error during singlepass generation: {e}")
        
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        
        device_info = f"GPU {generator.device}" if hasattr(generator, 'device') and torch.cuda.is_available() else "CPU"
        return f"[{device_info}] Image {image_index}: " + "; ".join(results)
        
    except Exception as e:
        return f"❌ Error processing image {row_data.get('image_index', 'unknown')}: {e}"

def process_batch_generation(num_gpus=None, max_workers=None):
    """
    Process batch generation from CSV and image zip file with multi-GPU support using Flux Kontext.
    
    Args:
        num_gpus: Number of GPUs to use. If None, uses all available GPUs.
        max_workers: Maximum number of parallel workers. If None, uses num_gpus.
    """
    csv_file = "oai_instruction_generation_output.csv"
    zip_file = "input_images_resize_512.zip"
    output_dir = "FLUX_generation"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Filter for rows where turns is 3
    df_turn3 = df[df['turns'] == 3].copy()
    print(f"Filtered for turn 3: {df_turn3.shape[0]} rows")
    
    if df_turn3.empty:
        print("No rows found with turns = 3")
        return
    
    # Setup multi-GPU configuration
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No GPUs available, using CPU")
        num_gpus = 0
        device_ids = [None]
    else:
        num_gpus = num_gpus or available_gpus
        num_gpus = min(num_gpus, available_gpus)
        device_ids = list(range(num_gpus))
        print(f"Using {num_gpus} GPUs: {device_ids}")
        
        # Print GPU info
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Using Flux Kontext model
    print("🔄 Will load Flux Kontext model from Hugging Face when initializing workers...")
    
    # Get list of files in zip
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_files = zip_ref.namelist()
        print(f"Found {len(zip_files)} files in zip")
    
    if num_gpus > 1:
        # Set spawn method for CUDA multiprocessing
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, which is fine
            pass
        
        # Multi-GPU processing using torch.multiprocessing
        print(f"\n{'='*60}")
        print(f"Starting multi-GPU processing with {num_gpus} workers...")
        print(f"Processing {len(df_turn3)} images...")
        
        # Create queues for task distribution
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Add all tasks to queue
        for idx, row in df_turn3.iterrows():
            task_queue.put(row.to_dict())
        
        # Add poison pills to stop workers
        for _ in range(num_gpus):
            task_queue.put(None)
        
        # Start worker processes
        processes = []
        for gpu_id in device_ids:
            p = mp.Process(target=worker_process, args=(
                gpu_id, task_queue, result_queue, zip_file, output_dir, zip_files
            ))
            p.start()
            processes.append(p)
        
        # Collect results
        processed_count = 0
        total_tasks = len(df_turn3)
        total_results = 0
        
        # First, wait for all workers to start and load models
        workers_ready = 0
        print("Waiting for workers to initialize...")
        
        while workers_ready < num_gpus:
            try:
                result = result_queue.get(timeout=300)  # Give more time for model loading
                print(result)
                if "started successfully" in result or "Model loaded" in result:
                    if "Model loaded" in result:
                        workers_ready += 1
                elif "❌" in result:
                    print(f"Worker failed to initialize: {result}")
                total_results += 1
            except Exception as e:
                print(f"Timeout waiting for worker initialization: {e}")
                break
        
        print(f"Workers ready: {workers_ready}/{num_gpus}")
        
        # Now collect task results
        while processed_count < total_tasks and total_results < (total_tasks + num_gpus * 3):  # Buffer for worker messages
            try:
                result = result_queue.get(timeout=300)  # Longer timeout for processing
                print(result)
                
                # Count actual image processing results
                if "Image" in result and (": ✅" in result or ": ❌" in result):
                    processed_count += 1
                    print(f"Progress: {processed_count}/{total_tasks}")
                
                total_results += 1
                
            except Exception as e:
                print(f"Timeout waiting for result: {e}")
                break
        
        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=300)
            if p.is_alive():
                p.terminate()
    
    else:
        # Single GPU/CPU processing
        print(f"\n{'='*60}")
        print(f"Starting single-device processing...")
        print(f"Processing {len(df_turn3)} images...")
        
        # Setup single generator
        device = device_ids[0] if device_ids[0] is not None else "cpu"
        if device != "cpu":
            torch.cuda.set_device(0)
        
        generator = FluxKontextGenerator(device=device)
        
        processed_count = 0
        for idx, row in df_turn3.iterrows():
            try:
                result = process_single_image_on_gpu(
                    row.to_dict(), zip_file, output_dir, generator, zip_files
                )
                print(result)
                processed_count += 1
                print(f"Progress: {processed_count}/{len(df_turn3)}")
            except Exception as e:
                print(f"❌ Error in single-device processing: {e}")
    
    print(f"\n{'='*60}")
    print(f"Flux Kontext batch processing completed!")
    print(f"Successfully processed: {processed_count}/{len(df_turn3)} images")
    print(f"Output directory: {output_dir}")
    print(f"Generated files for each image:")
    print(f"  Multipass subdirectory (sequential editing):")
    print(f"    - {{base_name}}.png (source image)")
    print(f"    - {{base_name}}_turn_1.png (first instruction applied)")
    print(f"    - {{base_name}}_turn_2.png (second instruction applied)")
    print(f"    - {{base_name}}_turn_3.png (third instruction applied)")
    print(f"  Singlepass subdirectory (progressive instruction sets):")
    print(f"    - {{base_name}}.png (source image)")
    print(f"    - {{base_name}}_turn_1.png (instruction 1 only)")
    print(f"    - {{base_name}}_turn_2.png (instructions 1+2 combined)")
    print(f"    - {{base_name}}_turn_3.png (instructions 1+2+3 combined)")

def main():
    parser = argparse.ArgumentParser(description="Batch Flux Kontext image generation with multi-GPU support")
    parser.add_argument('--num_gpus', type=int, default=8, 
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: same as num_gpus)')
    
    args = parser.parse_args()
    
    process_batch_generation(num_gpus=args.num_gpus, max_workers=args.max_workers)

if __name__ == "__main__":
    print("=" * 80)
    print("FLUX KONTEXT BATCH IMAGE EDITING TOOL")
    print("=" * 80)
    print("Implementation using Flux Kontext pipeline from Black Forest Labs")
    print("Adapted from InstructPix2Pix batch processing structure")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        main()
    else:
        # Default behavior for backward compatibility
        process_batch_generation() 