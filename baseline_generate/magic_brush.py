# MagicBrush Batch Image Editing Implementation
# This implementation adapts the OmniGen batch processing structure to use MagicBrush model
# for sequential image editing based on CSV instructions
# Paper: "MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing"
# GitHub: https://github.com/OSU-NLP-Group/MagicBrush

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
import requests
from typing import List, Dict, Any

# Import the MagicBrush dependencies
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# MagicBrush Configuration
# Use the properly converted diffusers models
MAGICBRUSH_DIFFUSERS_LATEST = "vinesmsuic/magicbrush-jul7"  # Working diffusers version
MAGICBRUSH_DIFFUSERS_ALTERNATIVE = "osunlp/MagicBrush-Jul17"  # Alternative version
DEFAULT_MODEL = MAGICBRUSH_DIFFUSERS_LATEST  # Use the working version
TORCH_DTYPE = torch.float16

class MagicBrushGenerator:
    def __init__(self, device="cuda", model_name: str = None):
        """Initialize the MagicBrush pipeline using diffusers-compatible models."""
        self.device = device
        self.model_name = model_name or DEFAULT_MODEL
        
        print(f"Initializing MagicBrush on device: {device}")
        print(f"Using model: {self.model_name}")
        
        try:
            # Load MagicBrush model directly from HuggingFace (diffusers-compatible)
            print(f"Loading MagicBrush model from {self.model_name}...")
            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_name,  # Use MagicBrush diffusers model directly
                torch_dtype=TORCH_DTYPE, 
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipe.to(device)
            
            # Use the recommended scheduler for better results
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Enable memory efficient attention if available
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            
            # NOTE: Disabling model CPU offload for multi-GPU processing
            # enable_model_cpu_offload() can interfere with multi-GPU parallelism
            # by moving model components to CPU, causing inference to happen on fewer GPUs
            # if hasattr(self.pipe, "enable_model_cpu_offload") and device != "cpu":
            #     self.pipe.enable_model_cpu_offload()
                
            print("MagicBrush pipeline initialized successfully!")
            print("✓ MagicBrush is optimized for fine-grained real image editing")
            print("✓ Trained on manually annotated real images (not synthetic data)")
            print(f"✓ Using diffusers-compatible model: {self.model_name}")
            
            # Verify that MagicBrush model works correctly
            self._verify_weights_loaded()
            
        except Exception as e:
            print(f"Error initializing MagicBrush: {e}")
            raise

    def _verify_weights_loaded(self):
        """Verify that MagicBrush model is working correctly."""
        try:
            # Create a small test input
            import torch
            from PIL import Image
            import numpy as np
            
            # Create a dummy image
            dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
            
            # Test with a simple instruction
            test_instruction = "change the color to red"
            
            # Generate with current weights
            with torch.no_grad():
                result = self.pipe(
                    test_instruction,
                    image=dummy_image,
                    num_inference_steps=1,  # Just 1 step for quick test
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                )
            
            print("✓ MagicBrush model verification: Model generates outputs successfully")
            return True
            
        except Exception as e:
            print(f"⚠ MagicBrush model verification failed: {e}")
            return False

    def generate_single_edit(self, instruction: str, current_image: Image.Image, 
                           image_guidance_scale: float = 1.5, 
                           guidance_scale: float = 7,  # Match the example
                           num_inference_steps: int = 20,
                           seed: int = None) -> Image.Image:
        """Generate a single edited image based on instruction using MagicBrush."""
        try:
            # Set random seed and create generator if provided
            generator = None
            if seed is not None:
                generator = torch.manual_seed(seed)
            
            # Generate the edited image (matching the example format)
            result = self.pipe(
                instruction,  # instruction prompt first (like example)
                image=current_image,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
            # Get the generated image
            edited_image = result.images[0]
            
            return edited_image
            
        except Exception as e:
            print(f"Error in image generation: {e}")
            raise

def worker_process(gpu_id, task_queue, result_queue, zip_file, output_dir, zip_files, model_name):
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
            print(f"📥 Loading MagicBrush model on GPU {gpu_id}...")
            generator = MagicBrushGenerator(device=device, model_name=model_name)
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
    """Process a single image using the pre-loaded MagicBrush generator with both multipass and singlepass approaches"""
    
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
            # Resize image for optimal MagicBrush processing
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
                
                # Generate edited image using MagicBrush
                edited_image = generator.generate_single_edit(
                    instruction=instruction,
                    current_image=current_ref_image,
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                    num_inference_steps=100,
                    seed=1234
                )
                
                if edited_image:
                    edited_image.save(output_path)
                    # Keep the result in memory for next iteration
                    current_ref_image = edited_image
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
                
                # Generate edited image using MagicBrush
                edited_image = generator.generate_single_edit(
                    instruction=progressive_prompt,
                    current_image=ref_image,
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                    num_inference_steps=100,
                    seed=1234
                )
                
                if edited_image:
                    edited_image.save(turn_output_path)
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

def process_batch_generation(num_gpus=None, max_workers=None, model_name=None):
    """
    Process batch generation from CSV and image zip file with multi-GPU support using MagicBrush.
    
    Args:
        num_gpus: Number of GPUs to use. If None, uses all available GPUs.
        max_workers: Maximum number of parallel workers. If None, uses num_gpus.
        model_name: Specific MagicBrush model to use (osunlp/MagicBrush-Jul17 or osunlp/MagicBrush-Paper)
    """
    csv_file = "oai_instruction_generation_output.csv"
    zip_file = "input_images_resize_512.zip"
    output_dir = "MagicBrush_generation"
    
    # Use default model if none specified
    model_name = model_name or DEFAULT_MODEL
    
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
    
    # Using MagicBrush model
    print(f"🔄 Will load MagicBrush model: {model_name}")
    
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
                gpu_id, task_queue, result_queue, zip_file, output_dir, zip_files, model_name
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
                result = result_queue.get(timeout=600)  # Give more time for model + checkpoint loading
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
        
        generator = MagicBrushGenerator(device=device, model_name=model_name)
        
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
    print(f"MagicBrush batch processing completed!")
    print(f"Successfully processed: {processed_count}/{len(df_turn3)} images")
    print(f"Output directory: {output_dir}")
    print(f"Using model: {model_name}")
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
    parser = argparse.ArgumentParser(description="Batch MagicBrush image generation with multi-GPU support")
    parser.add_argument('--num_gpus', type=int, default=4, 
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: same as num_gpus)')
    parser.add_argument('--model', type=str, default=None,
                        choices=[MAGICBRUSH_DIFFUSERS_LATEST, MAGICBRUSH_DIFFUSERS_ALTERNATIVE],
                        help=f'Model to use (default: {DEFAULT_MODEL})')
    
    args = parser.parse_args()
    
    process_batch_generation(num_gpus=args.num_gpus, max_workers=args.max_workers, model_name=args.model)

if __name__ == "__main__":
    print("=" * 80)
    print("MAGICBRUSH BATCH IMAGE EDITING TOOL")
    print("=" * 80)
    print("Implementation of 'MagicBrush: A Manually Annotated Dataset for")
    print("Instruction-Guided Image Editing' by Zhang et al. (NeurIPS 2023)")
    print("GitHub: https://github.com/OSU-NLP-Group/MagicBrush")
    print(f"Available models: {MAGICBRUSH_DIFFUSERS_LATEST}, {MAGICBRUSH_DIFFUSERS_ALTERNATIVE}")
    print(f"Default model: {DEFAULT_MODEL}")
    print("=" * 80)
    print("Note: MagicBrush is optimized for fine-grained real image editing")
    print("and may not perform well on style transfer or large region modifications.")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        main()
    else:
        # Default behavior for backward compatibility
        process_batch_generation()