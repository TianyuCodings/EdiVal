#!/usr/bin/env python3
"""
Grounding DINO Multi-Node Object Filtering Script

This script processes JSON files containing object descriptions and filters them
using Grounding DINO visual grounding. It adapts the multi-processing framework
from ip2p_batch.py to work with Grounding DINO.

For each JSON file:
1. Read parsed_json data
2. For each object key, run Grounding DINO on the corresponding image
3. If boxes are detected, keep the object and add grounding_count and grounding_phrase
4. Save filtered results to output directory
"""

import os
import json
import torch
import torch.multiprocessing as mp
import argparse
import sys
from typing import List, Dict, Any, Tuple
import glob
from PIL import Image
import numpy as np

# Import Grounding DINO dependencies
try:
    from groundingdino.models import build_model
    from groundingdino.util import box_ops
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util.inference import annotate, load_image, predict
    import groundingdino.datasets.transforms as T
except ImportError as e:
    print(f"Error importing Grounding DINO: {e}")
    print("Please ensure Grounding DINO is properly installed")
    sys.exit(1)

# Configuration
INPUT_DIR = "/scratch/EditVal/generate_instructions/oai_all_objects"
OUTPUT_DIR = "/scratch/EditVal/generate_instructions/grounding_all_objects"
IMAGE_DIR = "/scratch/EditVal/input_images_resize_512"

# Grounding DINO model configuration
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

class GroundingDINOProcessor:
    def __init__(self, device="cuda"):
        """Initialize the Grounding DINO model."""
        self.device = device
        print(f"Initializing Grounding DINO on device: {device}")
        
        try:
            # Load model configuration
            args = SLConfig.fromfile(GROUNDING_DINO_CONFIG_PATH)
            args.device = device
            
            # Build and load model
            self.model = build_model(args)
            checkpoint = torch.load(GROUNDING_DINO_CHECKPOINT_PATH, map_location="cpu")
            load_res = self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            print(f"Model loading result: {load_res}")
            
            # Move model to device
            self.model.to(device)
            self.model.eval()
            
            # Setup transform
            self.transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            print("Grounding DINO model initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing Grounding DINO: {e}")
            raise

    def detect_objects(self, image_path: str, text_prompt: str, 
                      box_threshold: float = 0.35, 
                      text_threshold: float = 0.35,
                      max_box_size: float = 0.8) -> Tuple[List, List, List]:
        """
        Detect objects in image using Grounding DINO.
        
        Args:
            image_path: Path to the image file
            text_prompt: Text prompt for object detection
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold
            max_box_size: Maximum allowed box size (width or height) in normalized coordinates
        
        Returns:
            boxes: List of bounding boxes (filtered)
            phrases: List of detected phrases (filtered)
            scores: List of confidence scores (filtered)
        """
        try:
            # Load and preprocess image
            image_pil = Image.open(image_path).convert("RGB")
            image_source, image = load_image(image_path)
            
            # Run prediction
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
            # Filter out boxes that are too large
            if len(boxes) > 0:
                filtered_indices = []
                for i, box in enumerate(boxes):
                    # Box format is: center_x, center_y, width, height (all normalized)
                    center_x, center_y, width, height = box
                    
                    # Filter out boxes where width or height exceeds max_box_size
                    if (width > max_box_size or height > max_box_size) and (width * height > 0.4):
                        return [], [], []
                
                # Apply filtering to all three arrays
                filtered_boxes = boxes
                filtered_logits = logits
                filtered_phrases = phrases if phrases else []
                
                # Convert to list format
                boxes_list = filtered_boxes.cpu().numpy().tolist() if len(filtered_boxes) > 0 else []
                phrases_list = filtered_phrases if filtered_phrases else []
                scores_list = filtered_logits.cpu().numpy().tolist() if len(filtered_logits) > 0 else []
            else:
                boxes_list = []
                phrases_list = []
                scores_list = []
            
            return boxes_list, phrases_list, scores_list
            
        except Exception as e:
            print(f"Error in object detection for {image_path}: {e}")
            return [], [], []

def worker_process(gpu_id, task_queue, result_queue, max_box_size=0.9):
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
            print(f"📥 Loading Grounding DINO model on GPU {gpu_id}...")
            processor = GroundingDINOProcessor(device=device)
            result_queue.put(f"✅ Model loaded on GPU {gpu_id}")
            
            # Process tasks from the queue
            tasks_processed = 0
            while True:
                try:
                    task = task_queue.get(timeout=300)
                    if task is None:  # Poison pill to stop worker
                        result_queue.put(f"🛑 Worker {gpu_id} received stop signal after processing {tasks_processed} tasks")
                        break
                    
                    json_file_path = task
                    result = process_single_json_file(json_file_path, processor, max_box_size)
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

def process_single_json_file(json_file_path: str, processor: GroundingDINOProcessor, max_box_size: float = 0.8):
    """Process a single JSON file using Grounding DINO"""
    
    try:
        filename = os.path.basename(json_file_path)
        print(f"Processing {filename}...")
        
        # Load JSON data
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        if 'parsed_json' not in data:
            return f"❌ No parsed_json found in {filename}"
        
        parsed_json = data['parsed_json']
        image_index = data.get('image_index', filename.replace('.json', '').replace('_input_raw', ''))
        
        # Find corresponding image file
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = os.path.join(IMAGE_DIR, f"{image_index}_input_raw{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            return f"❌ Image not found for {filename}"
        
        # Process each object key (excluding "All Objects")
        filtered_objects = {}
        objects_processed = 0
        objects_kept = 0
        
        for key, value in parsed_json.items():
            if key == "All Objects":
                continue
                
            objects_processed += 1
            
            # Use the key as the text prompt for Grounding DINO
            text_prompt = key
            
            # Run Grounding DINO detection with box size filtering
            boxes, phrases, scores = processor.detect_objects(image_path, text_prompt, max_box_size=max_box_size)
            
            if boxes:  # If any boxes were detected
                # Keep this object and add grounding information
                filtered_objects[key] = value.copy()
                filtered_objects[key]['grounding_count'] = len(boxes)
                filtered_objects[key]['grounding_phrases'] = phrases
                filtered_objects[key]['grounding_boxes'] = boxes
                filtered_objects[key]['grounding_scores'] = scores
                objects_kept += 1
        
        # Add "All Objects" back if it exists
        if "All Objects" in parsed_json:
            # Create new "All Objects" string from kept objects
            kept_object_names = [key for key in filtered_objects.keys() if key != "All Objects"]
            filtered_objects["Filtered All Objects"] = ". ".join(kept_object_names) + "."
            filtered_objects["All Objects"] = parsed_json["All Objects"]
        
        # Update the data with filtered objects
        filtered_data = data.copy()
        filtered_data['parsed_json'] = filtered_objects
        filtered_data['grounding_filter_stats'] = {
            'objects_processed': objects_processed,
            'objects_kept': objects_kept,
            'objects_filtered_out': objects_processed - objects_kept,
            'max_box_size_used': max_box_size
        }
        
        # Save to output directory
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        device_info = f"GPU {processor.device}" if hasattr(processor, 'device') and torch.cuda.is_available() else "CPU"
        return f"[{device_info}] {filename}: {objects_kept}/{objects_processed} objects kept"
        
    except Exception as e:
        return f"❌ Error processing {json_file_path}: {e}"

def process_batch_filtering(num_gpus=None, max_workers=None, max_box_size=0.9):
    """
    Process batch filtering from JSON files with multi-GPU support using Grounding DINO.
    
    Args:
        num_gpus: Number of GPUs to use. If None, uses all available GPUs.
        max_workers: Maximum number of parallel workers. If None, uses num_gpus.
        max_box_size: Maximum allowed box size (width or height) in normalized coordinates.
    """
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON files to process")
    
    if not json_files:
        print("No JSON files found to process")
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
    
    print("🔄 Will load Grounding DINO model when initializing workers...")
    print(f"📏 Max box size threshold: {max_box_size}")
    
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
        print(f"Processing {len(json_files)} JSON files...")
        
        # Create queues for task distribution
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Add all tasks to queue
        for json_file in json_files:
            task_queue.put(json_file)
        
        # Add poison pills to stop workers
        for _ in range(num_gpus):
            task_queue.put(None)
        
        # Start worker processes
        processes = []
        for gpu_id in device_ids:
            p = mp.Process(target=worker_process, args=(
                gpu_id, task_queue, result_queue, max_box_size
            ))
            p.start()
            processes.append(p)
        
        # Collect results
        processed_count = 0
        total_tasks = len(json_files)
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
        while processed_count < total_tasks and total_results < (total_tasks + num_gpus * 3):
            try:
                result = result_queue.get(timeout=300)
                print(result)
                
                # Count actual file processing results
                if ".json:" in result and (": ✅" in result or "objects kept" in result):
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
        print(f"Processing {len(json_files)} JSON files...")
        
        # Setup single processor
        device = device_ids[0] if device_ids[0] is not None else "cpu"
        if device != "cpu":
            torch.cuda.set_device(0)
        
        processor = GroundingDINOProcessor(device=device)
        
        processed_count = 0
        for json_file in json_files:
            try:
                result = process_single_json_file(json_file, processor, max_box_size)
                print(result)
                processed_count += 1
                print(f"Progress: {processed_count}/{len(json_files)}")
            except Exception as e:
                print(f"❌ Error in single-device processing: {e}")
    
    print(f"\n{'='*60}")
    print(f"Grounding DINO filtering completed!")
    print(f"Successfully processed: {processed_count}/{len(json_files)} files")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max box size used: {max_box_size}")
    print(f"Each output file contains:")
    print(f"  - Original data with filtered parsed_json")
    print(f"  - grounding_count: number of detected boxes per object")
    print(f"  - grounding_phrases: detected phrases from Grounding DINO")
    print(f"  - grounding_boxes: bounding box coordinates")
    print(f"  - grounding_scores: confidence scores")
    print(f"  - grounding_filter_stats: filtering statistics")

def main():
    # Update global variables
    global INPUT_DIR, OUTPUT_DIR, IMAGE_DIR
    
    parser = argparse.ArgumentParser(description="Batch Grounding DINO object filtering with multi-GPU support")
    parser.add_argument('--num_gpus', type=int, default=8, 
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: same as num_gpus)')
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR,
                        help='Input directory containing JSON files')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for filtered JSON files')
    parser.add_argument('--image_dir', type=str, default=IMAGE_DIR,
                        help='Directory containing input images')
    parser.add_argument('--max_box_size', type=float, default=0.9,
                        help='Maximum allowed box size (width or height) in normalized coordinates (default: 0.8)')
    
    args = parser.parse_args()
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    IMAGE_DIR = args.image_dir
    
    print("Grounding DINO Object Filtering")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Number of GPUs: {args.num_gpus if args.num_gpus else 'all available'}")
    print(f"Max box size threshold: {args.max_box_size}")
    
    # Start processing
    process_batch_filtering(num_gpus=args.num_gpus, max_workers=args.max_workers, max_box_size=args.max_box_size)

if __name__ == "__main__":
    main()
