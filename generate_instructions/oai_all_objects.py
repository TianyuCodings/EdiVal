import os
import json
import base64
from openai import OpenAI
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import io
import random
import glob
from pathlib import Path
from dotenv import load_dotenv

MODEL_NAME = "gpt-4o"


PROMPT = """
You will be given an image. Your task is to identify and describe all clearly visible objects in the image in a structured JSON format.

Output Rules:

1. Each object must be listed as a key in the JSON, using the format:
   "{material} {color} {object name}"
   - If the material or color is unknown, omit that part from the key.
   - Do NOT include any visible text in the key.
   - Do NOT use "person" as an object name. Instead, describe each wearable item they have (e.g., "blue cotton shirt").

2. For each object, the value should be a dictionary with the following fields:
   - "object": the type of object (e.g., "shirt", "cup")
   - "color": the dominant color (e.g., "blue", "white") — use null if unknown
   - "material": the likely material (e.g., "cotton", "plastic", "metal") — use null if unknown
   - "text": any clearly visible printed or written text — use null if no text is visible or legible
   - "count": the number of visually indistinguishable instances — use 1 if it appears only once
   - "foreground": a boolean indicating whether the object is in the foreground (true) or background (false)

3. Do NOT include objects that:
   - Are too small to confidently describe
   - Are mostly occluded or incomplete
   - Appear only as background scenery (e.g., distant sky, ground, wall, floor)

4. At the end of the JSON, include an additional key called "All Objects". Its value must be a single string listing all identified object names, formatted as:
   "{material} {color} {object name}. {color} {object name}. {material} {object name}. {object name}."
   - Exclude the words "null", "None", or any field with unknown value
   - Separate object descriptions using a period and a space
   - Do NOT include any text content in this list

Example Output:

{
  "cotton blue shirt": {
    "object": "shirt",
    "color": "blue",
    "material": "cotton",
    "text": null,
    "count": 1,
    "foreground": true
  },
  "ceramic white cup": {
    "object": "cup",
    "color": "white",
    "material": "ceramic",
    "text": "GOOD DAY",
    "count": 1,
    "foreground": false
  },
  "leather bag": {
    "object": "bag",
    "color": null,
    "material": "leather",
    "text": null,
    "count": 2,
    "foreground": true
  },
  "red scarf": {
    "object": "scarf",
    "color": "red",
    "material": null,
    "text": null,
    "count": 1,
    "foreground": true
  },
  "All Objects": "cotton blue shirt. ceramic white cup. leather bag. red scarf."
}
"""

class OpenAIImageProcessor:
    def __init__(self, api_key: str):
        """Initialize the OpenAI client with API key."""
        base_url = (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_ENDPOINT")
        )
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")
        self.client = OpenAI(**client_kwargs)
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string with compression."""
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for JPEG compression)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Compress image to reduce size
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            compressed_data = output.getvalue()
            
            return base64.b64encode(compressed_data).decode('utf-8')
    
    def parse_json_response(self, response_content: str) -> Dict[str, Any]:
        """Parse JSON objects from the API response."""
        try:
            # Try to find JSON content in the response
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                return json.loads(response_content)
                
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return {"error": "Failed to parse JSON", "raw_content": response_content}
    
    def generate_json_from_image(self, image_path: str, image_index: str) -> Dict[str, Any]:
        """
        Generate JSON objects from an image using the OpenAI API.
        
        Args:
            image_path: Path to the input image
            image_index: Image index for the image
            
        Returns:
            Dictionary containing the API response and parsed results
        """
        try:
            # Encode the image
            base64_image = self.encode_image(image_path)
            
            # Prepare the message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{PROMPT}\n\nImage Index: {image_index}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Make the API call
            response = self.client.chat.completions.create(
                messages=messages,
                max_tokens=4096,
                temperature=0.7,
                top_p=1.0,
                model=MODEL_NAME,
            )
            
            # Extract the response content
            content = response.choices[0].message.content
            
            # Try to parse the JSON objects from the response
            json_objects = self.parse_json_response(content)
            
            return {
                "success": True,
                "image_index": image_index,
                "image_path": image_path,
                "raw_response": content,
                "parsed_json": json_objects,
                "usage": response.usage.dict() if response.usage else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "image_index": image_index,
                "image_path": image_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

def process_all_images(input_folder: str, output_folder: str, api_key: str):
    """
    Process all images in the input folder and save JSON outputs.
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to folder where JSON outputs will be saved
        api_key: OpenAI API key
    """
    # Initialize the processor
    processor = OpenAIImageProcessor(api_key)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files from input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Extract image index from filename
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Try to extract index from filename (assuming format like "123_something.jpg")
            if '_' in name_without_ext:
                image_index = name_without_ext.split('_')[0]
            else:
                image_index = name_without_ext
            
            # Generate JSON output filename
            output_filename = f"{image_index}_input_raw.json"
            output_path = os.path.join(output_folder, output_filename)
            
            # Skip if output already exists
            if os.path.exists(output_path):
                print(f"Skipping {filename} - output already exists")
                continue
            
            print(f"Processing {filename} (index: {image_index})...")
            
            # Process the image
            result = processor.generate_json_from_image(image_path, image_index)
            
            # Save the result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            if result["success"]:
                print(f"Successfully processed {filename}")
            else:
                print(f"Error processing {filename}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

def main():
    """Main function to run the image processing."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set up paths
    input_folder = "/scratch/EditVal/input_images_resize_512"
    output_folder = "/scratch/EditVal/generate_instructions/oai_all_objects"
    
    # Get API key from environment variable
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
    )
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        print("Please set it using one of these methods:")
        print("1. Export: export OPENAI_API_KEY='your_api_key_here'")
        print("2. Create .env file with: OPENAI_API_KEY=your_api_key_here")
        return
    
    # Process all images
    process_all_images(input_folder, output_folder, api_key)

if __name__ == "__main__":
    main()
