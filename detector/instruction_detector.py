from vllm import LLM, SamplingParams
import re
from PIL import Image
from groundingdino.util.inference import load_model, predict, load_image
import groundingdino.datasets.transforms as T
import numpy as np
import cv2
from typing import List, Dict, Any
import base64
import io
from .utils import load_resize_image

# VLM sampling parameters are now handled in _query_vlm function

VLM_COLOR_PROMPT = """
Look at the object in the image. Is the {object_name} {new_color}? Please answer only 'YES' or 'NO'.
"""

VLM_MATERIAL_PROMPT = """Is it possible that the {object_name} is made of {new_material}? Please answer only 'YES' or 'NO'.
"""

VLM_MATERIAL_PROMPT_STYLIZED = """Is it possible that the {object_name} is made of {new_material} in a stylized way? Please answer only 'YES' or 'NO'.
"""

VLM_TEXT_PROMPT = """
What text do you see in this image? Output only the text content, nothing else.
"""

VLM_BACKGROUND_PROMPT = """
Look at the background of this image. Does the background show [{background}]? Please answer only 'YES' or 'NO'.
"""

VLM_STRICT_COUNT_PROMPT = """
You are asked to count the number of an object. Please answer only the number. 
For example, if there are 3 dogs, you should answer '3'. If there is no dog, you should answer '0'.

Object name: {object_name}
Number of objects:
"""

VLM_REMOVE_CLOTHING_RELATED_PROMPT = """
Determine whether the image and instruction are related to editing the clothing (excluding jewelry) worn by a person. Only answer yes or no.
The editing instruction is: {instruction}.
"""

VLM_REMOVE_CLOTHING_INSTRUCTION_PROMPT = """
The first image is the original, and the second image reflects the changes made according to the editing instruction. Can you determine if the editing instruction was successfully applied?
The editing instruction is: {instruction}

Please respond with "yes" or "no."
"""

VLM_REPLACE_PROMPT = """
The first image is the original, and the second image reflects the changes made according to the editing instruction in subject replacement. Can you determine if the editing instruction was successfully applied?
The editing instruction is: {instruction}

Please respond with "yes" or "no."
"""

VLM_ADD_PROMPT = """
The first image is the original, and the second image reflects the changes made according to the editing instruction in subject addition. Can you determine if the editing instruction was successfully applied?
The editing instruction is: {instruction}

Please respond with "yes" or "no."
"""

VLM_DD_PROMPT = """
Compare image 1 (original) and image 2 (edited). Do you observe any object added in image 2 that was not present in image 1? Please answer only 'YES' or 'NO'.
"""

VLM_DD_ADD_OBJECT_PROMPT = """
Compare Image 1 (original) and Image 2 (edited). Did a {object_name} newly appear in Image 2 (absent in Image 1)? Answer only YES or NO. If uncertain, answer NO.
"""

VLM_DD_REPLACE_PROMPT = """
Compare Image 1 (original) and Image 2 (edited). Was a {object_name} replaced by a {new_object} in Image 2? Answer only YES or NO.
"""

VLM_DD_MATERIAL_ALTER_PROMPT = """
You are an expert atevaluating image editing success. 

I will show you two images: an original image and an edited image. The editing instruction is: "{instruction}"

Please analyze whether the edit successfully implements the instruction. Consider:
1. Was the requested change properly applied?
2. Is the edit accurate and complete?
3. Does the edited image look natural and coherent?

Respond with only "Yes" if the edit is successful, or "No" if it is not successful.
"""

VLM_DD_COLOR_ALTER_PROMPT = """
You are an expert atevaluating image editing success. 

I will show you two images: an original image and an edited image. The editing instruction is: "{instruction}"

Please analyze whether the edit successfully implements the instruction. Consider:
1. Was the requested change properly applied?
2. Is the edit accurate and complete?
3. Does the edited image look natural and coherent?

Respond with only "Yes" if the edit is successful, or "No" if it is not successful.
"""

LOCAL_TASK_TYPES = [
    "subject_replace", "subject_remove",
    "material_alter", "color_alter", "subject_add", "text_change",
    "position_change", "count_change"
]

GLOBAL_TASK_TYPES = [
    "background_change"
]


SUBJECT_REPLACE_FORMAT = "Replace [{object_name}] with [{new_object}]"

SUBJECT_REMOVE_FORMAT = "Remove [{object_name}]"
MATERIAL_ALTER_FORMAT = "Change the material of [{object_name}] to [{new_material}]"
COLOR_ALTER_FORMAT = "Change the color of [{object_name}] to [{new_color}]"

SUBJECT_ADD_FORMAT = "Add [{new_object}] on the [{position}] of [{reference_object}]"
SUBJECT_ADD_FORMAT_NO_REFERENCE = "Add [{object_name}]"

TEXT_CHANGE_FORMAT = "Replace the text '[{existing_text}]' on [{object_name}] with '[{new_text}]'"
TEXT_CHANGE_FORMAT_NO_EXISTING = "Add text '[{new_text}]' on the image"

POSITION_CHANGE_FORMAT = "Change the position of [target_object] to [position] of [reference_object]"
COUNT_CHANGE_FORMAT = "Change the count of [object_name] to [{target_count}]"
BACKGROUND_CHANGE_FORMAT = "Change the background to [{background}]"

POSITIONS = ["left", "right", "above", "below"]

# model is an already loaded model
def load_instruction_model(tensor_parallel_size=2):
    """Load the grounding dino model and VLM model"""
    # Load Grounding DINO model
    config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    grounding_dino_model = load_model(config_path, weights_path)
    
    # Load Qwen2-VL-7B-Instruct model
    vlm_model = LLM(
        model="Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 2},
        gpu_memory_utilization=0.5,
        tensor_parallel_size=tensor_parallel_size,
    )
    
    return grounding_dino_model, vlm_model

def _detect_single_object_from_img(model, image, target_object, return_all=False, threshold=0.3, delete_large_box=False):
    """
    Detect objects from a PIL image using GroundingDINO.
    
    Args:
        model: the grounding dino model
        image: PIL Image (RGB format)
        target_object: specific object to detect
        threshold: threshold for box score as well as text score
    
    Returns:
        Dict with detected objects information
    """
    if model is None:
        print("Warning: Grounding DINO model not available, returning empty detection")
        return {"label": [], "score": [], "box": [], "center": []}
    
    try:
        # Ensure PIL image is in RGB format
        if isinstance(image, Image.Image):
            image_pil = image.convert('RGB')
        else:
            # Fallback for numpy arrays (for backwards compatibility)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            image_pil = Image.fromarray(image_rgb)
        
        # Apply DINO's transforms directly to PIL image
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Transform the image
        image_tensor, _ = transform(image_pil, None)

        # Create text prompt from target object
        text_prompt = target_object.strip().lower().replace(".", "") + " ."
        print(f"Grounding DINO detection prompt: {text_prompt}")
        
        # Predict with DINO
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=threshold,
            text_threshold=threshold
        )
        
        # Convert results to our format
        h, w = image_pil.size[1], image_pil.size[0]  # height, width from PIL
        
        scores = []
        bbox_list = []
        centers = []
        normalized_centers = []
        labels = []

        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            # Convert normalized coordinates to pixel coordinates
            center_x, center_y, width, height = box
            normalized_centers.append([center_x, center_y])
            center_x, center_y, width, height = center_x * w, center_y * h, width * w, height * h
            
            # Convert to x1, y1, x2, y2 format for bbox (normalized coordinates)
            x1 = (center_x - width / 2) / w
            y1 = (center_y - height / 2) / h
            x2 = (center_x + width / 2) / w
            y2 = (center_y + height / 2) / h
            
            if delete_large_box:
                if abs(x1 - x2) > 0.98 and abs(y1 - y2) > 0.98:
                    continue
            
            scores.append(float(logit))
            bbox_list.append([x1, y1, x2, y2])
            centers.append([int(center_x), int(center_y)])
            labels.append(phrase)
        
        # only return the highest scoring detected object
        if scores:
            if return_all:
                result = {
                    "label": labels,
                    "score": scores,
                    "box": bbox_list,
                    "center": centers,
                    "normalized_center": normalized_centers
                }
            else:
                max_score_idx = scores.index(max(scores))
                result = {
                    "label": [labels[max_score_idx]],
                    "score": [scores[max_score_idx]],
                    "box": [bbox_list[max_score_idx]],
                    "center": [centers[max_score_idx]],
                    "normalized_center": [normalized_centers[max_score_idx]]
                }
        else:
            result = {"label": [], "score": [], "box": [], "center": [], "normalized_center": []}
        
        print(f"Detected {len(scores)} objects with target: {target_object}")
        return result
        
    except Exception as e:
        print(f"Error in object detection from image: {str(e)}")
        return {"label": [], "score": [], "box": [], "center": [], "normalized_center": []}


def _image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string for vLLM.
    
    Args:
        image: PIL Image
        
    Returns:
        str: Base64 encoded image data URL
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def _query_vlm(vlm_model, image: Image.Image, prompt: str) -> str:
    """
    Query the VLM model with an image and text prompt using vLLM.
    
    Args:
        vlm_model: The VLM model instance
        image: PIL Image
        prompt: Text prompt
        
    Returns:
        str: Model response
    """
    if vlm_model is None:
        print("Warning: VLM model not available")
        return "no"
    
    try:
        # Convert image to base64 for vLLM
        image_data = _image_to_base64(image)
        
        # Create conversation format for Qwen-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            min_tokens=1,
        )
        
        # Generate response
        outputs = vlm_model.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        # Extract response text
        response = outputs[0].outputs[0].text.strip()
        
        print(f"VLM Query: {prompt}")
        print(f"VLM Response: {response}")
        print("--------------------------------")
        
        return response.lower()  # Return lowercase for easier matching
        
    except Exception as e:
        print(f"Error querying VLM: {e}")
        import traceback
        traceback.print_exc()
        return "no"

def _query_vlm_2_image(vlm_model, image1: Image.Image, image2: Image.Image, prompt: str) -> str:
    """
    Query the VLM model with two images and a text prompt using vLLM.

    Args:
        vlm_model: The VLM model instance
        image1: First PIL Image
        image2: Second PIL Image
        prompt: Text prompt

    Returns:
        str: Model response
    """
    if vlm_model is None:
        print("Warning: VLM model not available")
        return "no"

    try:
        # Convert images to base64 for vLLM
        image_data1 = _image_to_base64(image1)
        image_data2 = _image_to_base64(image2)

        # Create conversation format for Qwen-VL with two images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data1}},
                    {"type": "image_url", "image_url": {"url": image_data2}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            min_tokens=1,
        )

        # Generate response
        outputs = vlm_model.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        # Extract response text
        response = outputs[0].outputs[0].text.strip()

        print(f"VLM Query (2 images): {prompt}")
        print(f"VLM Response: {response}")
        print("--------------------------------")

        return response.lower()  # Return lowercase for easier matching

    except Exception as e:
        print(f"Error querying VLM with 2 images: {e}")
        import traceback
        traceback.print_exc()
        return "no"

def _query_vlm_for_object_name(vlm_model, object_name: str) -> str:
    """
    Query the VLM model to extract the core object name from a potentially complex description.
    
    This function removes descriptive attributes like colors, materials, and other modifiers
    to return just the basic object name. For example:
    - "red wooden chair" -> "chair"
    - "large blue ceramic vase" -> "vase"
    - "small metal spoon" -> "spoon"
    
    """
    
    try:
        # Clean and prepare the input
        cleaned_input = object_name.strip()
        
        # Improved prompt with clear instructions and examples
        prompt = f"""Extract only the core object name from the following description: "{cleaned_input}"

                only remove descriptive attributes of color and material, do not remove other attributes:
                - Colors (red, blue, green, etc.)
                - Materials (wooden, metal, plastic, etc.)

                Return only the basic object name. Examples:
                - "red wooden chair" → "chair"
                - "large blue ceramic vase" → "vase"
                - "small metal spoon" → "spoon"
                - "metal gray back sign" → "back sign"

                Object name: {cleaned_input}
                Core object name:"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=20,
            min_tokens=1,
        )
        
        response = vlm_model.chat(messages=messages, sampling_params=sampling_params)
        
        if response and len(response) > 0 and response[0].outputs:
            result = response[0].outputs[0].text.strip().lower()
            
            # Basic validation - ensure we got a reasonable response
            if result:
                result = result.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
                print(f"VLM object name extraction: '{cleaned_input}' -> '{result}'")
                return result
            else:
                print(f"VLM returned invalid object name: '{result}', using original")
                return cleaned_input
        else:
            print("VLM returned empty response, using original object name")
            return cleaned_input
            
    except Exception as e:
        print(f"Error querying VLM for object name: {e}")
        import traceback
        traceback.print_exc()
        return object_name.strip()

def _enlarge_object(img, factor=1):
    """
    Enlarge the object in the image
    """
    width, height = img.size
    new_width = int(width * factor)
    new_height = int(height * factor)
    return img.resize((new_width, new_height))

def _crop_object(image, box, padding=0.0, original_size=False, double_small_object=False):
    """
    Crop the object from the image using the box

    Args:
        image: PIL Image
        box: bounding box coordinates [x1, y1, x2, y2] (normalized coordinates from grounding dino)
        padding: padding in percentage of the image size
        double_small_object: if True, double the small object to make it larger

    Returns:
        cropped PIL Image
    """
    # Convert normalized coordinates to pixel coordinates
    width, height = image.size
    if double_small_object:
        if abs(box[0] - box[2]) < 0.05 or abs(box[1] - box[3]) < 0.05:
            double_small_object = True
            padding = padding * 2
        else:
            double_small_object = False
    x1 = int(box[0] * width - padding * width)
    y1 = int(box[1] * height - padding * height)
    x2 = int(box[2] * width + padding * width)
    y2 = int(box[3] * height + padding * height)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    cropped_image = image.crop((x1, y1, x2, y2))
    if double_small_object:
        cropped_image = _enlarge_object(cropped_image, 2)

    if original_size:
        # return the cropped image and mask all other place such that image is the original size
        result = Image.new(image.mode, image.size, 0)  # Use same mode as original image to preserve color
        result.paste(cropped_image, (x1, y1))  # Paste at correct position
        return result
    else:
        return cropped_image

def _check_small_object(box):
    """
    Check if the object is small
    """
    return abs(box[0] - box[2]) < 0.05 or abs(box[1] - box[3]) < 0.05

def paraser_instruction(task_type, instruction):
    """
    Parse the instruction to get the important information based on specific format patterns

    Args:
        task_type: the type of the task
        instruction: the instruction to parse

    Returns:
        list of extracted content in the expected order for each task type
    """
    
    if task_type == "subject_replace":
        # Pattern: "Replace [{object_name}] with [{new_object}]"
        pattern = r"Replace \[([^\]]+)\] with \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]  # [object_name, new_object]
    
    elif task_type == "subject_remove":
        # Pattern: "Remove [{object_name}]"
        pattern = r"Remove \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1)]  # [object_name]
    
    elif task_type == "material_alter":
        # Pattern: "Change the material of [{object_name}] to [{new_material}]"
        pattern = r"Change the material of \[([^\]]+)\] to \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]  # [object_name, new_material]
    
    elif task_type == "color_alter":
        # Pattern: "Change the color of [{object_name}] to [{new_color}]"
        pattern = r"Change the color of \[([^\]]+)\] to \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]  # [object_name, new_color]
    
    elif task_type == "subject_add":
        # Pattern: "Add [{new_object}] on the [{position}] of [{reference_object}]"
        pattern1 = r"Add \[([^\]]+)\] on the \[([^\]]+)\] of \[([^\]]+)\]"
        match1 = re.search(pattern1, instruction)
        if match1:
            return [match1.group(1), match1.group(2), match1.group(3)]  # [new_object, position, reference_object]
        
        # Pattern: "Add [{object_name}]"
        pattern2 = r"Add \[([^\]]+)\]"
        match2 = re.search(pattern2, instruction)
        if match2:
            return [match2.group(1)]  # [object_name]
    
    elif task_type == "text_change":
        # Pattern: "Replace the text '[{existing_text}]' on [{object_name}] with '[{new_text}]'"
        pattern1 = r"Replace the text '\[([^\]]+)\]' on \[([^\]]+)\] with '\[([^\]]+)\]'"
        match1 = re.search(pattern1, instruction)
        if match1:
            return [match1.group(1), match1.group(2), match1.group(3)]  # [existing_text, object_name, new_text]
        
        # Pattern: "Add text '[{new_text}]' on the image"
        pattern2 = r"Add text '\[([^\]]+)\]' on the image"
        match2 = re.search(pattern2, instruction)
        if match2:
            return [match2.group(1)]  # [new_text]
    
    elif task_type == "position_change":
        # Pattern: "Change the position of [target_object] to [position] of [reference_object]"
        pattern = r"Change the position of \[([^\]]+)\] to \[([^\]]+)\] of \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2), match.group(3)]  # [target_object, position, reference_object]
    
    elif task_type == "count_change":
        # Pattern: "Change the count of [object_name] to [{target_count}]"
        pattern = r"Change the count of \[([^\]]+)\] to \[([^\]]+)\]"
        match = re.search(pattern, instruction)
        if match:
            return [match.group(1), match.group(2)]  # [object_name, target_count]
    
    elif task_type == "background_change":
        # not use formatted instruction for background change, use natural language instruction instead
        # Pattern: "Change the background to [{background}]"
        # Pattern: "Change the background to [{background}], remain the [{objects}] unchanged"
        pattern1 = r"Change the background to ([^,]+), remain the (.+) unchanged"
        match1 = re.search(pattern1, instruction)
        if match1:
            return [match1.group(1)] + match1.group(2).split(",")  # [background, remain_objects]
        
        # Pattern: "Change the background to [{background}]"
        pattern2 = r"Change the background to \[([^\]]+)\]"
        match2 = re.search(pattern2, instruction)
        if match2:
            return [match2.group(1)]  # [background]
    else:
        raise ValueError(f"Invalid instruction: {instruction}")

def _calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] normalized coordinates
        box2: [x1, y1, x2, y2] normalized coordinates
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        intersection = 0.0
    else:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Calculate IoU
    if union == 0:
        return 0.0
    return intersection / union

def _a_contains_b(box1, box2):
    """
    Check if box1 contains box2
    """
    return (box1[0] <= box2[0] and box1[1] <= box2[1] and box1[2] >= box2[2] and box1[3] >= box2[3])

def _count_object(boxes, iou_threshold=0.8):
    # boxes: a list of bounding boxes, each box is a list of 4 numbers [x1, y1, x2, y2]
    # delete boxes until all pairs of boxes have iou < iou_threshold
    if len(boxes) == 0:
        return 0
    
    # Make a copy to avoid modifying the original list
    boxes = boxes.copy()
    
    while True:
        if len(boxes) <= 1:
            return len(boxes)
        
        # Track if any boxes were removed in this iteration
        boxes_removed = False
        
        # Check all pairs of boxes
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                if _calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                    # Remove the box with lower confidence (or just remove the second one)
                    boxes.pop(j)
                    boxes_removed = True
                    # Don't increment j since we removed an element
                else:
                    j += 1
            i += 1
        
        # If no boxes were removed in this iteration, we're done
        if not boxes_removed:
            return len(boxes)


def _verify_subject_add(src_img, target_img, instruction, grounding_model, vlm_model, position_threshold=0.03, output_reason=False):
    # Parse instruction to extract new object (and optional positional relation)
    parsed_info = paraser_instruction("subject_add", instruction)
    if parsed_info is None:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    if len(parsed_info) == 1:
        new_object = parsed_info[0]
    elif len(parsed_info) == 3:
        new_object = parsed_info[0]
        position = parsed_info[1]
        ref_object = parsed_info[2]
    else:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0

    # VLM pre-check: confirm the specified object appears newly in image 2
    try:
        if vlm_model is not None:
            vlm_response = _query_vlm_2_image(
                vlm_model,
                src_img,
                target_img,
                VLM_DD_ADD_OBJECT_PROMPT.format(object_name=new_object),
            )
            if "yes" not in vlm_response.strip().lower():
                reason = f"VLM pre-check: [{new_object}] not observed as newly added. VLM response: '{vlm_response}'"
                return (0, reason) if output_reason else 0
    except Exception as e:
        # Fail open on VLM gate (don't block if VLM unavailable); continue with detector-based checks
        print(f"Warning: VLM pre-check failed in _verify_subject_add: {e}")

    
    # detect the object in the image (PIL images passed directly)
    if len(parsed_info) == 3:
        src_object = _detect_single_object_from_img(grounding_model, src_img, ref_object, return_all=True)
    target_object = _detect_single_object_from_img(grounding_model, target_img, new_object, return_all=True)

    if len(parsed_info) == 1:
        if len(target_object.get("score", [])) > 0:
            # there is at least one object detected by the model
            score = 1
            reason = f"Successfully added [{new_object}] to the image. Detected {len(target_object.get('score', []))} instance(s)."
        else:
            # no object detected by the model
            score = 0
            reason = f"Failed to add [{new_object}] to the image. No instances detected in target image."
    elif len(parsed_info) == 3:
        if len(src_object.get("score", [])) > 0 and len(target_object.get("score", [])) > 0:
            # make sure the src_img has the ref_object (may not have reference object since src_img can also be edited img and not able to generate reference object)
            ref_obj_centers = src_object.get("normalized_center", [])
            target_obj_centers = target_object.get("normalized_center")

            if not ref_obj_centers or not target_obj_centers:
                score = 0
                reason = f"Failed to get object centers. Reference object detected: {len(ref_obj_centers)}, Target object deteced: {len(target_obj_centers)}"
            
            else:
                # Check position relative to all reference objects
                # If the target object is in correct position relative to ANY reference object, return success
                position_satisfied = False
                successful_ref_center = None
                
                for ref_obj_center in ref_obj_centers:
                    for target_obj_center in target_obj_centers:
                        if position == "left":
                            if target_obj_center[0] - ref_obj_center[0] < -position_threshold:
                                position_satisfied = True
                                successful_ref_center = ref_obj_center
                                successful_target_center = target_obj_center
                                break
                        elif position == "right":
                            if target_obj_center[0] - ref_obj_center[0] > position_threshold:
                                position_satisfied = True
                                successful_ref_center = ref_obj_center
                                successful_target_center = target_obj_center
                                break
                        elif position == "above":
                            if target_obj_center[1] - ref_obj_center[1] < -position_threshold:
                                position_satisfied = True
                                successful_ref_center = ref_obj_center
                                successful_target_center = target_obj_center
                                break
                        elif position == "below":
                            if target_obj_center[1] - ref_obj_center[1] > position_threshold:
                                position_satisfied = True
                                successful_ref_center = ref_obj_center
                                successful_target_center = target_obj_center
                                break
                
                if position_satisfied:
                    score = 1
                    reason = f"Successfully added [{new_object}] to the {position} of [{ref_object}]. Target position: {successful_target_center} (checked {len(target_obj_centers)} target objects), Reference position: {successful_ref_center} (checked {len(ref_obj_centers)} reference objects)"
                elif position in ["left", "right", "above", "below"]:
                    score = 0
                    reason = f"Failed to add [{new_object}] to the {position} of [{ref_object}]. Checked {len(target_obj_centers)} target objects, {new_object} is not {position} of any reference positions: {ref_object} (checked {len(ref_obj_centers)} reference objects)"
                else:
                    score = 0
                    reason = f"Invalid position: {position}"
        else:
            score = 0
            src_detected = len(src_object.get("score", []))
            target_detected = len(target_object.get("score", []))
            reason = f"Failed to detect required objects. Reference object [{ref_object}] detected: {src_detected > 0}, Target object [{new_object}] detected: {target_detected > 0}"
    else:
        score = 0
        reason = f"Invalid instruction format: {instruction}"
    
    return (score, reason) if output_reason else score


def _verify_subject_remove(src_img, target_img, instruction, grounding_model, strict=True, output_reason=False):
    parsed_info = paraser_instruction("subject_remove", instruction)
    if parsed_info is None:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    if len(parsed_info) == 1:
        object_name = parsed_info[0]
    else:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    
    # detect the object in the image (PIL images passed directly)
    src_object = _detect_single_object_from_img(grounding_model, src_img, object_name, threshold=0.35, delete_large_box=True)
    if len(src_object.get("score", [])) == 0:
        score = 0
        reason = f"Failed to detect [{object_name}] in source image."
    else:
        src_box = src_object.get("box", [])[0]
        logit = src_object.get("score", [])[0]
        threshold = 0.35

        #cropped_target_object = _crop_object(target_img, src_box, padding=0.05, original_size=True, double_small_object=True)
        target_object = _detect_single_object_from_img(grounding_model, target_img, object_name, threshold=threshold, delete_large_box=True)
        # print(cropped_target_object_dino)
        # print(logit)
        # import matplotlib.pyplot as plt
        # plt.imshow(cropped_target_object)
        # plt.show()
        if len(target_object.get("score", [])) == 0:
            score = 1
            reason = f"Successfully removed [{object_name}] from the image. Source target box: {src_box}."
            return (score, reason) if output_reason else score
        if (_calculate_iou(src_box, target_object.get("box", [])[0]) < 0.2):
            # two objects are far away, not the same one
            score = 1
            reason = f"Successfully removed [{object_name}] from the image. Source target box: {src_box}."
            return (score, reason) if output_reason else score
        

        # corner cases that grounding dino will confuse.
        if "hat" in object_name.lower():
            target_object = _detect_single_object_from_img(grounding_model, target_img, "hat", threshold=threshold)
            if len(target_object.get("score", [])) == 0:
                score = 1
                reason = f"Successfully removed [{object_name}] from the image. Source target box: {src_box}."
                return (score, reason) if output_reason else score
        if "bird" in object_name.lower():
            target_object = _detect_single_object_from_img(grounding_model, target_img, "bird", threshold=threshold)
            if len(target_object.get("score", [])) == 0:
                score = 1
                reason = f"Successfully removed [{object_name}] from the image. Source target box: {src_box}."
                return (score, reason) if output_reason else score
        
        score = 0
        reason = f"Failed to remove [{object_name}] from the image. Source target box: {src_box}."
    return (score, reason) if output_reason else score

def _verify_subject_replace(src_img, target_img, instruction, grounding_model, vlm_model=None, output_reason=False):
    parsed_info = paraser_instruction("subject_replace", instruction)
    if parsed_info is None:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    if len(parsed_info) == 2:
        object_name = parsed_info[0]
        new_object = parsed_info[1]
    else:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    
    # VLM pre-check to quickly filter obvious non-replacements
    try:
        if vlm_model is not None:
            vlm_resp = _query_vlm_2_image(
                vlm_model,
                src_img,
                target_img,
                VLM_DD_REPLACE_PROMPT.format(object_name=object_name, new_object=new_object),
            )
            if "yes" not in vlm_resp.strip().lower():
                reason = f"VLM pre-check: replacement not detected from [{object_name}] to [{new_object}]. VLM response: '{vlm_resp}'"
                return (0, reason) if output_reason else 0
    except Exception as e:
        print(f"Warning: VLM pre-check failed in _verify_subject_replace: {e}")

    # Remove 's' or 'es' at the end of the string to get singular form
    if object_name.endswith('es'):
        object_name = object_name[:-2]
    elif object_name.endswith('s'):
        object_name = object_name[:-1]
    
    if new_object.endswith('es'):
        new_object = new_object[:-2]
    elif new_object.endswith('s'):
        new_object = new_object[:-1]

    src_object = _detect_single_object_from_img(grounding_model, src_img, object_name, return_all=True)
    target_object = _detect_single_object_from_img(grounding_model, target_img, new_object, return_all=True)

    if len(src_object.get("score", [])) > 0 and len(target_object.get("score", [])) > 0:
        src_object_boxes = src_object.get("box", [])
        target_object_boxes = target_object.get("box", [])

        # Success if any IoU between a src box (old object) and target box (new object) is > 0
        success = False
        best_iou = 0.0
        for src_box in src_object_boxes:
            for target_box in target_object_boxes:
                iou = _calculate_iou(src_box, target_box)
                best_iou = max(best_iou, iou)
                if iou > 0.0:
                    success = True
                    break
            if success:
                break
        if success:
            score = 1
            reason = f"Successfully replaced [{object_name}] with [{new_object}]. Max IoU between src [{object_name}] and target [{new_object}] boxes: {best_iou:.3f} (> 0)."
        else:
            score = 0
            reason = f"Failed to replace [{object_name}] with [{new_object}]. All src/target box IoUs are 0. Max IoU: {best_iou:.3f}."
    else:
        score = 0
        src_detected = len(src_object.get("score", []))
        target_detected = len(target_object.get("score", []))
        reason = f"Failed to detect required objects. Source object [{object_name}] detected: {src_detected > 0}, Target object [{new_object}] detected: {target_detected > 0}"

    return (score, reason) if output_reason else score

def _verify_position_change(src_img, target_img, instruction, grounding_model, vlm_model, output_reason=False):
    parsed_info = paraser_instruction("position_change", instruction)
    if parsed_info is None or len(parsed_info) != 3:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    
    target_object = parsed_info[0]
    position = parsed_info[1]
    reference_object = parsed_info[2]
    
    # Check if the count of objects changed between source and target images.
    # Some model will add an object to hack the position change detection.

    cleaned_target_object = _query_vlm_for_object_name(vlm_model, target_object)
    cleaned_reference_object = _query_vlm_for_object_name(vlm_model, reference_object)
    src_ref_object = _detect_single_object_from_img(grounding_model, src_img, cleaned_reference_object, return_all=True, threshold=0.4)
    src_target_obj = _detect_single_object_from_img(grounding_model, src_img, cleaned_target_object, return_all=True, threshold=0.4)
    target_ref_object = _detect_single_object_from_img(grounding_model, target_img, cleaned_reference_object, return_all=True, threshold=0.4)
    target_target_obj = _detect_single_object_from_img(grounding_model, target_img, cleaned_target_object, return_all=True, threshold=0.4)
    
    src_ref_count = _count_object(src_ref_object.get("box", []))
    target_ref_count = _count_object(target_ref_object.get("box", []))
    src_target_count = _count_object(src_target_obj.get("box", []))
    target_target_count = _count_object(target_target_obj.get("box", []))

    if src_ref_count != target_ref_count:
        reason = f"Does not remain object count. Count of reference object [{reference_object}] changed from {src_ref_count} to {target_ref_count}"
        return (0, reason) if output_reason else 0
    
    if src_target_count != target_target_count:
        reason = f"Does not remain object count. Count of target object [{target_object}] changed from {src_target_count} to {target_target_count}"
        return (0, reason) if output_reason else 0

    # Continue with existing logic - use target image detections for position checking
    ref_object = _detect_single_object_from_img(grounding_model, target_img, reference_object, return_all=False)
    target_obj = _detect_single_object_from_img(grounding_model, target_img, target_object, return_all=False)

    if len(ref_object.get("score", [])) > 0 and len(target_obj.get("score", [])) > 0:
        ref_object_box = ref_object.get("box", [])[0]
        target_object_box = target_obj.get("box", [])[0]
        if _a_contains_b(ref_object_box, target_object_box):
            score = 0
            reason = f"Reference object {reference_object} contains target object {target_object}. It is unlikely to be a position change. May cause by grounding dino's false positive."
            return (0, reason) if output_reason else 0
        if _a_contains_b(target_object_box, ref_object_box):
            score = 0
            reason = f"Target object {target_object} contains reference object {reference_object}. It is unlikely to be a position change. May cause by grounding dino's false positive."
            return (0, reason) if output_reason else 0
        
        # if the ref_object_boxes and target_object_boxes are not overlapping, then it is a position change
        ref_object_center = ref_object.get("center")[0] if ref_object.get("center") else None
        target_object_center = target_obj.get("center")[0] if target_obj.get("center") else None

        if ref_object_center is not None and target_object_center is not None:
            if position == "left":
                if target_object_center[0] < ref_object_center[0]:
                    score = 1
                    reason = f"Successfully moved [{target_object}] to the left of [{reference_object}]. Target position: {target_object_center}, Reference position: {ref_object_center}"
                else:
                    score = 0
                    reason = f"Failed to move [{target_object}] to the left of [{reference_object}]. Target position: {target_object_center} is not left of reference position: {ref_object_center}"
            elif position == "right":
                if target_object_center[0] > ref_object_center[0]:
                    score = 1
                    reason = f"Successfully moved [{target_object}] to the right of [{reference_object}]. Target position: {target_object_center}, Reference position: {ref_object_center}"
                else:
                    score = 0
                    reason = f"Failed to move [{target_object}] to the right of [{reference_object}]. Target position: {target_object_center} is not right of reference position: {ref_object_center}"
            elif position == "above":
                if target_object_center[1] < ref_object_center[1]:
                    score = 1
                    reason = f"Successfully moved [{target_object}] above [{reference_object}]. Target position: {target_object_center}, Reference position: {ref_object_center}"
                else:
                    score = 0
                    reason = f"Failed to move [{target_object}] above [{reference_object}]. Target position: {target_object_center} is not above reference position: {ref_object_center}"
            elif position == "below":
                if target_object_center[1] > ref_object_center[1]:
                    score = 1
                    reason = f"Successfully moved [{target_object}] below [{reference_object}]. Target position: {target_object_center}, Reference position: {ref_object_center}"
                else:
                    score = 0
                    reason = f"Failed to move [{target_object}] below [{reference_object}]. Target position: {target_object_center} is not below reference position: {ref_object_center}"
            else:
                score = 0
                reason = f"Invalid position: {position}"
        else:
            score = 0
            reason = f"Failed to get object centers. Reference object center: {ref_object_center}, Target object center: {target_object_center}"
    else:
        score = 0
        ref_detected = len(ref_object.get("score", []))
        target_detected = len(target_obj.get("score", []))
        reason = f"Failed to detect required objects. Reference object [{reference_object}] detected: {ref_detected > 0}, Target object [{target_object}] detected: {target_detected > 0}"

    return (score, reason) if output_reason else score


def _verify_count_change(src_img, target_img, instruction, grounding_model, vlm_model, output_reason=False):
    parsed_info = paraser_instruction("count_change", instruction)
    if parsed_info is None:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    if len(parsed_info) == 2:
        object_name = parsed_info[0]
        target_count = parsed_info[1]
    else:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0

    # detect the object in the image
    src_object = _detect_single_object_from_img(grounding_model, src_img, object_name)
    target_object = _detect_single_object_from_img(grounding_model, target_img, object_name, return_all=True)

    target_object_count = _count_object(target_object.get("box", []))

    if len(src_object.get("score", [])) > 0 and len(target_object.get("score", [])) > 0:
        if int(target_object_count) == int(target_count):
            score = 1
            reason = f"Successfully changed count of [{object_name}] to {target_count}. Actual count in target image: {target_object_count}"
        else:
            score = 0
            reason = f"Failed to change count of [{object_name}] to {target_count}. Actual count in target image: {target_object_count}"
            
            # # ask for vlm to double check. there is a srtict version
            # response = _query_vlm(vlm_model, target_img, VLM_STRICT_COUNT_PROMPT.format(object_name=object_name))
            # if int(response.strip().lower()) == int(target_count):
            #     score = 1
            #     reason = f"Successfully changed count of [{object_name}] to {target_count}. Grounding DINO count: {target_object_count}, VLM count: '{response}'"
            # else:
            #     score = 0
            #     reason = f"Failed to change count of [{object_name}] to {target_count}. Grounding DINO count: {target_object_count}, VLM count: '{response}'"
            

            # response = _query_vlm(vlm_model, target_img, VLM_TOLERANT_COUNT_PROMPT.format(object_name=object_name, count=target_count))
            # if "yes" in response.strip().lower():
            #     score = 1
            #     reason = f"Successfully changed count of [{object_name}] to {target_count}. Grounding DINO count: {target_object_count}, VLM response: '{response}'"
            # else:
            #     score = 0
            #     reason = f"Failed to change count of [{object_name}] to {target_count}. Grounding DINO count: {target_object_count}, VLM response: '{response}'"
    else:
        score = 0
        src_detected = len(src_object.get("score", []))
        target_detected = len(target_object.get("score", []))
        reason = f"Failed to detect [{object_name}] in images. Source detected: {src_detected > 0}, Target detected: {target_detected > 0}"

    return (score, reason) if output_reason else score


def _verify_color_alter(src_img, target_img, instruction, grounding_model, vlm_model, output_reason=False):
    parsed_info = paraser_instruction("color_alter", instruction)
    if parsed_info is None or len(parsed_info) != 2:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    object_name = parsed_info[0]
    new_color = parsed_info[1]

    try:
        response = _query_vlm_2_image(
            vlm_model,
            src_img,
            target_img,
            VLM_DD_COLOR_ALTER_PROMPT.format(instruction=instruction),
        )
        last_response = response
        if "yes" in response.strip().lower():
            score = 1
            reason = f"Successfully changed color of [{object_name}] to [{new_color}]. VLM response: '{response}'"
            return (score, reason) if output_reason else score
    except Exception as e:
        print(f"Warning: VLM color alter check failed for a crop: {e}")

    score = 0
    reason = (
        f"Failed to change color of [{object_name}] to [{new_color}] per VLM. VLM response: '{last_response}'"
        if last_response is not None else
        f"VLM could not confirm color change for [{object_name}] to [{new_color}]"
    )
    return (score, reason) if output_reason else score

def _verify_material_alter(src_img, target_img, instruction, grounding_model, vlm_model, output_reason=False):
    parsed_info = paraser_instruction("material_alter", instruction)
    if parsed_info is None or len(parsed_info) != 2:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    object_name, new_material = parsed_info[0], parsed_info[1]

    # Only use VLM on the full, uncropped images to judge material change
    try:
        response = _query_vlm_2_image(
            vlm_model,
            src_img,
            target_img,
            VLM_DD_MATERIAL_ALTER_PROMPT.format(instruction=instruction),
        )
        if "yes" in response.strip().lower():
            score = 1
            reason = f"Material of [{object_name}] changed to [{new_material}]. VLM response: '{response}'"
        else:
            score = 0
            reason = f"Material of [{object_name}] not changed to [{new_material}] per VLM. VLM response: '{response}'"
        return (score, reason) if output_reason else score
    except Exception as e:
        print(f"Error during VLM material alter check: {e}")
        reason = f"VLM error while checking material change for [{object_name}] to [{new_material}]"
        return (0, reason) if output_reason else 0


def _verify_text_change(src_img, target_img, instruction, grounding_model, vlm_model, output_reason=False):
    parsed_info = paraser_instruction("text_change", instruction)
    if parsed_info is None:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    if len(parsed_info) == 3:
        existing_text = parsed_info[0]
        object_name = parsed_info[1]
        new_text = parsed_info[2]
    elif len(parsed_info) == 1:
        new_text = parsed_info[0]
    else:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    
    # detect the object in the image
    if len(parsed_info) == 3:
        src_object = _detect_single_object_from_img(grounding_model, src_img, object_name)
        target_object = _detect_single_object_from_img(grounding_model, target_img, object_name)

        if len(src_object.get("score", [])) > 0 and len(target_object.get("score", [])) > 0:
            # make sure the object is remained and only change the text
            src_object_boxes = src_object.get("box", [])
            if len(src_object_boxes) > 0:
                # use the first detected object's box
                src_object_box = src_object_boxes[0]
                cropped_target_object = _crop_object(target_img, src_object_box)

                response = _query_vlm(vlm_model, cropped_target_object, VLM_TEXT_PROMPT)

                if new_text.lower() in response.lower():
                    score = 1
                    reason = f"Successfully changed text on [{object_name}] to '[{new_text}]'. VLM detected text: '{response}'"
                else:
                    score = 0
                    reason = f"Failed to change text on [{object_name}] to '[{new_text}]'. VLM detected text: '{response}'"
            else:
                score = 0
                reason = f"Failed to get bounding box for [{object_name}] in source image"
        else:
            score = 0
            src_detected = len(src_object.get("score", []))
            target_detected = len(target_object.get("score", []))
            reason = f"Failed to detect [{object_name}] in images. Source detected: {src_detected > 0}, Target detected: {target_detected > 0}"
    elif len(parsed_info) == 1:
        # make sure the image has the new text
        response = _query_vlm(vlm_model, target_img, VLM_TEXT_PROMPT)

        if new_text.lower() in response.lower():
            score = 1
            reason = f"Successfully added text '[{new_text}]' to the image. VLM detected text: '{response}'"
        else:
            score = 0
            reason = f"Failed to add text '[{new_text}]' to the image. VLM detected text: '{response}'"
    else:
        score = 0
        reason = f"Invalid instruction format: {instruction}"

    return (score, reason) if output_reason else score


def _verify_background_change(src_img, target_img, instruction, grounding_model, vlm_model, detect_foreground=True, output_reason=False):
    parsed_info = paraser_instruction("background_change", instruction)
    if parsed_info is None:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    if len(parsed_info) == 1:
        background = parsed_info[0]
    elif len(parsed_info) > 1:
        background = parsed_info[0]
        remain_objects = parsed_info[1:]
        # check if the remain objects are remained in the image
        if detect_foreground:
            for object_name in remain_objects:
                object_detected = _detect_single_object_from_img(grounding_model, target_img, object_name, return_all=False, threshold=0.25)
                if len(object_detected.get("score", [])) == 0:
                    reason = f"Failed to detect [{object_name}] in target image"
                    return (0, reason) if output_reason else 0
                x1, y1, x2, y2 = object_detected.get("box", [])[0]
                if abs(x2 - x1) > 0.9 and abs(y2 - y1) > 0.9:
                    # foreground object is impossible to be a whole image
                    # filter out the grounding dino false positive detection
                    reason = f"Failed to detect [{object_name}] in target image. The object is too large to be a whole image. A false positive detection by grounding dino."
                    return (0, reason) if output_reason else 0
        else:
            pass
    else:
        reason = f"Invalid instruction format: {instruction}"
        return (0, reason) if output_reason else 0
    
    # detect the object in the image
    response = _query_vlm(vlm_model, target_img, VLM_BACKGROUND_PROMPT.format(background=background))

    if "yes" in response:
        score = 1
        reason = f"Successfully changed background to [{background}]. VLM response: '{response}'"
    else:
        score = 0
        response = _query_vlm(vlm_model, target_img, VLM_BACKGROUND_PROMPT.format(background=background).replace("Please answer only 'YES' or 'NO'", "Please give some reason why you think it is not"))
        reason = f"Failed to change background to [{background}]. VLM response: '{response}'"

    return (score, reason) if output_reason else score


def intruction_following_detection(formatted_instruction, instruction, task_type, src_img, target_img, grounding_model, vlm_model, output_reason=False):
    """
    Detect the instruction following detection

    Args:
        formatted_instruction: the formatted instruction string
        instruction: the instruction string
        task_type: the type of the task (e.g., "subject_add", "subject_remove", etc.)
        src_img: source image (PIL Image)
        target_img: target image (PIL Image)
        grounding_model: the grounding dino model
        vlm_model: the vision language model
        output_reason: whether to output the reason for success/failure

    Returns:
        score: 1 if instruction is followed correctly, 0 otherwise
        reason (optional): explanation of why the task succeeded or failed
    """
    try:
        if task_type == "subject_add":
            return _verify_subject_add(src_img, target_img, formatted_instruction, grounding_model, vlm_model, output_reason=output_reason)
        
        elif task_type == "subject_remove":
            return _verify_subject_remove(src_img, target_img, formatted_instruction, grounding_model, vlm_model, output_reason=output_reason)
        
        elif task_type == "subject_replace":
            return _verify_subject_replace(src_img, target_img, formatted_instruction, grounding_model, vlm_model, output_reason=output_reason)
        
        elif task_type == "position_change":
            return _verify_position_change(src_img, target_img, formatted_instruction, grounding_model, vlm_model, output_reason=output_reason)
        
        elif task_type == "count_change":
            return _verify_count_change(src_img, target_img, formatted_instruction, grounding_model, vlm_model, output_reason=output_reason)
        
        elif task_type == "color_alter":
            return _verify_color_alter(src_img, target_img, formatted_instruction, grounding_model, vlm_model, output_reason=output_reason)
        
        elif task_type == "material_alter":
            return _verify_material_alter(src_img, target_img, formatted_instruction, grounding_model, vlm_model, output_reason=output_reason)
        
        elif task_type == "text_change":
            return _verify_text_change(src_img, target_img, formatted_instruction, grounding_model, vlm_model, output_reason=output_reason)
        
        elif task_type == "background_change":
            return _verify_background_change(src_img, target_img, instruction, grounding_model, vlm_model, output_reason=output_reason)
        
        else:
            reason = f"Unknown task type: {task_type}"
            return (0, reason) if output_reason else 0
            
    except Exception as e:
        print(f"Error in instruction following detection: {str(e)}")
        reason = f"Error during evaluation: {str(e)}"
        return (0, reason) if output_reason else 0


def evaluate_instruction_following(src_image, target_image, formatted_instruction, instruction, task_type, grounding_model=None, vlm_model=None, output_reason=False):
    """
    Complete pipeline to evaluate instruction following
    
    Args:
        src_image: source image - can be a file path (str) or PIL Image
        target_image: target image - can be a file path (str) or PIL Image
        formatted_instruction: the instruction string (e.g., "Add [dog] on the [left] of [tree]")
        instruction: the instruction string (e.g., "Add dog on the left of tree")
        task_type: the type of task (e.g., "subject_add")
        grounding_model: pre-loaded grounding model (optional)
        vlm_model: pre-loaded VLM model (optional)
        output_reason: whether to output the reason for success/failure
        
    Returns:
        score: 1 if instruction is followed correctly, 0 otherwise
        reason (optional): explanation of why the task succeeded or failed
        
    Example:
        score = evaluate_instruction_following(
            "source.jpg", 
            "target.jpg", 
            "Add [dog] on the [left] of [tree]", 
            "subject_add"
        )
        
        score, reason = evaluate_instruction_following(
            pil_src_image, 
            pil_target_image, 
            "Add [dog] on the [left] of [tree]", 
            "subject_add",
            output_reason=True
        )
    """

    if grounding_model is None:
        grounding_model, vlm_model = load_instruction_model()
    
    src_img_pil = load_resize_image(src_image)
    target_img_pil = load_resize_image(target_image)
    
    # Evaluate instruction following (the detection functions will handle GroundingDINO's load_image internally)
    result = intruction_following_detection(
        formatted_instruction, instruction, task_type, src_img_pil, target_img_pil, grounding_model, vlm_model, output_reason=output_reason
    )
    
    return result
