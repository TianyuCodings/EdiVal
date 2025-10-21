import numpy as np
import torch
from PIL import Image
import cv2

def _make_json_serializable(obj):
    """Convert numpy and tensor objects to JSON serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def load_resize_image(image, target_size=(512, 512), return_cv2=False):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError("image must be a file path (str) or PIL Image")
    if image.size != target_size:
        print(f"Warning: Resizing image from {image.size} to {target_size}")
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    if return_cv2:
        return image, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image