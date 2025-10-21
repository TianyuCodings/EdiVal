import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import List
import matplotlib.pyplot as plt
from .utils import load_resize_image

from groundingdino.util.inference import load_model, predict, load_image
import groundingdino.datasets.transforms as T
import torchvision.transforms as transforms
try:
    from transformers import AutoImageProcessor, AutoModel
except Exception:
    AutoImageProcessor = None
    AutoModel = None


def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize a tensor along the last dimension."""
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)


def _masked_global_cosine(tokens1: torch.Tensor, w1: torch.Tensor,
                          tokens2: torch.Tensor, w2: torch.Tensor,
                          weight_thresh: float = 0.0) -> float:
    """
    Compute cosine similarity between two sets of tokens using soft per-token weights.

    Args:
        tokens1, tokens2: shape (1, N, D)
        w1, w2: shape (1, N) with values in [0, 1]
        weight_thresh: optional threshold; if > 0, tokens with weight < threshold
                       are effectively zeroed out.

    Returns:
        Scalar similarity (float).
    """
    if weight_thresh > 0:
        m1 = (w1 >= weight_thresh).float()
        m2 = (w2 >= weight_thresh).float()
        w1, w2 = w1 * m1, w2 * m2

    # Normalize token features before pooling
    t1 = _l2norm(tokens1.squeeze(0))  # (N, D)
    t2 = _l2norm(tokens2.squeeze(0))  # (N, D)

    # Weighted average (masked GAP)
    w1e = w1.squeeze(0).unsqueeze(-1)  # (N,1)
    w2e = w2.squeeze(0).unsqueeze(-1)
    f1 = (t1 * w1e).sum(dim=0) / (w1e.sum() + 1e-6)  # (D,)
    f2 = (t2 * w2e).sum(dim=0) / (w2e.sum() + 1e-6)  # (D,)

    sim = torch.nn.functional.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
    return float(sim)

def load_consistency_model(device=None, load_grounding_dino=False):
    """Load DINOv3 model (Hugging Face) for similarity and optionally GroundingDINO for detection.

    DINOv3 path: facebook/dinov3-vitb16-pretrain-lvd1689m
    Returns:
        - if load_grounding_dino is False: (dino_model, dino_processor)
        - else: ((dino_model, dino_processor), grounding_dino_model)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if AutoImageProcessor is None or AutoModel is None:
        raise ImportError(
            "transformers is required for DINOv3. Please install transformers and try again."
        )

    dino_processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    dino_model = AutoModel.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    dino_model.eval().to(device)

    if load_grounding_dino:
        config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weights_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
        grounding_dino_model = load_model(config_path, weights_path)
        return (dino_model, dino_processor), grounding_dino_model

    return dino_model, dino_processor


def _detect_multiple_objects_from_img(model, image, objects:List[str]):
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
        if not isinstance(objects, list):
            raise ValueError("objects should be a list of strings")

        # avoid rebundant period sign in object name
        objects = [object.strip().lower().replace(".", "") for object in objects]
        text_prompt = " . ".join(objects) + " ."
        print(f"Grounding DINO detection prompt: {text_prompt}")
        
        # Predict with DINO
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=0.35,
            text_threshold=0.35
        )
        
        # Convert results to our format
        w, h = image_pil.size  # width, height from PIL
        
        scores = []
        bbox_list = []
        centers = []
        labels = []
        
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            # Convert normalized coordinates to pixel coordinates
            center_x, center_y, width, height = box
            center_x, center_y, width, height = center_x * w, center_y * h, width * w, height * h
            
            # Convert to x1, y1, x2, y2 format for bbox (normalized coordinates)
            x1 = (center_x - width / 2) / w
            y1 = (center_y - height / 2) / h
            x2 = (center_x + width / 2) / w
            y2 = (center_y + height / 2) / h
            
            scores.append(float(logit))
            bbox_list.append([x1, y1, x2, y2])
            centers.append([int(center_x), int(center_y)])
            labels.append(phrase)
        
        # only return the highest scoring detected object
        if scores:
            result = {
                "label": labels,
                "score": scores,
                "box": bbox_list,
                "center": centers
            }
        else:
            result = {"label": [], "score": [], "box": [], "center": []}
        
        print(f"Detected {len(scores)} objects with target: {text_prompt}")
        return result
        
    except Exception as e:
        print(f"Error in object detection from image: {str(e)}")
        return {"label": [], "score": [], "box": [], "center": []}

def _crop_object(image, box):
    """
    Crop the object from the image using the box

    Args:
        image: PIL Image
        box: bounding box coordinates [x1, y1, x2, y2] (normalized coordinates from grounding dino)

    Returns:
        cropped PIL Image
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image should be a PIL Image")
    
    # Convert normalized coordinates to pixel coordinates
    width, height = image.size
    x1 = int(box[0] * width)
    y1 = int(box[1] * height)
    x2 = int(box[2] * width)
    y2 = int(box[3] * height)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    return image.crop((x1, y1, x2, y2))


def _get_background(image, boxes: List[List[int]], return_mask: bool = False, mask: Image.Image | None = None):
    """
    Get the background of the image by masking all the boxed objects.
    Optionally return or accept a precomputed mask to avoid recomputation.

    Args:
        image: PIL Image
        boxes: list of bounding box coordinates [x1, y1, x2, y2] (normalized coordinates from grounding dino)
        return_mask: if True, also return the mask image used for compositing
        mask: optional PIL 'L' mode mask to reuse (255=keep background, 0=mask objects)

    Returns:
        If return_mask is False: background PIL Image
        If return_mask is True: Tuple[background PIL Image, mask PIL Image]
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image should be a PIL Image")

    # if no boxes, return the original image (and a full-white mask if requested)
    if not boxes or len(boxes) == 0:
        bg = image.copy()
        if return_mask:
            if mask is None:
                mask = Image.new('L', image.size, 255)
            return bg, mask
        return bg
    
    # Create a copy of the original image
    background = image.copy()
    
    # Create or reuse a mask to cover all detected objects
    if mask is None:
        mask = Image.new('L', image.size, 255)  # White mask (255 = keep)
        mask_draw = ImageDraw.Draw(mask)
        
        width, height = image.size
        for box in boxes:
            # Convert normalized coordinates to pixel coordinates
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Draw black rectangle on mask (0 = remove)
            mask_draw.rectangle([x1, y1, x2, y2], fill=0)
    
    # Create black background for masked areas
    # Convert to RGB if not already
    if background.mode != 'RGB':
        background = background.convert('RGB')
    
    # Create black fill image for masked areas
    fill_image = Image.new('RGB', image.size, (0, 0, 0))
    
    # Composite: use original where mask is white, fill where mask is black
    background = Image.composite(background, fill_image, mask)
    
    if return_mask:
        return background, mask
    return background


def _calculate_dinov3_similarity(dino_bundle, object1: Image.Image, object2: Image.Image, alpha: float = 0.5):
    """Compute similarity with DINOv3 features using HF AutoModel/Processor.

    Handles variable input shapes by relying on the HF processor to resize/normalize.
    Returns a scalar: alpha * semantic_sim(CLS) + (1-alpha) * texture_sim(mean patches).
    """
    if not isinstance(object1, Image.Image) or not isinstance(object2, Image.Image):
        raise ValueError("object1 and object2 should be PIL Images")

    dino_model, dino_processor = dino_bundle
    device = next(dino_model.parameters()).device

    with torch.no_grad():
        inputs = dino_processor(images=[object1, object2], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        outputs = dino_model(pixel_values=pixel_values)
        # last_hidden_state: [B, L, D]. ViT CLS token at index 0.
        feats = outputs.last_hidden_state  # type: ignore[attr-defined]
        # If model returns tuple, try attribute access fallback
        if feats is None:
            # some models may return a tuple, fallback to first element
            feats = outputs[0]

        cls1 = feats[0, 0:1, :]  # [1, D]
        cls2 = feats[1, 0:1, :]
        semantic_sim = torch.nn.functional.cosine_similarity(cls1, cls2, dim=1)  # [1]

        patch1 = feats[0, 1:, :].mean(dim=0, keepdim=True)  # [1, D]
        patch2 = feats[1, 1:, :].mean(dim=0, keepdim=True)
        texture_sim = torch.nn.functional.cosine_similarity(patch1, patch2, dim=1)  # [1]

        combined_similarity = alpha * semantic_sim + (1 - alpha) * texture_sim  # [1]
        return combined_similarity.item()
    


def _calculate_object_consistency(dinov3_model, grounding_dino_model, src_img, target_img, unchanged_objects: List[str]):   
    if not isinstance(src_img, Image.Image) or not isinstance(target_img, Image.Image):
        raise ValueError("src_img and target_img should be PIL Images")
    if src_img.size != target_img.size:
        raise ValueError("src_img and target_img should have the same size")

    if len(unchanged_objects) == 0:
        return {
            "object_dinov3_consistency": [],
            "object_dinov3_consistency_mean": None,
            "object_l1_consistency": [],
            "object_l1_consistency_mean": None
        }
    
    grounding_result = _detect_multiple_objects_from_img(grounding_dino_model, src_img, unchanged_objects)
    dinov3_similarity = []
    l1_consistency = []

    # use the box from src_img to make sure the position is unchanged
    for label, score, box in zip(grounding_result["label"], grounding_result["score"], grounding_result["box"]):
        src_object = _crop_object(src_img, box)
        target_object = _crop_object(target_img, box)
        # dinov3_model here carries (model, processor)
        similarity = _calculate_dinov3_similarity(dinov3_model, src_object, target_object)
        dinov3_similarity.append(similarity)

        # calculate l1 consistency
        transform = transforms.ToTensor()
        src_object_tensor = transform(src_object)
        target_object_tensor = transform(target_object)
        l1_loss = torch.nn.functional.l1_loss(src_object_tensor, target_object_tensor)
        l1_consistency.append(1 - l1_loss.item())

        print(f"{label} similarity: {similarity}")
    grounding_result["object_dinov3_consistency"] = dinov3_similarity
    grounding_result["object_dinov3_consistency_mean"] = float(np.mean(dinov3_similarity)) if dinov3_similarity else 0.0
    grounding_result["object_l1_consistency"] = l1_consistency
    grounding_result["object_l1_consistency_mean"] = float(np.mean(l1_consistency)) if l1_consistency else 0.0

    return grounding_result

def _calculate_background_consistency_dino(dinov3_bundle, src_img: Image.Image, target_img: Image.Image,
                                          mask_img: Image.Image | None = None,
                                          all_objects: List[str] | None = None,
                                          grounding_dino_model=None):
    """
    Compute DINOv3 masked background similarity using ViT patch tokens.

    - Converts a pixel-space background mask to a ViT patch-grid mask.
    - Extracts patch tokens (ignoring CLS) and computes mean cosine similarity
      only over patches marked as background by the mask.

    Args:
        dinov3_bundle: tuple(model, processor) from `load_consistency_model`.
        src_img, target_img: PIL Images (must be same size if mask needs building).
        mask_img: optional PIL 'L' mask (255=background keep, 0=object). If None,
                  the mask is built from detections using `all_objects` + `grounding_dino_model`.
        all_objects: list of object text prompts; required if `mask_img` is None.
        grounding_dino_model: GroundingDINO model; required if `mask_img` is None.

    Returns: dict with keys
        - 'bg_dinov3_masked_similarity': float | None
        - 'bg_dinov3_masked_patches': int
        - 'bg_dinov3_grid': [Hp, Wp] | None
    """
    if dinov3_bundle is None:
        return {
            'bg_dinov3_masked_similarity': None,
            'bg_dinov3_masked_patches': 0,
            'bg_dinov3_grid': None,
        }

    if mask_img is None:
        if all_objects is None or grounding_dino_model is None:
            raise ValueError("mask_img is None; all_objects and grounding_dino_model are required to build mask")
        if not isinstance(src_img, Image.Image) or not isinstance(target_img, Image.Image):
            raise ValueError("src_img and target_img should be PIL Images")
        if src_img.size != target_img.size:
            raise ValueError("src_img and target_img should have the same size")

        src_res = _detect_multiple_objects_from_img(grounding_dino_model, src_img, all_objects)
        tgt_res = _detect_multiple_objects_from_img(grounding_dino_model, target_img, all_objects)
        all_boxes = src_res["box"] + tgt_res["box"]
        # Reuse _get_background to construct the mask
        _, mask_img = _get_background(src_img, all_boxes, return_mask=True)

    # DINO forward on original images
    dino_model, dino_processor = dinov3_bundle
    device = next(dino_model.parameters()).device

    bg_dinov3_masked_similarity = None
    bg_dinov3_masked_patches = 0
    bg_dinov3_grid = None

    try:
        with torch.no_grad():
            inputs = dino_processor(images=[src_img, target_img], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            _, _, Hpixels, Wpixels = pixel_values.shape

            outputs = dino_model(pixel_values=pixel_values)
            feats = outputs.last_hidden_state
            if feats is None:
                feats = outputs[0]

            # Determine (Hp, Wp) from token count Np and input aspect
            Np = feats.shape[1] - 1
            import math
            aspect = float(Hpixels) / float(max(1, Wpixels))
            best_diff = None
            Hp, Wp = 1, Np
            for w in range(1, int(math.sqrt(Np)) + 1):
                if Np % w != 0:
                    continue
                h = Np // w
                for hh, ww in ((h, w), (w, h)):
                    diff = abs((hh / max(1.0, ww)) - aspect)
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        Hp, Wp = int(hh), int(ww)
            bg_dinov3_grid = [Hp, Wp]

            # Build soft patch weights in [0,1] by averaging the pixel mask over each ViT patch
            # First resize the pixel mask to the model input resolution
            mask_to_model = mask_img.resize((Wpixels, Hpixels), resample=Image.NEAREST)
            mask_tensor_img = transforms.ToTensor()(mask_to_model).unsqueeze(0).to(device)  # [1,1,H',W'] in {0,1}

            # Pool to patch grid using exact kernel sizes when divisible; otherwise interpolate
            kH = Hpixels // Hp
            kW = Wpixels // Wp
            if Hpixels % Hp == 0 and Wpixels % Wp == 0 and kH > 0 and kW > 0:
                weights = torch.nn.functional.avg_pool2d(mask_tensor_img, kernel_size=(kH, kW), stride=(kH, kW))
            else:
                weights = torch.nn.functional.interpolate(mask_tensor_img, size=(Hp, Wp), mode='area')
            weights = weights.squeeze(0).squeeze(0).clamp(0.0, 1.0)  # [Hp, Wp] on same device as feats

            # Reshape tokens to [Hp, Wp, D], then flatten to [1, N, D]
            tokens_src = feats[0, 1:, :].reshape(Hp, Wp, -1)
            tokens_tgt = feats[1, 1:, :].reshape(Hp, Wp, -1)
            ts_flat = tokens_src.reshape(-1, tokens_src.shape[-1]).unsqueeze(0)
            tt_flat = tokens_tgt.reshape(-1, tokens_tgt.shape[-1]).unsqueeze(0)

            w_flat = weights.reshape(1, -1)  # (1, N) soft weights

            # Use soft-weighted global pooling to compute a single cosine similarity
            bg_dinov3_masked_similarity = _masked_global_cosine(ts_flat, w_flat, tt_flat, w_flat, weight_thresh=0.5)
            bg_dinov3_masked_patches = int((w_flat > 0).sum().item())
    except Exception as e:
        print(f"Warning: DINOv3 masked background similarity failed: {e}")

    return {
        'bg_dinov3_masked_similarity': bg_dinov3_masked_similarity,
        'bg_dinov3_masked_patches': bg_dinov3_masked_patches,
        'bg_dinov3_grid': bg_dinov3_grid,
    }


def _calculate_background_consistency(grounding_dino_model, src_img, target_img, all_objects: List[str], plot=False, dinov3_bundle=None):
    if not isinstance(src_img, Image.Image) or not isinstance(target_img, Image.Image):
        raise ValueError("src_img and target_img should be PIL Images")
    if src_img.size != target_img.size:
        raise ValueError("src_img and target_img should have the same size")
    
    if len(all_objects) == 0:
        return {
            "bg_l1_consistency": None,
            "src_objects_detected_in_bg": [],
            "target_objects_detected_in_bg": [],
            "total_boxes_used": 0
        }
    
    # Detect objects in both images
    src_grounding_result = _detect_multiple_objects_from_img(grounding_dino_model, src_img, all_objects)
    target_grounding_result = _detect_multiple_objects_from_img(grounding_dino_model, target_img, all_objects)

    # Union all boxes from both images
    all_boxes = src_grounding_result["box"] + target_grounding_result["box"]
    
    # Get backgrounds by masking out all detected objects; reuse the same mask for both
    src_background, mask_img = _get_background(src_img, all_boxes, return_mask=True)
    target_background = _get_background(target_img, all_boxes, mask=mask_img)
    
    # Convert PIL images to tensors
    transform = transforms.ToTensor()
    src_bg_tensor = transform(src_background)  # [C, H, W], values in [0, 1]
    target_bg_tensor = transform(target_background)

    # Convert returned mask to tensor: [1, H, W] in {0., 1.}
    mask_tensor = transforms.ToTensor()(mask_img)
    # Expand mask to channels
    mask_tensor = mask_tensor.expand_as(src_bg_tensor)
    valid_count = mask_tensor.sum()

    if valid_count.item() == 0:
        # No unmasked pixels; cannot compute a meaningful background consistency
        bg_consistency = None
        l1_loss_value = None
    else:
        # Masked L1 over unmasked pixels only
        diff = (src_bg_tensor - target_bg_tensor).abs()
        l1_loss = (diff * mask_tensor).sum() / valid_count
        l1_loss_value = l1_loss.item()
        bg_consistency = 1 - l1_loss_value

    # DINOv3 masked patch-token similarity over background region (optional)
    bg_dino = {'bg_dinov3_masked_similarity': None, 'bg_dinov3_masked_patches': 0, 'bg_dinov3_grid': None}
    if dinov3_bundle is not None:
        bg_dino = _calculate_background_consistency_dino(
            dinov3_bundle, src_img, target_img, mask_img=mask_img
        )

    # Optional plotting
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(src_img)
        axes[0, 0].set_title('Source Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(target_img)
        axes[0, 1].set_title('Target Image')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(src_background)
        axes[1, 0].set_title('Source Background')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(target_background)
        axes[1, 1].set_title('Target Background')
        axes[1, 1].axis('off')

        title_loss = (f"{l1_loss_value:.4f}" if l1_loss_value is not None else "N/A")
        plt.suptitle(f'Background Consistency Analysis\nMasked L1 Loss: {title_loss}')
        plt.tight_layout()
        plt.savefig("background_consistency.png")
        plt.show()
    
    return {
        'bg_l1_consistency': bg_consistency,
        'src_objects_detected_in_bg': src_grounding_result["label"],
        'target_objects_detected_in_bg': target_grounding_result["label"],
        'total_boxes_used': len(all_boxes),
        'bg_dinov3_masked_similarity': bg_dino['bg_dinov3_masked_similarity'],
        'bg_dinov3_masked_patches': bg_dino['bg_dinov3_masked_patches'],
        'bg_dinov3_grid': bg_dino['bg_dinov3_grid'],
    }

def evaluate_consistency(src_image, target_image, unchanged_objects, all_objects, grounding_model=None, dinov3_model=None):
    if grounding_model is None and dinov3_model is None:
        dinov3_model, grounding_model = load_consistency_model(load_grounding_dino=True)  # dinov3_model is a (model, processor) tuple
    if dinov3_model is None:
        dinov3_model = load_consistency_model(load_grounding_dino=False)  # (model, processor)
    
    if isinstance(unchanged_objects, str):
        unchanged_objects = [obj.strip() for obj in unchanged_objects.split(".") if obj.strip()]
    if isinstance(all_objects, str):
        all_objects = [obj.strip() for obj in all_objects.split(".") if obj.strip()]
    
    src_img_pil = load_resize_image(src_image)
    target_img_pil = load_resize_image(target_image)

    object_result = _calculate_object_consistency(dinov3_model, grounding_model, src_img_pil, target_img_pil, unchanged_objects)
    background_result = _calculate_background_consistency(grounding_model, src_img_pil, target_img_pil, all_objects, plot=False, dinov3_bundle=dinov3_model)

    return object_result, background_result

if __name__ == "__main__":
    import json
    from utils import _make_json_serializable
    dinov3_bundle, grounding_dino_model = load_consistency_model(load_grounding_dino=True)
    image = Image.open("./test/single_instruction_img/base.png")
    target_image = Image.open("./test/single_instruction_img/replace.png")
    objects = ["a purple bag", "a dog", "an umbrella"]
    all_objects = ["a purple bag", "a dog", "an umbrella"]
    result = _detect_multiple_objects_from_img(grounding_dino_model, image, objects)
    print(result)
    background = _get_background(image, result["box"])
    background.save("background.jpg")
    for label, score, box in zip(result["label"], result["score"], result["box"]):
        cropped_image = _crop_object(image, box)
        # add title to the image
        draw = ImageDraw.Draw(cropped_image)
        draw.text((10, 10), f"{label} ({score:.2f})", fill=(255, 255, 255))
        # save the image with the title
        cropped_image.save(f"{label}.jpg")
    result = _calculate_object_consistency(dinov3_bundle, grounding_dino_model, image, target_image, objects)
    # Make sure all values are JSON serializable
    json_result = _make_json_serializable(result)
    json.dump(json_result, open("result.json", "w"), indent=2)
    background_result = _calculate_background_consistency(grounding_dino_model, image, target_image, all_objects, plot=True, dinov3_bundle=dinov3_bundle)
    json_background_result = _make_json_serializable(background_result)
    json.dump(json_background_result, open("background_result.json", "w"), indent=2)
