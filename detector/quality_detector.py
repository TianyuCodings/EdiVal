import torch
import torch.nn as nn
from transformers import ViTModel, T5Tokenizer, T5ForConditionalGeneration, AutoImageProcessor
from PIL import Image, ImageOps
import os
from pathlib import Path
import gdown
import cv2
import numpy as np
from .utils import load_resize_image
from tempfile import NamedTemporaryFile

# Cache for optional HPSv3 inferencer
_HPSV3_INFERENCER = None

def load_human_preference_inferencer(device: str | None = None):
    """Lazily load HPSv3 inferencer if available.

    Returns the inferencer instance, or None if the package is not installed.
    """
    global _HPSV3_INFERENCER
    if _HPSV3_INFERENCER is not None:
        return _HPSV3_INFERENCER

    try:
        from hpsv3 import HPSv3RewardInferencer  # type: ignore
    except Exception as e:
        print(f"HPSv3 not available ({e}). Skipping human_preference_score.")
        return None

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        _HPSV3_INFERENCER = HPSv3RewardInferencer(device=device)
    except Exception as e:
        print(f"Failed to initialize HPSv3 inferencer on device {device}: {e}")
        _HPSV3_INFERENCER = None
    return _HPSV3_INFERENCER

class LayerNorm(nn.Module):
    """T5-style LayerNorm over the channel dimension (No bias and no subtraction of mean)."""
    def __init__(self, n_channels):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(n_channels, 1, 1))

    def forward(self, x: torch.Tensor):
        # x is a feature map of shape: batch_size x n_channels x h x w
        var = x.square().mean(dim=1, keepdim=True)
        out = x * (var + 1e-8).rsqrt()
        out = out * self.scale
        return out


class HeatmapPredictor(nn.Module):
    def __init__(self, n_channels):
        super(HeatmapPredictor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 768, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            LayerNorm(768),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            LayerNorm(384),
        )

        self.deconv_layers = nn.ModuleList()
        self.conv_layers2 = nn.ModuleList()
        in_channels = 384
        for out_channels in [768, 384, 384, 192]:
            self.deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    LayerNorm(out_channels),
                    nn.ReLU()
                )
            )
            self.conv_layers2.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
                    LayerNorm(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
                    LayerNorm(out_channels),
                )
            )
            in_channels = out_channels

        self.relu = nn.ReLU()

        self.last_conv1 = nn.Conv2d(in_channels, 192, kernel_size=3, stride=1, padding='same')
        # relu
        self.last_conv2 = nn.Conv2d(192, 1, kernel_size=3, stride=1, padding='same') # final_channel size
        # sigmoid
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv_layers(x)
        for deconv, conv in zip(self.deconv_layers, self.conv_layers2):
            x = deconv(x)
            identity = x
            x = conv(x)
            x = x + identity
            x = self.relu(x)

        x = self.last_conv1(x)
        x = self.relu(x)
        x = self.last_conv2(x)
        x = self.sigmoid(x)  # (batch_size, 1, height, width)

        output = x.squeeze(1)
        return output


class ScorePredictor(nn.Module):
    def __init__(self, n_channels, n_patches=14*14):
        super(ScorePredictor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 2),
            nn.ReLU(),
            nn.Conv2d(n_channels // 2, n_channels // 4, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 4),
            nn.ReLU(),
            nn.Conv2d(n_channels // 4, n_channels // 8, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 8),
            nn.ReLU(),
            nn.Conv2d(n_channels // 8, n_channels // 16, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 16),
            nn.ReLU(),
            nn.Conv2d(n_channels // 16, n_channels // 64, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 64),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(n_channels // 64 * n_patches, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        conv_output = self.conv_layers(x)
        conv_output = conv_output.flatten(1)
        output = self.linear_layers(conv_output)
        return output


class RAHF(nn.Module):
    def __init__(
            self,
            score_types=('plausibility', 'alignment', 'aesthetics', 'overall'),
            heatmap_types=('implausibility', 'misalignment'),
            vit_model="google/vit-large-patch16-384",
            t5_model="t5-base",
            multi_heads=True,
            patch_size=16,
            image_size=384,
        ):
        super(RAHF, self).__init__()
        self.multi_heads = multi_heads
        self.score_types = score_types
        self.heatmap_types = heatmap_types
        self.n_patches = image_size // patch_size

        # Load pre-trained ViT model for image encoding
        self.vit = ViTModel.from_pretrained(vit_model)

        # Load pre-trained T5 model
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model)

        # Linear layer to align visual token dimensions to T5 hidden size
        self.visual_token_projection = nn.Linear(self.vit.config.hidden_size, self.t5.config.d_model)

        n_channels = self.t5.config.d_model
        if self.multi_heads:
            self.heatmap_predictor = nn.ModuleDict({hm: HeatmapPredictor(n_channels) for hm in heatmap_types})
            self.score_predictor = nn.ModuleDict({score: ScorePredictor(n_channels, self.n_patches ** 2) for score in score_types})
        else:
            # Single head
            self.heatmap_predictor = HeatmapPredictor(n_channels)
            self.score_predictor = ScorePredictor(n_channels, self.n_patches ** 2)

    def encode_vis_text(self, visual_tokens, caption, prepend_text=None):
        if prepend_text is not None:
            caption = [f"<output> {prepend_text} </output> {c}" for c in caption]
        # Tokenize the caption
        input_ids = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).input_ids.to(visual_tokens.device)

        # Embed textual tokens using T5's embedding layer
        textual_embeddings = self.t5.encoder.embed_tokens(input_ids)  # (batch_size, seq_len, t5_hidden_dim)

        # Concatenate visual tokens and textual embeddings along the sequence length dimension
        concatenated_tokens = torch.cat([visual_tokens, textual_embeddings], dim=1)  # (batch_size, seq_len + num_patches, t5_hidden_dim)

        # Encode the concatenated tokens with T5 encoder
        encoder_outputs = self.t5.encoder(inputs_embeds=concatenated_tokens)

        if prepend_text is not None:
            # Extract visual tokens from encoder outputs (remove CLS token)
            visual_tokens = encoder_outputs.last_hidden_state[:, 1:visual_tokens.size(1), :]  # Visual tokens portion
            batch_size = visual_tokens.shape[0]
            feature_map = visual_tokens.transpose(1, 2).view(batch_size, -1, self.n_patches, self.n_patches)  # Reshape to (batch_size, t5_hidden_size, height, width)
            return feature_map
        
        return encoder_outputs

    def forward(self, image, caption, target_text=None, max_new_tokens=100):
        # Encode the image using ViT
        vit_outputs = self.vit(pixel_values=image)
        visual_tokens = vit_outputs.last_hidden_state  # (batch_size, num_patches, vit_hidden_dim)
        batch_size = visual_tokens.shape[0]

        # Project visual tokens to T5 hidden size
        visual_tokens = self.visual_token_projection(visual_tokens)  # (batch_size, num_patches, t5_hidden_dim)

        encoder_outputs = self.encode_vis_text(visual_tokens, caption)

        last_hidden_state = encoder_outputs.last_hidden_state
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[:, visual_tokens.size(1):]

        outputs = {}

        # Compute outputs using T5's decoder with a generation head
        if target_text is not None:
            # Prepare decoder inputs for teacher forcing
            caption_inputs = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True).to(image.device)
            decoder_input_ids = caption_inputs.input_ids

            t5_outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=decoder_input_ids  # Using labels computes the loss automatically
            )

            loss = t5_outputs.loss  # Teacher-forcing loss is automatically computed
            # logits = t5_outputs.logits  # Predicted vocabulary scores
            outputs['seq_loss'] = loss
        else:
            # Generation texts
            output_seq = self.t5.generate(
                encoder_outputs=encoder_outputs,
                max_new_tokens=max_new_tokens,
            )
            pred_seq = self.tokenizer.batch_decode(output_seq, skip_special_tokens=True)
            outputs['output_seq'] = pred_seq

        if self.multi_heads:
            # Extract visual tokens from encoder outputs (remove CLS token)
            visual_tokens = last_hidden_state[:, 1:visual_tokens.size(1), :]  # Visual tokens portion
            feature_map = visual_tokens.transpose(1, 2).view(batch_size, -1, self.n_patches, self.n_patches)  # Reshape to (batch_size, t5_hidden_size, height, width)

            heatmaps = {hm: hmp(feature_map) for hm, hmp in self.heatmap_predictor.items()}
            scores = {sc: scp(feature_map).flatten() for sc, scp in self.score_predictor.items()}
        else:
            scores = {}
            for score in self.score_types:
                feature_map = self.encode_vis_text(visual_tokens, caption, prepend_text=f'SCORE: {score}')
                scores[score] = self.score_predictor(feature_map).flatten()

            heatmaps = {}
            for heatmap in self.heatmap_types:
                feature_map = self.encode_vis_text(visual_tokens, caption, prepend_text=f'HEATMAP: {heatmap}')
                heatmaps[heatmap] = self.heatmap_predictor(feature_map)

        outputs['heatmaps'] = heatmaps
        outputs['scores'] = scores

        return outputs


def preprocess_image(image):
    #image = Image.open(image_path).convert("RGB")
    if isinstance(image, Image.Image):
        pass
    else:
        image = Image.open(image).convert("RGB")

    transform = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
    
    return transform(image, return_tensors="pt")['pixel_values'][0].unsqueeze(0)


def load_quality_model(device=None, use_hf=True):
    """Initialize the RichHF model once."""
    # Detect device (GPU if available, else CPU)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = RAHF()

    if use_hf:
        # Load from HuggingFace Hub
        from huggingface_hub import hf_hub_download
        weight_repo = "C-Tianyu/RAHF"
        weight_file = "rahf_model.pt"
        
        print(f"Downloading model weights from HuggingFace Hub: {weight_repo}")
        ckpt_path = hf_hub_download(repo_id=weight_repo, filename=weight_file)
        print("Model weights downloaded from HuggingFace successfully!")
    else:
        ckpt_path = Path("rahf_ckpts/rahf_model.pt")  # Keep within EditBench directory
        
        # Create directory if it doesn't exist
        if not ckpt_path.parent.exists():
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download model if it doesn't exist
        if not ckpt_path.exists():
            # see https://github.com/youweiliang/RichHF?tab=readme-ov-file
            print("Model weights not found. Downloading from Google Drive...")
            file_id = "1-jKfmpyGtJ0UAgEQ23zylRsmQ82qigzB"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(ckpt_path), quiet=False)
            print("Model weights downloaded successfully!")
    
    # Load the model weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)  # Move model to device
    model.eval()
    print("RAHF model initialized successfully")

    # Also initialize HPSv3 inferencer (optional) and return it alongside the model.
    hps_inferencer = None
    try:
        hps_inferencer = load_human_preference_inferencer(device=str(device))
        if hps_inferencer is not None:
            print("HPSv3 inferencer initialized successfully")
        else:
            print("HPSv3 inferencer not available; human_preference_score will be None")
    except Exception as e:
        print(f"Failed to load HPSv3 inferencer: {e}")

    return model, hps_inferencer


def human_preference_score(image, prompt: str | None = None, inferencer=None):
    """Compute human preference score using HPSv3 if available.

    Args:
        image: PIL Image or image file path
        prompt: Optional prompt to condition the scoring
        inferencer: Optional pre-initialized HPSv3 inferencer

    Returns:
        float | None: The human preference score (mu) or None if unavailable
    """
    try:
        inferencer = inferencer or load_human_preference_inferencer(None)
        if inferencer is None:
            return None

        use_prompt = prompt if prompt is not None else ""
        if isinstance(image, str):
            img_path = image
            cleanup_tmp = False
            pil_image = None
        else:
            pil_image = image if isinstance(image, Image.Image) else load_resize_image(image)
            tmp = NamedTemporaryFile(delete=False, suffix=".png")
            pil_image.save(tmp.name)
            img_path = tmp.name
            cleanup_tmp = True

        try:
            rewards = inferencer.reward(prompts=[use_prompt], image_paths=[img_path])
            if len(rewards) > 0:
                r0 = rewards[0]
                if torch.is_tensor(r0):
                    return float(r0[0].item()) if r0.numel() >= 1 else float(r0.item())
                elif isinstance(r0, (list, tuple)) and len(r0) > 0:
                    v = r0[0]
                    return float(v.item()) if torch.is_tensor(v) else float(v)
        finally:
            if cleanup_tmp:
                try:
                    os.remove(img_path)
                except Exception:
                    pass
        return None
    except Exception as e:
        print(f"Failed to compute human_preference_score: {e}")
        return None

def _to_uint8_bgr(img):
    if img.dtype == np.uint8:
        return img
    # assume float in [0,1] or any real; clamp then scale
    f = img.astype(np.float32)
    f = np.clip(f, 0.0, 1.0)
    return (f * 255.0).astype(np.uint8)

def saturation_metrics(img, mask=None, high_thr=0.9, v_min=0.10):
    """Returns (mean_saturation, high_saturation_ratio) in [0,1]."""
    img8 = _to_uint8_bgr(img)
    hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0
    V = hsv[..., 2].astype(np.float32) / 255.0
    m = (V > v_min)
    if mask is not None: m &= (mask > 0)
    if not np.any(m):
        return 0.0, 0.0
    s = S[m]
    return float(s.mean()), float((s > high_thr).mean())

def sharpness_metrics(img, mask=None):
    """Variance of Laplacian after light blur to reduce noise sensitivity."""
    img8 = _to_uint8_bgr(img)
    gray = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    x = lap[mask > 0] if mask is not None else lap
    return float(x.var())

def contrast_metrics(img, mask=None, normalized=True, space="lab"):
    """RMS/global contrast on L* (Lab) by default; optionally normalized by mean."""
    img8 = _to_uint8_bgr(img)
    if space.lower() == "lab":
        L = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255.0
    else:
        L = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    x = L[mask > 0] if mask is not None else L
    if x.size == 0:
        return 0.0
    s = float(x.std())
    return float(s / (x.mean() + 1e-8)) if normalized else s


def to_linear_srgb_from_rgb8(img_rgb_uint8):
    img = img_rgb_uint8.astype(np.float32) / 255.0
    a = 0.055
    lin = np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4)
    return lin  # linear RGB in [0,1]


def overexposure_metrics_from_rgb(
    img_rgb_uint8,
    y_thr=0.98,
    ch_thr=0.98,
    flat_std_thr=0.01,
    min_region=50,
):
    lin = to_linear_srgb_from_rgb8(img_rgb_uint8)
    R, G, B = lin[..., 0], lin[..., 1], lin[..., 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

    H = (Y >= y_thr)
    near = (
        (R >= ch_thr).astype(np.uint8)
        + (G >= ch_thr).astype(np.uint8)
        + (B >= ch_thr).astype(np.uint8)
    )
    multi_ch_clip = (near >= 2)

    # Local flatness on luminance
    Y_blur = cv2.GaussianBlur(Y, (0, 0), 1.0)
    Y2_blur = cv2.GaussianBlur(Y * Y, (0, 0), 1.0)
    local_std = np.sqrt(np.maximum(0.0, Y2_blur - Y_blur * Y_blur))
    flat = (local_std < flat_std_thr)

    # Remove tiny specular dots
    H_flat = (H & flat).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(H_flat, 8)
    blown_flat = np.zeros_like(H_flat, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_region:
            blown_flat |= (labels == i)

    total = Y.size
    clip_frac = H.sum() / total
    multi_channel_clip_frac = multi_ch_clip.sum() / total
    blown_flat_frac = blown_flat.sum() / total

    top_mask = Y >= y_thr
    tail_std = float(np.std(Y[top_mask])) if np.any(top_mask) else 0.0
    p99 = float(np.percentile(Y, 99))
    p999 = float(np.percentile(Y, 99.9))
    oei = blown_flat_frac + 0.5 * multi_channel_clip_frac

    return dict(
        clip_frac=float(clip_frac),
        multi_channel_clip_frac=float(multi_channel_clip_frac),
        blown_flat_frac=float(blown_flat_frac),
        OEI=float(oei),
        p99=p99,
        p999=p999,
        tail_std=tail_std,
    )


def overexposure_metrics_pil(pil_img, **kwargs):
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(bg, pil_img.convert("RGBA")).convert("RGB")
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr_rgb = np.asarray(pil_img)
    return overexposure_metrics_from_rgb(arr_rgb, **kwargs)


def evaluate_quality(target_image, model=None, prompt: str | None = None, hps_inferencer=None):
    """
    Main function to verify if target_image follows the format_instruction compared with src_image.
    
    Args:
        target_image: Target image to verify - can be a file path (str) or PIL Image
        
    Returns:
        dict: Dictionary containing quality metrics
    """
    # Load images if they are file paths
    if model is None:
        model, hps_inferencer = load_quality_model()

    pil_image, cv_image = load_resize_image(target_image, return_cv2=True)

    # Preprocess for model
    image_tensor = preprocess_image(pil_image)

    # Move input tensor to the same device as the model
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    # Get model predictions
    out = model(image_tensor, '')['scores']
    plausibility = round(out['plausibility'].item(), 2)
    
    # Use 'aesthetics' instead of 'artifact' since 'artifact' is not in default score_types
    aesthetics = round(out['aesthetics'].item(), 2)
    
    # Calculate traditional image metrics using OpenCV image
    mean_s, high_sat_ratio = saturation_metrics(cv_image)
    sharpness = sharpness_metrics(cv_image)
    contrast = contrast_metrics(cv_image)
    # Overexposure metrics using PIL image (handles EXIF/alpha)
    overexp = overexposure_metrics_pil(pil_image)
    
    result = dict()
    result['plausibility'] = plausibility
    result['aesthetics'] = aesthetics  # Changed from 'artifact' to 'aesthetics'
    result['mean_s'] = mean_s
    result['high_sat_ratio'] = high_sat_ratio
    result['sharpness'] = sharpness
    result['contrast'] = contrast
    # Overexposure related metrics
    result.update(overexp)

    # Optional: Human Preference Score via HPSv3 (delegates to dedicated function)
    hps = human_preference_score(target_image, prompt=prompt, inferencer=hps_inferencer)
    result['human_preference_score'] = round(hps, 4) if isinstance(hps, (int, float)) else None
    return result


if __name__ == "__main__":
    import os
    results = dict()
    src_image = './test/single_instruction_img/base.png'
    target_image = './test/single_instruction_img/replace.png'
    instruction = "Replace [dog] with [person]"
    
    # Load model once for efficiency
    model, hps_inferencer = load_quality_model()
    
    print("*" * 100)
    print("Evaluating target image quality...")
    result = evaluate_quality(target_image, model, hps_inferencer=hps_inferencer)
    print("Quality evaluation results:", result)
    print("*" * 100)
