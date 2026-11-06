import os
import io
import base64
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Ensure matplotlib and transformers write to local writable directory
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
UPLOADS_DIR = Path(__file__).resolve().parents[2] / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(UPLOADS_DIR))

MODEL_ID = "dima806/deepfake_vs_real_image_detection"


class HuggingFaceModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


def reshape_transform_vit(tensor, height=14, width=14):
    """
    Reshapes ViT sequence tokens (197 tokens: 1 [CLS] + 196 patch tokens)
    into standard 2D spatial feature maps of shape (B, C, H, W) for Grad-CAM.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


class ExplainabilityService:
    """
    Forensic Explainability Service:
    Generates class-activation heatmaps (Grad-CAM) targeting manipulation artifacts,
    localizes anomalous facial sectors, and produces visual overlays.
    """

    def __init__(self):
        self.device = torch.device("cpu")
        print("Initializing DFX Grad-CAM Explainability Engine...")

        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

        self.wrapper = HuggingFaceModelWrapper(self.model)

        # Target layer: last transformer encoder block layernorm_before
        self.target_layers = [self.model.vit.layers[-1].layernorm_before]

        self.cam = GradCAM(
            model=self.wrapper,
            target_layers=self.target_layers,
            reshape_transform=reshape_transform_vit
        )
        print("DFX Grad-CAM Explainability Engine ready.")

    def analyze_sectors(self, heatmap: np.ndarray) -> str:
        """
        Calculates activation density across facial sectors (top, middle, bottom)
        to identify the primary focus of anomalous activation.
        """
        h, w = heatmap.shape
        top_sector = np.mean(heatmap[: int(h * 0.35), :])
        mid_sector = np.mean(heatmap[int(h * 0.35): int(h * 0.70), :])
        bot_sector = np.mean(heatmap[int(h * 0.70):, :])

        sectors = [
            ("Forehead / Hairline / Upper Boundary", top_sector),
            ("Periorbital Region / Eyes & Bridge", mid_sector),
            ("Oral-Mandibular / Chin & Blending Seam", bot_sector)
        ]
        sectors.sort(key=lambda s: s[1], reverse=True)
        primary = sectors[0][0]
        return primary

    def generate_heatmap(
        self,
        face_crop: np.ndarray,
        target_class: int = 1,
        save_to_disk: bool = True
    ) -> Dict[str, Any]:
        """
        Generates Grad-CAM heatmap for a given face crop (BGR or RGB).
        Returns base64 data URL, intensity metrics, and sector localization.
        """
        if isinstance(face_crop, np.ndarray):
            if face_crop.shape[2] == 3:
                # Expecting BGR from OpenCV; convert to RGB for PIL and model
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
            else:
                raise ValueError("Expected 3-channel image array")
        elif isinstance(face_crop, Image.Image):
            pil_image = face_crop.convert("RGB")
            face_rgb = np.array(pil_image)
        else:
            raise TypeError("face_crop must be a numpy array or PIL Image")

        # Resize to standard model input (224, 224)
        if pil_image.size != (224, 224):
            pil_image = pil_image.resize((224, 224), Image.Resampling.BILINEAR)
            face_rgb = np.array(pil_image)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        input_tensor = inputs["pixel_values"].to(self.device)

        # Target class 1 = Fake / Manipulation
        targets = [ClassifierOutputTarget(target_class)]

        # Run Grad-CAM forward and backward passes
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Normalize RGB image to [0, 1] for overlay blending
        rgb_float = np.float32(face_rgb) / 255.0
        vis_rgb = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        # Sector analysis
        primary_sector = self.analyze_sectors(grayscale_cam)
        peak_intensity = float(np.max(grayscale_cam))
        mean_intensity = float(np.mean(grayscale_cam))

        # Encode visualization overlay to Base64 data URL
        success, encoded_jpg = cv2.imencode(".jpg", vis_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            b64_str = base64.b64encode(encoded_jpg.tobytes()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64_str}"
        else:
            data_url = ""

        # Optionally save file to backend/uploads
        saved_file_name = None
        if save_to_disk:
            saved_file_name = f"gradcam_{uuid.uuid4().hex[:12]}.jpg"
            saved_path = UPLOADS_DIR / saved_file_name
            cv2.imwrite(str(saved_path), vis_bgr)

        return {
            "heatmap_data_url": data_url,
            "saved_filename": saved_file_name,
            "peak_intensity": round(peak_intensity, 3),
            "mean_intensity": round(mean_intensity, 3),
            "focal_region": primary_sector,
            "interpretation": (
                f"Peak activation concentrated in the {primary_sector}. "
                "Highlight areas denote visual tokens with anomalous feature attribution."
            )
        }

