import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Set offline mode by default so startup does not stall on DNS timeouts
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SWIN_MODEL_ID = "umm-maybe/AI-image-detector"
VIT_MODEL_ID = "dima806/deepfake_vs_real_image_detection"


class DeepfakeDetectionService:
    """
    Forensic Dual-Engine Media Analysis Service:
    - Primary Engine: Swin Transformer (Hierarchical shifted-window self-attention,
      yielding high specificity and resistance to studio/lighting false positives).
    - Secondary Engine: Vision Transformer (ViT-Base, specialized in GAN/Diffusion
      synthesis artifacts and compression stability).
    """

    def __init__(self):
        self.device = torch.device("cpu")
        print("Initializing DFX Forensic Dual-Engine Service (CPU)...")

        # 1. Swin Transformer Engine
        print(f"Loading Swin Engine ({SWIN_MODEL_ID})...")
        self.swin_processor = AutoImageProcessor.from_pretrained(SWIN_MODEL_ID)
        self.swin_model = AutoModelForImageClassification.from_pretrained(SWIN_MODEL_ID)
        self.swin_model.to(self.device)
        self.swin_model.eval()

        # Resolve Swin label mappings
        # umm-maybe: 0 -> artificial, 1 -> human
        self.swin_fake_idx = next(
            k for k, v in self.swin_model.config.id2label.items() if "art" in v.lower()
        )
        self.swin_real_idx = next(
            k for k, v in self.swin_model.config.id2label.items() if "hum" in v.lower()
        )

        # 2. Vision Transformer Engine
        print(f"Loading ViT Engine ({VIT_MODEL_ID})...")
        self.vit_processor = AutoImageProcessor.from_pretrained(VIT_MODEL_ID)
        self.vit_model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_ID)
        self.vit_model.to(self.device)
        self.vit_model.eval()

        # Resolve ViT label mappings
        # dima806: 0 -> Real, 1 -> Fake
        self.vit_real_idx = next(
            k for k, v in self.vit_model.config.id2label.items() if "real" in v.lower()
        )
        self.vit_fake_idx = next(
            k for k, v in self.vit_model.config.id2label.items() if "fake" in v.lower()
        )

        print("DFX Forensic Dual-Engine loaded successfully.")

    def predict(self, image: Image.Image):
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image")

        image = image.convert("RGB")

        # 1. Swin Forward Pass
        swin_inputs = self.swin_processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            swin_logits = self.swin_model(**swin_inputs).logits
            swin_probs = torch.softmax(swin_logits, dim=1)[0]

        swin_fake = swin_probs[self.swin_fake_idx].item()
        swin_real = swin_probs[self.swin_real_idx].item()

        # 2. ViT Forward Pass
        vit_inputs = self.vit_processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            vit_logits = self.vit_model(**vit_inputs).logits
            vit_probs = torch.softmax(vit_logits, dim=1)[0]

        vit_fake = vit_probs[self.vit_fake_idx].item()
        vit_real = vit_probs[self.vit_real_idx].item()

        # Weighted Ensemble:
        # Swin provides superior authentic grounding (60%),
        # ViT detects high-frequency GAN synthesis (40%).
        combined_fake = (0.60 * swin_fake + 0.40 * vit_fake) * 100.0
        combined_real = 100.0 - combined_fake

        # Binary label for frontend schema compatibility
        label = "REAL" if combined_real >= 50.0 else "FAKE"
        confidence = round(combined_real if label == "REAL" else combined_fake, 2)

        # Forensic uncertainty: distance from the 50/50 decision boundary
        # Scale: 0% (total certainty) to 100% (maximum uncertainty)
        margin_from_boundary = abs(combined_real - combined_fake)
        uncertainty = round(100.0 - margin_from_boundary, 2)

        # Calibrated qualitative assessment
        if combined_fake >= 65.0:
            assessment = "Manipulation Indicated"
        elif combined_fake <= 35.0:
            assessment = "Authentic Indicated"
        else:
            assessment = "Inconclusive / Elevated Uncertainty"

        return {
            "label": label,
            "confidence": confidence,
            "real_score": round(combined_real, 2),
            "fake_score": round(combined_fake, 2),
            "uncertainty_score": uncertainty,
            "assessment": assessment,
            "engine_agreement": (swin_fake >= 0.5) == (vit_fake >= 0.5),
            "indicators": {
                "swin_fake_prob": round(swin_fake * 100, 2),
                "vit_fake_prob": round(vit_fake * 100, 2),
            }
        }
