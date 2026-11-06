import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "backend", "uploads")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from app.services.face_detector import detect_faces

MODEL_ID = "dima806/deepfake_vs_real_image_detection"

class HuggingFaceModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)

print("Initializing ViT Grad-CAM Engine...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
hf_model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
hf_model.eval()

wrapper = HuggingFaceModelWrapper(hf_model)

# The exact layer identified: last layer's layernorm_before
target_layers = [hf_model.vit.layers[-1].layernorm_before]

cam = GradCAM(
    model=wrapper,
    target_layers=target_layers,
    reshape_transform=reshape_transform_vit
)

for target_name, rel_path, class_idx in [
    ("Fake StyleGAN", "dataset/fake/random-person.jpeg", 1),
    ("Fake DeepSwap", "dataset/fake/fake_hidf_01.jpg", 1),
    ("Real Webcam", "dataset/real/WIN_20261104_14_03_36_Pro.jpg", 0)
]:
    p = Path("backend") / rel_path
    if not p.exists():
        continue

    raw = p.read_bytes()
    fd = detect_faces(raw, margin=0.50)
    if not fd.get("faces"):
        continue

    face = fd["faces"][0]
    face_crop_bgr = face["face_crop"]
    face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(face_crop_rgb)

    inputs = processor(images=pil_face, return_tensors="pt")
    input_tensor = inputs["pixel_values"]

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])[0, :]

    rgb_float = np.float32(face_crop_rgb) / 255.0
    vis = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    safe_name = target_name.replace(" ", "_").lower()
    out_file = Path("backend/uploads") / f"cam_{safe_name}.jpg"
    cv2.imwrite(str(out_file), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Generated CAM for {target_name}: {out_file} (CAM max: {grayscale_cam.max():.3f}, min: {grayscale_cam.min():.3f})")

print("Grad-CAM verification complete!")
