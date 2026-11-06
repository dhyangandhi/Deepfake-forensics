import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import io
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from app.services.face_detector import detect_faces

MODEL_ID = "dima806/deepfake_vs_real_image_detection"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

def predict(img_pil):
    inputs = processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    return {"Real": round(probs[0].item() * 100, 2), "Fake": round(probs[1].item() * 100, 2)}

dataset_dir = Path(__file__).resolve().parent / "dataset"
test_cases = [
    ("REAL_1", dataset_dir / "real" / "WIN_20261104_14_03_36_Pro.jpg", "REAL"),
    ("REAL_2", dataset_dir / "real" / "WIN_20261104_16_56_02_Pro.jpg", "REAL"),
    ("FAKE_1", dataset_dir / "fake" / "random-person.jpeg", "FAKE"),
]

def extract_face_with_margin(img_bgr, face, margin=0.60):
    H, W = img_bgr.shape[:2]
    fx, fy, fw, fh = face["x"], face["y"], face["width"], face["height"]
    mx = int(fw * margin)
    my = int(fh * margin)
    x1 = max(0, fx - mx)
    y1 = max(0, fy - my)
    x2 = min(W, fx + fw + mx)
    y2 = min(H, fy + fh + my)
    crop = img_bgr[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

print("==========================================================================")
print("JPEG COMPRESSION SWEEP WITH 60% MARGIN (FORENSIC EVALUATION)")
print("==========================================================================")

qualities = [
    ("Original", None),
    ("PNG", "PNG"),
    ("JPEG 100", 100),
    ("JPEG 95", 95),
    ("JPEG 90", 90),
    ("JPEG 85", 85),
    ("JPEG 75", 75),
    ("JPEG 50", 50),
]

for label, path, expected in test_cases:
    if not path.exists():
        continue
    raw = path.read_bytes()
    img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    fd = detect_faces(raw)
    face = fd["faces"][0]
    crop_pil = extract_face_with_margin(img_bgr, face, margin=0.60)
    
    print(f"\n--- Target: {label} ({path.name}) | Expected: {expected} ---")
    for q_name, q_val in qualities:
        if q_val is None:
            tested_img = crop_pil.copy()
        else:
            buf = io.BytesIO()
            if q_val == "PNG":
                crop_pil.save(buf, format="PNG")
            else:
                crop_pil.save(buf, format="JPEG", quality=q_val)
            buf.seek(0)
            tested_img = Image.open(buf).convert("RGB")
            
        res = predict(tested_img)
        pred_label = "REAL" if res["Real"] >= 50 else "FAKE"
        status = "PASS" if pred_label == expected else "FAIL"
        conf = res["Real"] if pred_label == "REAL" else res["Fake"]
        print(f"  {q_name:10} | [{status:4}] | Pred: {pred_label:4} ({conf:6.2f}%) | Real: {res['Real']:6.2f}% | Fake: {res['Fake']:6.2f}%")

