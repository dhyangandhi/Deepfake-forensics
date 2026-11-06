import sys
import csv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoImageProcessor, AutoModelForImageClassification
from app.services.face_detector import detect_faces

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

MODELS_TO_TEST = [
    {
        "name": "umm-maybe/AI-image-detector",
        "arch": "Swin",
        "fake_label_key": "artificial",
        "real_label_key": "human",
        "margin": 0.40
    },
    {
        "name": "prithivMLmods/Deep-Fake-Detector-v2-Model",
        "arch": "ViT",
        "fake_label_key": "deepfake",
        "real_label_key": "realism",
        "margin": 0.40
    }
]

def extract_face(img_bgr, face, margin=0.40):
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

def evaluate_model(cfg):
    model_id = cfg["name"]
    print(f"\n=======================================================")
    print(f"Loading: {model_id} ({cfg['arch']})...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    model.to("cpu")
    print("id2label:", model.config.id2label)

    # Resolve indices
    fake_idx = next(k for k, v in model.config.id2label.items() if cfg["fake_label_key"] in v.lower())
    real_idx = next(k for k, v in model.config.id2label.items() if cfg["real_label_key"] in v.lower())

    y_true = []
    y_pred = []

    for folder, true_label, true_code in [("real", "REAL", 0), ("fake", "FAKE", 1)]:
        folder_path = DATASET_DIR / folder
        images = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        
        for p in images:
            raw = p.read_bytes()
            img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            fd = detect_faces(raw)
            faces = fd.get("faces", [])
            if not faces:
                print(f"  SKIPPED: {p.name}")
                continue

            crop = extract_face(img_bgr, faces[0], margin=cfg["margin"])
            inputs = processor(images=crop, return_tensors="pt")
            with torch.inference_mode():
                probs = torch.softmax(model(**inputs).logits, dim=1)[0]

            fake_prob = probs[fake_idx].item()
            real_prob = probs[real_idx].item()
            pred_code = 1 if fake_prob >= 0.50 else 0
            status = "PASS" if pred_code == true_code else "FAIL"

            y_true.append(true_code)
            y_pred.append(pred_code)
            print(f"  [{status}] {p.name:25} | Pred: {'FAKE' if pred_code==1 else 'REAL':4} | Fake: {fake_prob*100:5.2f}% | Real: {real_prob*100:5.2f}%")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("-" * 55)
    print(f"SUMMARY FOR {model_id}:")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Fake Prec: {prec*100:.2f}% | Fake Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%")
    print(f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]} | FN={cm[1][0]}, TP={cm[1][1]}")
    print("=======================================================")

if __name__ == "__main__":
    for cfg in MODELS_TO_TEST:
        evaluate_model(cfg)

