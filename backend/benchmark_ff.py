import os
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
MODEL_ID = "HrutikAdsare/deepfake-detector-faceforensics"

def extract_face_with_margin(img_bgr, face, margin=0.30):
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

def main():
    print(f"Loading FaceForensics candidate: {MODEL_ID}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    model.to("cpu")
    print("Label Mapping:", model.config.id2label)

    # In HrutikAdsare: id2label is {0: 'Real', 1: 'Deepfake'}
    real_idx = next((k for k, v in model.config.id2label.items() if "real" in v.lower()), 0)
    fake_idx = next((k for k, v in model.config.id2label.items() if "fake" in v.lower() or "deepfake" in v.lower()), 1)

    y_true = []
    y_pred = []

    for folder, label_code in [("real", 0), ("fake", 1)]:
        folder_path = DATASET_DIR / folder
        images = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        
        print(f"\nTesting {folder.upper()} images...")
        for p in images:
            raw = p.read_bytes()
            img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            fd = detect_faces(raw)
            faces = fd.get("faces", [])
            if not faces:
                print(f"  SKIPPED: {p.name} (No face)")
                continue

            crop_pil = extract_face_with_margin(img_bgr, faces[0], margin=0.30)
            inputs = processor(images=crop_pil, return_tensors="pt")
            with torch.inference_mode():
                probs = torch.softmax(model(**inputs).logits, dim=1)[0]

            fake_prob = probs[fake_idx].item()
            real_prob = probs[real_idx].item()
            pred_code = 1 if fake_prob >= 0.50 else 0
            status = "PASS" if pred_code == label_code else "FAIL"

            y_true.append(label_code)
            y_pred.append(pred_code)
            print(f"  [{status}] {p.name:25} | Pred: {'FAKE' if pred_code==1 else 'REAL'} | Fake: {fake_prob*100:5.2f}% | Real: {real_prob*100:5.2f}%")

    if y_true:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        print("\n" + "="*50)
        print(f"RESULTS FOR {MODEL_ID}:")
        print(f"Accuracy : {acc*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall   : {rec*100:.2f}%")
        print(f"F1-Score : {f1*100:.2f}%")
        print("Confusion Matrix:")
        print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
        print("="*50)

if __name__ == "__main__":
    main()

