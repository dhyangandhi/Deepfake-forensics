import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import csv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from transformers import AutoImageProcessor, AutoModelForImageClassification
from app.services.face_detector import detect_faces

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_CSV = BASE_DIR / "evaluation_results_v2.csv"
MODEL_ID = "dima806/deepfake_vs_real_image_detection"

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

def main():
    print(f"Loading candidate model: {MODEL_ID}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    model.to("cpu")
    print("Model initialized successfully.")

    y_true = []
    y_pred = []
    y_scores = []
    records = []

    categories = [
        ("real", 0, "REAL"),
        ("fake", 1, "FAKE")
    ]

    print("\nStarting evaluation over dataset...")
    for folder, label_code, label_name in categories:
        folder_path = DATASET_DIR / folder
        if not folder_path.exists():
            continue

        images = sorted([
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ])

        print(f"\nEvaluating {len(images)} images in '{folder}' (Ground Truth: {label_name})...")
        for img_path in images:
            try:
                raw_bytes = img_path.read_bytes()
                img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    print(f"  [CORRUPT] {img_path.name}")
                    continue

                face_data = detect_faces(raw_bytes)
                faces = face_data.get("faces", [])

                if not faces:
                    print(f"  [SKIPPED - NO FACE] {img_path.name}")
                    records.append({
                        "filename": img_path.name,
                        "ground_truth": label_name,
                        "face_detected": False,
                        "predicted_label": "N/A",
                        "real_score": 0.0,
                        "fake_score": 0.0,
                        "status": "NO_FACE"
                    })
                    continue

                # Evaluate primary face (highest confidence)
                primary_face = faces[0]
                face_crop_pil = extract_face_with_margin(img_bgr, primary_face, margin=0.60)

                inputs = processor(images=face_crop_pil, return_tensors="pt")
                with torch.inference_mode():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)[0]

                real_prob = probs[0].item()
                fake_prob = probs[1].item()

                pred_code = 1 if fake_prob >= 0.50 else 0
                pred_label = "FAKE" if pred_code == 1 else "REAL"
                status = "PASS" if pred_code == label_code else "FAIL"

                y_true.append(label_code)
                y_pred.append(pred_code)
                y_scores.append(fake_prob)

                print(
                    f"  [{status:4}] {img_path.name:25} | "
                    f"Pred: {pred_label:4} | Fake Prob: {fake_prob*100:5.2f}% | "
                    f"Real Prob: {real_prob*100:5.2f}%"
                )

                records.append({
                    "filename": img_path.name,
                    "ground_truth": label_name,
                    "face_detected": True,
                    "predicted_label": pred_label,
                    "real_score": round(real_prob * 100, 2),
                    "fake_score": round(fake_prob * 100, 2),
                    "status": status
                })

            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")

    # Write CSV
    if records:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"\nDetailed metrics written to: {OUTPUT_CSV}")

    # Calculate statistics
    if y_true and len(set(y_true)) > 1:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = float("nan")

        print("\n" + "=" * 60)
        print("           FORENSIC VALIDATION REPORT (ViT + 60% Margin)")
        print("=" * 60)
        print(f"Total Evaluated Samples : {len(y_true)}")
        print(f"Accuracy                : {acc * 100:.2f}%")
        print(f"Fake Precision          : {prec * 100:.2f}%")
        print(f"Fake Recall (Sensitivity: {rec * 100:.2f}%")
        print(f"Fake F1-Score           : {f1 * 100:.2f}%")
        print(f"ROC-AUC Score           : {auc:.4f}")
        print("-" * 60)
        print("Confusion Matrix:")
        print(f"  True Real  (TN): {cm[0][0]:2d}  | False Fake (FP): {cm[0][1]:2d}")
        print(f"  False Real (FN): {cm[1][0]:2d}  | True Fake  (TP): {cm[1][1]:2d}")
        print("-" * 60)
        print("\nFull Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))
        print("=" * 60)
    else:
        print("Insufficient diverse classes to compute full validation metrics.")

if __name__ == "__main__":
    main()

