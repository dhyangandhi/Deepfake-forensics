import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

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
test_files = [
    dataset_dir / "real" / "WIN_20261104_14_03_36_Pro.jpg",
    dataset_dir / "real" / "WIN_20261104_16_56_02_Pro.jpg",
    dataset_dir / "fake" / "random-person.jpeg"
]

out_file = Path(__file__).resolve().parent / "margin_results.txt"
with open(out_file, "w", encoding="utf-8") as out:
    def log(msg):
        print(msg, flush=True)
        out.write(msg + "\n")
        out.flush()

    log("\n--- Testing Margin & Aspect Ratio Variations ---")
    for p in test_files:
        if not p.exists():
            continue
        raw_bytes = p.read_bytes()
        img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        H, W = img_bgr.shape[:2]
        
        # Full image prediction
        full_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        full_res = predict(full_pil)
        log(f"\nTarget: {p.name}")
        log(f"  Full Image (no crop): Real={full_res['Real']}% | Fake={full_res['Fake']}%")
        
        fd = detect_faces(raw_bytes)
        faces = fd.get("faces", [])
        if not faces:
            log("  No face detected.")
            continue
        
        f = faces[0]
        fx, fy, fw, fh = f["x"], f["y"], f["width"], f["height"]
        log(f"  Face Box: x={fx}, y={fy}, w={fw}, h={fh}")
        
        for margin in [0.0, 0.20, 0.40, 0.60, 0.80, 1.00]:
            mx = int(fw * margin)
            my = int(fh * margin)
            x1 = max(0, fx - mx)
            y1 = max(0, fy - my)
            x2 = min(W, fx + fw + mx)
            y2 = min(H, fy + fh + my)
            
            crop = img_bgr[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            res = predict(crop_pil)
            log(f"  Margin {int(margin*100):3}% ({crop.shape[1]}x{crop.shape[0]}): Real={res['Real']:6.2f}% | Fake={res['Fake']:6.2f}%")
