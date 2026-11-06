from pathlib import Path
import io
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from app.services.face_detector import detect_faces

MODEL_ID = "dima806/deepfake_vs_real_image_detection"
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

def load_candidate_model():
    print(f"Loading candidate model: {MODEL_ID}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    model.to("cpu")
    print("Model loaded successfully.")
    print("Label Mapping (id2label):", model.config.id2label)
    return processor, model

def predict_crop(processor, model, pil_face):
    inputs = processor(images=pil_face, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0]
    
    predicted_idx = int(probabilities.argmax().item())
    raw_label = model.config.id2label[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100
    
    # Extract Real and Fake scores directly based on verified id2label
    # Typically: 0 -> Real, 1 -> Fake
    real_idx = next((k for k, v in model.config.id2label.items() if v.lower() == "real"), 0)
    fake_idx = next((k for k, v in model.config.id2label.items() if v.lower() == "fake"), 1)
    
    real_score = probabilities[real_idx].item() * 100
    fake_score = probabilities[fake_idx].item() * 100
    
    return {
        "label": raw_label.upper(),
        "confidence": confidence,
        "real_score": real_score,
        "fake_score": fake_score
    }

def run_dataset_benchmark(processor, model):
    print("\n================== DATASET BENCHMARK ==================")
    categories = [("real", "REAL"), ("fake", "FAKE")]
    results = []

    for folder, expected in categories:
        folder_path = DATASET_DIR / folder
        if not folder_path.exists():
            continue

        images = sorted([
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ])

        for img_path in images:
            image_bytes = img_path.read_bytes()
            face_data = detect_faces(image_bytes)
            faces = face_data.get("faces", [])

            if not faces:
                print(f"SKIPPED: {img_path.name} (No face detected)")
                continue

            for idx, face in enumerate(faces):
                face_crop = face["face_crop"]
                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_face = Image.fromarray(rgb_face)

                res = predict_crop(processor, model, pil_face)
                status = "PASS" if res["label"] == expected else "FAIL"

                print(
                    f"[{status}] {img_path.name} (Face {face['face_id']}) | "
                    f"Expected: {expected} | Pred: {res['label']} | "
                    f"Conf: {res['confidence']:.2f}% | "
                    f"Real: {res['real_score']:.2f}% | Fake: {res['fake_score']:.2f}%"
                )
                results.append((img_path.name, expected, res["label"], res["confidence"]))

    print("=======================================================")
    return results

def run_compression_sweep(processor, model):
    test_image_path = DATASET_DIR / "fake" / "random-person.jpeg"
    if not test_image_path.exists():
        print(f"Warning: {test_image_path} does not exist.")
        return

    print("\n========== CANDIDATE JPEG COMPRESSION SWEEP ==========")
    print(f"Target: {test_image_path.name} (Ground Truth: FAKE)")

    face_data = detect_faces(test_image_path.read_bytes())
    faces = face_data.get("faces", [])
    if not faces:
        print("ERROR: No face detected in test image.")
        return

    face_crop = faces[0]["face_crop"]
    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    original_crop = Image.fromarray(rgb_face).resize((224, 224))

    tests = [
        ("Original", None),
        ("PNG", "PNG"),
        ("JPEG 100", 100),
        ("JPEG 95", 95),
        ("JPEG 90", 90),
        ("JPEG 85", 85),
        ("JPEG 75", 75),
        ("JPEG 50", 50),
    ]

    for name, setting in tests:
        if setting is None:
            img = original_crop.copy()
        else:
            buf = io.BytesIO()
            if setting == "PNG":
                original_crop.save(buf, format="PNG")
            else:
                original_crop.save(buf, format="JPEG", quality=setting)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        res = predict_crop(processor, model, img)
        match = "OK" if res["label"] == "FAKE" else "MISCLASSIFIED"
        print(
            f"{name:10} | [{match:13}] | Pred: {res['label']:5} | "
            f"Conf: {res['confidence']:6.2f}% | "
            f"Real: {res['real_score']:6.2f}% | Fake: {res['fake_score']:6.2f}%"
        )
    print("=======================================================")

if __name__ == "__main__":
    processor, model = load_candidate_model()
    run_dataset_benchmark(processor, model)
    run_compression_sweep(processor, model)

