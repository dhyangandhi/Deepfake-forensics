
from pathlib import Path
import torch
import cv2

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from app.services.face_detector import detect_faces

MODEL_ID = "ashish-001/deepfake-detection-using-ViT"
DATASET = Path("dataset")

print("Loading ViT model...")

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()
model.to("cpu")

print("LABEL MAPPING:", model.config.id2label)

correct = 0
total = 0

for folder, expected in [("real", "REAL"), ("fake", "FAKE")]:
    folder_path = DATASET / folder

    for image_path in folder_path.iterdir():
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        try:
            faces = detect_faces(image_path.read_bytes())

            # Support either a list or dictionary response.
            if isinstance(faces, dict):
                faces = faces.get("faces", [])

            if not faces:
                print("SKIPPED:", image_path.name, "- no face")
                continue

            face_crop = faces[0]["face_crop"]
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            inputs = processor(images=pil_image, return_tensors="pt")

            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=1)[0]

            predicted_index = int(probabilities.argmax().item())
            predicted_label = model.config.id2label[predicted_index]

            confidence = probabilities[predicted_index].item() * 100

            print(
                f"{image_path.name} | "
                f"Expected: {expected} | "
                f"Raw label: {predicted_label} | "
                f"Class index: {predicted_index} | "
                f"Score: {confidence:.2f}%"
            )

            total += 1

        except Exception as error:
            print(f"ERROR: {image_path.name}: {type(error).__name__}: {error}")

print("\n========== RAW MODEL RESULTS ==========")
print("Total faces:", total)
print("=======================================")
print("Do not interpret labels until their meanings are verified.")
