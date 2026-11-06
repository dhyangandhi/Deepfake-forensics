
from pathlib import Path
import io
import cv2
from PIL import Image

from app.services.deepfake_detector import DeepfakeDetectionService
from app.services.face_detector import detect_faces

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "dataset" / "fake" / "random-person.jpeg"


def main():
    detector = DeepfakeDetectionService()

    face_data = detect_faces(IMAGE_PATH.read_bytes())
    faces = face_data["faces"]

    if not faces:
        print("ERROR: No face detected.")
        return

    face_crop = faces[0]["face_crop"]
    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    original = Image.fromarray(rgb_face).resize((224, 224))

    print("\n========== FACE CROP COMPRESSION TEST ==========")

    tests = [
        ("Original", None),
        ("JPEG 100", 100),
        ("JPEG 95", 95),
        ("JPEG 90", 90),
        ("JPEG 85", 85),
        ("JPEG 75", 75),
        ("JPEG 50", 50),
        ("PNG", "PNG"),
    ]

    for name, setting in tests:

        if setting is None:
            image = original.copy()

        else:
            buffer = io.BytesIO()

            if setting == "PNG":
                original.save(buffer, format="PNG")
            else:
                original.save(
                    buffer,
                    format="JPEG",
                    quality=setting
                )

            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")

        result = detector.predict(image)

        print(
            f"{name:12} | {result['label']:5} | "
            f"Confidence: {result['confidence']:6.2f}% | "
            f"Real: {result['real_score']:6.2f}% | "
            f"Fake: {result['fake_score']:6.2f}%"
        )

    print("================================================")


if __name__ == "__main__":
    main()
