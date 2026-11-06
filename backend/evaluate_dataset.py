
import csv
from pathlib import Path

import cv2
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from app.services.deepfake_detector import DeepfakeDetectionService
from app.services.face_detector import detect_faces


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_FILE = BASE_DIR / "evaluation_results.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    print("Loading DeepShield model...")
    detector = DeepfakeDetectionService()

    y_true = []
    y_pred = []
    results = []

    total_images = 0
    skipped_images = 0
    total_faces = 0
    error_images = 0

    categories = [
        ("real", "REAL"),
        ("fake", "FAKE")
    ]

    for folder, actual_label in categories:
        folder_path = DATASET_DIR / folder

        if not folder_path.exists():
            print(f"Missing folder: {folder_path}")
            return

        images = sorted([
            p for p in folder_path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        print(f"\nProcessing {len(images)} {actual_label} images...")

        for image_path in images:
            total_images += 1

            try:
                image_bytes = image_path.read_bytes()

                # Step 1: Detect faces
                face_data = detect_faces(image_bytes)
                faces = face_data["faces"]

                print(f"\nImage: {image_path.name}")
                print(f"Detected faces: {len(faces)}")

                for face in faces:
                    print(
                        f"Face {face['face_id']}: "
                        f"x={face['x']}, "
                        f"y={face['y']}, "
                        f"width={face['width']}, "
                        f"height={face['height']}"
                    )

                if not faces:
                    skipped_images += 1
                    print("SKIPPED: No face detected")

                    results.append({
                        "image": str(image_path),
                        "actual": actual_label,
                        "face_id": "",
                        "predicted": "NO_FACE",
                        "confidence": "",
                        "real_score": "",
                        "fake_score": "",
                        "status": "SKIPPED_NO_FACE"
                    })
                    continue

                # Step 2: Classify every detected face
                for face in faces:
                    face_id = face["face_id"]
                    face_crop = face["face_crop"]

                    if face_crop is None or face_crop.size == 0:
                        print(f"Invalid face crop: Face {face_id}")
                        continue

                    rgb_face = cv2.cvtColor(
                        face_crop,
                        cv2.COLOR_BGR2RGB
                    )

                    pil_face = Image.fromarray(rgb_face)

                    # JPEG compression sensitivity test
                    if image_path.name == "random-person.jpeg":
                        import numpy as np

                        print("\n--- JPEG COMPRESSION TEST ---")

                        for quality in [100, 95, 90, 85, 75]:
                            test_path = BASE_DIR / f"test_quality_{quality}.jpg"

                            pil_face.save(
                                test_path,
                                quality=quality
                            )

                            test_image = Image.open(
                                test_path
                            ).convert("RGB")

                            test_prediction = detector.predict(test_image)

                            print(
                                f"JPEG quality {quality}: "
                                f"{test_prediction}"
                            )

                        # Verify lossless PNG consistency
                        png_path = BASE_DIR / "evaluation_face_crop.png"
                        pil_face.save(png_path)

                        reopened_png = Image.open(
                            png_path
                        ).convert("RGB")

                        print(
                            "PNG pixels identical:",
                            np.array_equal(
                                np.array(pil_face),
                                np.array(reopened_png)
                            )
                        )

                        print(
                            "In-memory prediction:",
                            detector.predict(pil_face)
                        )

                        print(
                            "PNG prediction:",
                            detector.predict(reopened_png)
                        )

                    # Normal prediction
                    prediction = detector.predict(pil_face)
                    predicted_label = prediction["label"]

                    y_true.append(actual_label)
                    y_pred.append(predicted_label)
                    total_faces += 1

                    print(
                        f"Face {face_id} | "
                        f"Actual={actual_label} | "
                        f"Predicted={predicted_label} | "
                        f"Confidence={prediction['confidence']}%"
                    )

                    results.append({
                        "image": str(image_path),
                        "actual": actual_label,
                        "face_id": face_id,
                        "predicted": predicted_label,
                        "confidence": prediction["confidence"],
                        "real_score": prediction["real_score"],
                        "fake_score": prediction["fake_score"],
                        "status": "SUCCESS"
                    })

            except Exception as error:
                error_images += 1
                print(f"ERROR: {image_path.name}: {error}")

                results.append({
                    "image": str(image_path),
                    "actual": actual_label,
                    "face_id": "",
                    "predicted": "ERROR",
                    "confidence": "",
                    "real_score": "",
                    "fake_score": "",
                    "status": "ERROR"
                })

    # Step 3: Save results
    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "actual",
                "face_id",
                "predicted",
                "confidence",
                "real_score",
                "fake_score",
                "status"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # Step 4: Evaluation
    print("\n========== FACE-BASED EVALUATION ==========")

    print(f"Total images: {total_images}")
    print(f"Images skipped (no face): {skipped_images}")
    print(f"Images with errors: {error_images}")
    print(f"Successfully classified faces: {total_faces}")

    if not y_true:
        print("\nNo faces available for evaluation.")
        return

    print(f"\nAccuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

    print(
        "Precision: "
        f"{precision_score(y_true, y_pred, pos_label='FAKE', zero_division=0) * 100:.2f}%"
    )

    print(
        "Recall: "
        f"{recall_score(y_true, y_pred, pos_label='FAKE', zero_division=0) * 100:.2f}%"
    )

    print(
        "F1-score: "
        f"{f1_score(y_true, y_pred, pos_label='FAKE', zero_division=0) * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=["REAL", "FAKE"],
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=["REAL", "FAKE"]
    )

    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(matrix)

    print(f"\nResults saved to: {OUTPUT_FILE.resolve()}")
    print("============================================")


if __name__ == "__main__":
    main()
