from pathlib import Path
import cv2
import numpy as np

# Locate YuNet model in backend/models
BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BACKEND_DIR / "models" / "face_detection_yunet_2023mar.onnx"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"YuNet model not found: {MODEL_PATH}")

print(f"Loading YuNet model: {MODEL_PATH}")

face_detector = cv2.FaceDetectorYN.create(
    str(MODEL_PATH),
    "",
    (320, 320),
    0.50,
    0.30,
    5000
)

print("YuNet initialized successfully.")


def extract_contextual_crop(image, x, y, w, h, margin=0.50, target_size=(224, 224)):
    """
    Extracts a face crop with contextual margin to preserve anatomical landmarks
    (hairline, jawline, ears) and prevent high-frequency edge-slicing artifacts.
    """
    height, width = image.shape[:2]
    
    # Calculate margin expansion based on the larger dimension
    pad = int(max(w, h) * margin)
    
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
        
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)


def detect_faces_from_frame(image: np.ndarray, margin: float = 0.50):
    """
    Detects faces directly from a BGR OpenCV NumPy array.
    Used for high-efficiency video frame analysis and image analysis.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array")

    height, width = image.shape[:2]

    face_detector.setInputSize((width, height))
    _, detections = face_detector.detect(image)

    results = []

    if detections is not None:
        for detection in detections:
            x, y, w, h = map(int, detection[:4])
            score = float(detection[14])

            # Filter out tiny detection artifacts
            if w < 40 or h < 40:
                continue

            # Constrain bounding box to image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            if w <= 0 or h <= 0:
                continue

            # Extract contextual crop with margin for reliable classification
            contextual_crop = extract_contextual_crop(image, x, y, w, h, margin=margin)
            if contextual_crop is None:
                continue

            results.append({
                "face_id": len(results) + 1,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "detector_score": round(score, 4),
                "face_crop": contextual_crop
            })

    return {
        "image_width": width,
        "image_height": height,
        "face_count": len(results),
        "faces": results
    }


def detect_faces(image_bytes: bytes, margin: float = 0.50):
    """
    Decodes raw image bytes and executes YuNet face detection.
    """
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image")

    return detect_faces_from_frame(image, margin=margin)
