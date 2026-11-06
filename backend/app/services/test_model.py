
import sys
from PIL import Image
from app.services.deepfake_detector import DeepfakeDetectionService

if len(sys.argv) < 2:
    print("Usage: python -m app.services.test_model <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

detector = DeepfakeDetectionService()

image = Image.open(image_path)
result = detector.predict(image)

print("\n========== DEEPFAKE ANALYSIS ==========")
print("Image:", image_path)
print("Prediction:", result["label"])
print("Confidence:", result["confidence"], "%")
print("Real score:", result["real_score"], "%")
print("Fake score:", result["fake_score"], "%")
print("======================================")
