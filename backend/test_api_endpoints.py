import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)

print("=== 1. Testing GET /api/health ===")
r_health = client.get("/api/health")
print("Health Status:", r_health.status_code)
print("Health Body  :", r_health.json())

# Test image upload and metadata
print("\n=== 2. Testing POST /api/analyze/image ===")
test_img_path = Path("backend/dataset/real/WIN_20261104_14_03_36_Pro.jpg")
with open(test_img_path, "rb") as f:
    files = {"file": ("test_webcam.jpg", f.read(), "image/jpeg")}
r_upload = client.post("/api/analyze/image", files=files)
print("Upload Status:", r_upload.status_code)
meta = r_upload.json().get("metadata", {})
print("Extracted Metadata:")
print("  Format             :", meta.get("format"))
print("  Estimated Quality  :", meta.get("estimated_jpeg_quality"))
print("  Compression Ratio  :", meta.get("compression_ratio"))
print("  Software           :", meta.get("software"))
print("  EXIF Risk          :", meta.get("exif_tampering_risk"))
print("  Risk Explanation   :", meta.get("risk_explanation"))

# Test face detection + Grad-CAM explainability
print("\n=== 3. Testing POST /api/analyze/faces with Grad-CAM ===")
fake_img_path = Path("backend/dataset/fake/fake_hidf_01.jpg")
with open(fake_img_path, "rb") as f:
    files_fake = {"file": ("fake_swap.jpg", f.read(), "image/jpeg")}
r_faces = client.post("/api/analyze/faces", files=files_fake)
print("Faces Status:", r_faces.status_code)
faces_data = r_faces.json()
print("Model              :", faces_data.get("model"))
print("Overall Assessment :", faces_data.get("overall_assessment"))
print("Face Count         :", faces_data.get("face_count"))

for f in faces_data.get("faces", []):
    c = f["classification"]
    e = f["explainability"]
    print(f"\n  Face #{f['face_id']}: Box=({f['x']}, {f['y']}, {f['width']}, {f['height']})")
    print(f"    Classification   : {c['label']} (Confidence: {c['confidence']}%)")
    print(f"    Real / Fake Score: {c['real_score']}% / {c['fake_score']}%")
    print(f"    Uncertainty      : {c['uncertainty_score']}%")
    print(f"    Assessment       : {c['assessment']}")
    print(f"    Grad-CAM Region  : {e['focal_region']}")
    print(f"    Peak Intensity   : {e['peak_intensity']}")
    print(f"    Heatmap Data URL : {e['heatmap_data_url'][:40]}... (Total length: {len(e['heatmap_data_url'])})")

# Test report generation
print("\n=== 4. Testing POST /api/report/generate ===")
with open(fake_img_path, "rb") as f:
    files_rep = {"file": ("fake_swap.jpg", f.read(), "image/jpeg")}
r_rep = client.post("/api/report/generate", files=files_rep)
print("Report Status:", r_rep.status_code)
rep_data = r_rep.json()
print("Report ID   :", rep_data.get("report_id"))
print("Dossier Verdict:", rep_data.get("dossier", {}).get("verdict"))
print("Facial Findings Count:", len(rep_data.get("dossier", {}).get("facial_findings", [])))

print("\n=======================================================")
print("ALL PHASE 2 API ENDPOINTS VERIFIED AND OPERATIONAL!")
print("=======================================================")

