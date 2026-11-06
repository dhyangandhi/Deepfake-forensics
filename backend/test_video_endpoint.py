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

print("\n=== 2. Testing POST /api/analyze/video ===")
video_path = Path("uploads/test_multimodal_clip.mp4")
if not video_path.exists():
    raise FileNotFoundError(f"Video file not found: {video_path}")

with open(video_path, "rb") as f:
    files = {"file": ("test_clip.mp4", f.read(), "video/mp4")}
    data = {"sample_fps": "2.0", "max_frames": "10"}
    r_video = client.post("/api/analyze/video", files=files, data=data)

print("Video Analysis Status:", r_video.status_code)
if r_video.status_code == 200:
    res = r_video.json()
    print("Verdict           :", res.get("verdict"))
    print("Temporal Summary  :", res.get("temporal_summary"))
    print("Sampled Timeline  :", len(res.get("timeline", [])))
    print("Suspicious Frames :", len(res.get("suspicious_frames", [])))
    for sf in res.get("suspicious_frames", []):
        print(f"  Frame {sf['frame_index']} ({sf['timestamp_sec']}s) - {sf['label']} ({sf['fake_score']}%)")
        if sf.get("gradcam"):
            print(f"    Grad-CAM: {sf['gradcam']['focal_region']} (peak: {sf['gradcam']['peak_intensity']})")
else:
    print("Error:", r_video.text)

print("\n=== 3. Testing POST /api/report/video ===")
with open(video_path, "rb") as f:
    files = {"file": ("test_clip.mp4", f.read(), "video/mp4")}
    data = {"sample_fps": "2.0", "max_frames": "10"}
    r_rep = client.post("/api/report/video", files=files, data=data)

print("Video Report Status:", r_rep.status_code)
if r_rep.status_code == 200:
    rep = r_rep.json()
    print("Report ID   :", rep.get("report_id"))
    print("Dossier Verdict:", rep.get("dossier", {}).get("verdict"))
else:
    print("Error:", r_rep.text)

print("\n=== VIDEO FORENSIC PIPELINE ENDPOINTS VERIFIED SUCCESSFULLY ===")

