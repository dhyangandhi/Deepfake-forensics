import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Set environment
os.environ["MPLCONFIGDIR"] = str(Path(__file__).parent / "uploads")
sys.path.insert(0, str(Path(__file__).parent))

from app.services.video_processor import VideoForensicService

def create_synthetic_test_video(output_path: Path):
    real_path = Path("dataset/real/WIN_20261104_14_03_36_Pro.jpg")
    fake_path = Path("dataset/fake/random-person.jpeg")

    img_real = cv2.imread(str(real_path))
    img_fake = cv2.imread(str(fake_path))

    target_size = (640, 480)
    img_real = cv2.resize(img_real, target_size)
    img_fake = cv2.resize(img_fake, target_size)

    # 10 FPS, 3 seconds = 30 frames
    # Frames 0-14: Real face
    # Frames 15-29: Synthetic face
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 10.0, target_size)

    for _ in range(15):
        out.write(img_real)
    for _ in range(15):
        out.write(img_fake)

    out.release()
    print(f"Created test video: {output_path} (30 frames, 10 FPS, 3.0s)")

def main():
    test_video_path = Path("uploads/test_multimodal_clip.mp4")
    test_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_synthetic_test_video(test_video_path)

    service = VideoForensicService()
    print("Running video forensic analysis...")
    result = service.analyze_video(
        str(test_video_path),
        output_dir=Path("uploads"),
        sample_fps=2.0,  # 2 samples per second -> ~6 frames analyzed
        max_frames=10
    )

    print("\n--- FORENSIC VIDEO ANALYSIS RESULT ---")
    print(f"Status: {result['status']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Video Metadata: {result['video_metadata']}")
    print(f"Temporal Summary: {result['temporal_summary']}")
    print(f"Timeline entries: {len(result['timeline'])}")
    print(f"Suspicious frames flagged: {len(result['suspicious_frames'])}")
    for sf in result["suspicious_frames"]:
        print(f"  Rank #{sf['rank']}: Frame {sf['frame_index']} ({sf['timestamp_sec']}s) - Score: {sf['fake_score']}% [{sf['label']}]")
        if sf.get("gradcam"):
            print(f"    Grad-CAM: {sf['gradcam']['focal_region']} (peak: {sf['gradcam']['peak_intensity']})")

if __name__ == "__main__":
    main()

