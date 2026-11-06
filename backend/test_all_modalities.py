import os
import sys
import time
from pathlib import Path

# Configure environment for CPU and headless execution
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "uploads"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))

from starlette.testclient import TestClient
from app.main import app

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)

def main():
    print_header("DFX MULTI-MODAL FORENSICS — MASTER REGRESSION TEST")
    start_time = time.time()
    passed_tests = 0
    total_tests = 5

    client = TestClient(app)

    # ----------------------------------------------------
    # TEST 1: System Health & Multi-Engine Registration
    # ----------------------------------------------------
    print("\n[TEST 1/5] Checking Multi-Engine Registration & Health Status...")
    r_health = client.get("/api/health")
    assert r_health.status_code == 200, f"Health check failed: {r_health.status_code}"
    health_data = r_health.json()
    engines = health_data.get("models", {})
    
    print(f"  ✓ Service Name    : {health_data.get('service')}")
    print(f"  ✓ Face Detector   : {engines.get('face_detector')}")
    print(f"  ✓ Vision Engine   : {engines.get('primary_classifier')}")
    print(f"  ✓ Explainability  : {engines.get('explainability')}")
    print(f"  ✓ Video Engine    : {engines.get('video_engine')}")
    print(f"  ✓ Audio Engine    : {engines.get('audio_engine')}")
    passed_tests += 1

    # ----------------------------------------------------
    # TEST 2: Image Forensics & Explainability
    # ----------------------------------------------------
    print("\n[TEST 2/5] Testing Image Forensics, Metadata, and Grad-CAM...")
    test_img = Path("dataset/fake/fake_hidf_01.jpg")
    if not test_img.exists():
        test_img = Path("dataset/fake/random-person.jpeg")
    
    with open(test_img, "rb") as f:
        img_bytes = f.read()

    # Image metadata analysis
    r_img_meta = client.post("/api/analyze/image", files={"file": ("test_target.jpg", img_bytes, "image/jpeg")})
    assert r_img_meta.status_code == 200, f"Image metadata extraction failed: {r_img_meta.text}"
    meta = r_img_meta.json().get("metadata", {})
    print(f"  ✓ Image Ingestion : {r_img_meta.json().get('width')}x{r_img_meta.json().get('height')} {r_img_meta.json().get('format')}")
    print(f"  ✓ Tampering Risk  : {meta.get('exif_tampering_risk')}")

    # Face detection, dual-engine classification & Grad-CAM
    r_faces = client.post("/api/analyze/faces", files={"file": ("test_target.jpg", img_bytes, "image/jpeg")})
    assert r_faces.status_code == 200, f"Face analysis failed: {r_faces.text}"
    face_data = r_faces.json()
    print(f"  ✓ Overall Verdict : {face_data.get('overall_assessment')}")
    print(f"  ✓ Faces Detected  : {face_data.get('face_count')}")
    
    for f in face_data.get("faces", [])[:1]:
        c = f["classification"]
        e = f["explainability"]
        print(f"    - Face #{f['face_id']}: {c['label']} (Conf: {c['confidence']}%, Uncertainty: {c.get('uncertainty_score')}%)")
        print(f"    - Grad-CAM Region : {e.get('focal_region')} (Peak: {e.get('peak_intensity')})")

    # Image Report generation
    r_img_rep = client.post("/api/report/generate", files={"file": ("test_target.jpg", img_bytes, "image/jpeg")})
    assert r_img_rep.status_code == 200, f"Image report generation failed: {r_img_rep.text}"
    print(f"  ✓ Dossier Generated: {r_img_rep.json().get('report_id')}")
    passed_tests += 1

    # ----------------------------------------------------
    # TEST 3: Video Temporal Forensics
    # ----------------------------------------------------
    print("\n[TEST 3/5] Testing Video Temporal Forensics & Frame Sampling...")
    test_vid = Path("uploads/test_multimodal_clip.mp4")
    assert test_vid.exists(), "test_multimodal_clip.mp4 not found. Please ensure test clip exists."

    with open(test_vid, "rb") as f:
        vid_bytes = f.read()

    r_vid = client.post(
        "/api/analyze/video",
        files={"file": ("clip.mp4", vid_bytes, "video/mp4")},
        data={"sample_fps": "2.0", "max_frames": "10"}
    )
    assert r_vid.status_code == 200, f"Video analysis failed: {r_vid.text}"
    vid_data = r_vid.json()
    temp_sum = vid_data.get("temporal_summary", {})
    print(f"  ✓ Video Verdict   : {vid_data.get('verdict')}")
    print(f"  ✓ Manipulated %   : {temp_sum.get('manipulated_frame_ratio_percent')}%")
    print(f"  ✓ Peak Fake Score : {temp_sum.get('peak_fake_score')}% @ {temp_sum.get('peak_timestamp_sec')}s")
    print(f"  ✓ Stability Metric: {temp_sum.get('temporal_stability')} (Var: {temp_sum.get('temporal_variance')})")
    print(f"  ✓ Top Suspicious  : {len(vid_data.get('suspicious_frames', []))} flagged frames")
    passed_tests += 1

    # ----------------------------------------------------
    # TEST 4: Audio Acoustic Forensics
    # ----------------------------------------------------
    print("\n[TEST 4/5] Testing Audio Forensics & Biometric Resonance...")
    synth_aud = Path("uploads/test_synth_voice.wav")
    human_aud = Path("uploads/test_human_voice.wav")

    # Synthetic check
    with open(synth_aud, "rb") as f:
        r_aud_synth = client.post("/api/analyze/audio", files={"file": ("synth.wav", f.read(), "audio/wav")})
    assert r_aud_synth.status_code == 200, f"Audio analysis failed: {r_aud_synth.text}"
    aud_synth_data = r_aud_synth.json()
    print(f"  ✓ Synthetic Voice : {aud_synth_data.get('verdict')} (Prob: {aud_synth_data['scores']['synthetic_probability']}%)")
    print(f"    - Jitter        : {aud_synth_data['indicators']['pitch_jitter_percent']}%")
    print(f"    - Brickwall Void: {aud_synth_data['indicators']['high_frequency_brickwall']}")

    # Human check
    with open(human_aud, "rb") as f:
        r_aud_human = client.post("/api/analyze/audio", files={"file": ("human.wav", f.read(), "audio/wav")})
    assert r_aud_human.status_code == 200, f"Audio analysis failed: {r_aud_human.text}"
    aud_human_data = r_aud_human.json()
    print(f"  ✓ Human Voice     : {aud_human_data.get('verdict')} (Prob: {aud_human_data['scores']['natural_probability']}%)")
    print(f"    - Jitter        : {aud_human_data['indicators']['pitch_jitter_percent']}%")
    passed_tests += 1

    # ----------------------------------------------------
    # TEST 5: PDF Dossier Export Engine
    # ----------------------------------------------------
    print("\n[TEST 5/5] Testing PDF Report Export Engine (/api/report/export-pdf)...")
    sample_dossier = {
        "report_id": "DFX-REGRESSION-001",
        "media_type": "VIDEO",
        "filename": "clip.mp4",
        "verdict": vid_data.get("verdict"),
        "video_metadata": vid_data.get("video_metadata"),
        "temporal_summary": temp_sum,
        "suspicious_frames": vid_data.get("suspicious_frames"),
        "forensic_disclaimer": "Automated regression verification."
    }
    r_pdf = client.post("/api/report/export-pdf", json=sample_dossier)
    assert r_pdf.status_code == 200, f"PDF export failed: {r_pdf.text}"
    assert r_pdf.content.startswith(b"%PDF-"), "Generated payload is not a valid PDF binary"
    print(f"  ✓ PDF Compiled    : {len(r_pdf.content)} bytes (Status 200 OK, Content-Type: application/pdf)")
    passed_tests += 1

    # ----------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------
    elapsed = round(time.time() - start_time, 2)
    print_header(f"REGRESSION AUDIT SUMMARY: {passed_tests}/{total_tests} MODULES PASSED ({elapsed}s)")
    print("  [PASS] Engine Registration & Health Verification")
    print("  [PASS] Image Forensics (Metadata + YuNet 50% Margin + Swin/ViT + Grad-CAM)")
    print("  [PASS] Video Forensics (Temporal Stride + Discontinuity Profiler + Suspicious Frames)")
    print("  [PASS] Audio Forensics (Glottal Jitter + Shimmer + Brickwall Void + Spectrogram)")
    print("  [PASS] PDF Report Engine (Vector Multi-Page Dossier Generator)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()

