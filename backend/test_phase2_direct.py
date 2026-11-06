import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from app.services.metadata_extractor import extract_forensic_metadata
from app.main import analyze_faces_and_explain

test_cases = [
    ("REAL Webcam", Path("backend/dataset/real/WIN_20261104_14_03_36_Pro.jpg")),
    ("FAKE Swap", Path("backend/dataset/fake/fake_hidf_01.jpg")),
    ("FAKE StyleGAN", Path("backend/dataset/fake/random-person.jpeg")),
]

print("=== Direct Verification of Phase 2 Modules ===")

for name, p in test_cases:
    if not p.exists():
        continue

    raw = p.read_bytes()
    print(f"\n==========================================")
    print(f"Target: {name} ({p.name})")
    
    # 1. Metadata analysis
    meta = extract_forensic_metadata(raw)
    print(f"--- Metadata Analysis ---")
    print(f"  Format             : {meta.get('format')}")
    print(f"  Dimensions         : {meta.get('width')}x{meta.get('height')}")
    print(f"  Compression Ratio  : {meta.get('compression_ratio')}")
    print(f"  Estimated Quality  : {meta.get('estimated_jpeg_quality')}")
    print(f"  Software Tag       : {meta.get('software')}")
    print(f"  Editing Detected   : {meta.get('editing_detected')}")
    print(f"  Tampering Risk     : {meta.get('exif_tampering_risk')}")
    print(f"  Risk Explanation   : {meta.get('risk_explanation')}")
    if meta.get("ela_metrics"):
        print(f"  ELA Metrics        : {meta.get('ela_metrics')}")

    # 2. Face detection, dual-engine classification & Grad-CAM
    analysis = analyze_faces_and_explain(raw)
    print(f"\n--- Facial & Grad-CAM Forensics ---")
    print(f"  Model Engine       : {analysis.get('model')}")
    print(f"  Overall Assessment : {analysis.get('overall_assessment')}")
    print(f"  Face Count         : {analysis.get('face_count')}")

    for f in analysis.get("faces", []):
        c = f["classification"]
        e = f["explainability"]
        print(f"  Face #{f['face_id']}: Box=({f['x']}, {f['y']}, {f['width']}, {f['height']})")
        print(f"    Verdict          : {c['label']} (Confidence: {c['confidence']}%)")
        print(f"    Real / Fake Score: {c['real_score']}% / {c['fake_score']}%")
        print(f"    Uncertainty      : {c['uncertainty_score']}%")
        print(f"    Assessment       : {c['assessment']}")
        print(f"    Engine Consensus : {c['engine_agreement']}")
        print(f"    Focal Region     : {e['focal_region']}")
        print(f"    Peak Intensity   : {e['peak_intensity']}")
        print(f"    Heatmap Data URL : {e['heatmap_data_url'][:35]}... (len={len(e['heatmap_data_url'])})")
        print(f"    Saved Artifact   : {e.get('heatmap_url')}")

print("\n==========================================")
print("Phase 2 Core Engine Direct Verification Complete!")

