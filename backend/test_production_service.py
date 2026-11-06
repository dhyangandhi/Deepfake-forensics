import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from app.main import analyze_faces_and_classify

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
test_samples = [
    ("REAL Webcam 1", DATASET_DIR / "real" / "WIN_20261104_14_03_36_Pro.jpg"),
    ("REAL Webcam 2", DATASET_DIR / "real" / "WIN_20261104_16_56_02_Pro.jpg"),
    ("REAL Portrait", DATASET_DIR / "real" / "real_harris.jpg"),
    ("FAKE StyleGAN", DATASET_DIR / "fake" / "random-person.jpeg"),
    ("FAKE HiDF Swap", DATASET_DIR / "fake" / "fake_hidf_01.jpg"),
]

print("=== Testing Upgraded DFX Backend Pipeline ===")
for name, p in test_samples:
    if not p.exists():
        print(f"Skipping {name}: not found")
        continue

    content = p.read_bytes()
    res = analyze_faces_and_classify(content)
    
    print(f"\nTarget: {name} ({p.name})")
    print(f"  Status        : {res['status']}")
    print(f"  Model Engine  : {res['model']}")
    print(f"  Face Count    : {res['face_count']}")
    
    for f in res["faces"]:
        c = f["classification"]
        print(f"  Face #{f['face_id']}: Box=({f['x']}, {f['y']}, {f['width']}, {f['height']})")
        print(f"    Label       : {c['label']}")
        print(f"    Confidence  : {c['confidence']}%")
        print(f"    Real Score  : {c['real_score']}% | Fake Score: {c['fake_score']}%")
        print(f"    Uncertainty : {c['uncertainty_score']}%")
        print(f"    Assessment  : {c['assessment']}")
        print(f"    Agreement   : {c['engine_agreement']}")
        print(f"    Indicators  : {c['indicators']}")

print("\n=============================================")
print("DFX Forensic Backend Pipeline verification complete!")

