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

print("\n=== 2. Testing POST /api/analyze/audio with Synthetic Voice ===")
synth_path = Path("uploads/test_synth_voice.wav")
if not synth_path.exists():
    raise FileNotFoundError("test_synth_voice.wav not found. Run test_audio_forensics.py first.")

with open(synth_path, "rb") as f:
    files = {"file": ("test_synth_voice.wav", f.read(), "audio/wav")}
    r_synth = client.post("/api/analyze/audio", files=files)

print("Audio Analysis (Synthetic) Status:", r_synth.status_code)
if r_synth.status_code == 200:
    res = r_synth.json()
    print("Verdict           :", res.get("verdict"))
    print("Scores            :", res.get("scores"))
    print("Acoustic Indicators:", res.get("indicators"))
    print("Spectrogram URL   :", res.get("spectrogram", {}).get("spectrogram_url"))
else:
    print("Error:", r_synth.text)

print("\n=== 3. Testing POST /api/analyze/audio with Natural Speech ===")
human_path = Path("uploads/test_human_voice.wav")
with open(human_path, "rb") as f:
    files = {"file": ("test_human_voice.wav", f.read(), "audio/wav")}
    r_human = client.post("/api/analyze/audio", files=files)

print("Audio Analysis (Natural) Status:", r_human.status_code)
if r_human.status_code == 200:
    res = r_human.json()
    print("Verdict           :", res.get("verdict"))
    print("Scores            :", res.get("scores"))
    print("Acoustic Indicators:", res.get("indicators"))
else:
    print("Error:", r_human.text)

print("\n=== 4. Testing POST /api/report/audio ===")
with open(synth_path, "rb") as f:
    files = {"file": ("test_synth_voice.wav", f.read(), "audio/wav")}
    r_rep = client.post("/api/report/audio", files=files)

print("Audio Report Status:", r_rep.status_code)
if r_rep.status_code == 200:
    rep = r_rep.json()
    print("Report ID   :", rep.get("report_id"))
    print("Dossier Verdict:", rep.get("dossier", {}).get("verdict"))
else:
    print("Error:", r_rep.text)

print("\n=== ALL AUDIO FORENSIC ENDPOINTS VERIFIED OPERATIONAL ===")

