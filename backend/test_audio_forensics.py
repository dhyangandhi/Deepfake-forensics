import os
import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

# Set environment
os.environ["MPLCONFIGDIR"] = str(Path(__file__).parent / "uploads")
sys.path.insert(0, str(Path(__file__).parent))

from app.services.audio_detector import AudioForensicService

def generate_test_audio_files(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    sr = 24000
    duration = 2.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Synthetic TTS Simulation: rigid pitch, zero jitter, artificial digital silence floor
    f0 = 160.0
    synth_wave = (
        0.6 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    # Add absolute zero mathematical silence
    synth_wave[int(sr * 1.8):] = 0.0
    synth_wave_int16 = (synth_wave * 32767).astype(np.int16)

    synth_path = target_dir / "test_synth_voice.wav"
    wavfile.write(str(synth_path), sr, synth_wave_int16)

    # 2. Natural Human Speech Simulation: organic micro-jitter, air turbulence, room floor
    np.random.seed(42)
    f0_dynamic = 145.0 + 8.0 * np.sin(2 * np.pi * 2.5 * t) + np.random.normal(0, 1.2, len(t))
    phase = 2 * np.pi * np.cumsum(f0_dynamic) / sr
    human_wave = (
        0.5 * np.sin(phase) +
        0.25 * np.sin(2 * phase) +
        0.12 * np.sin(3 * phase) +
        0.06 * np.sin(4 * phase)
    )
    # Natural ambient mic room floor / breathing noise
    human_wave += np.random.normal(0, 0.02, len(t))
    human_wave_int16 = (np.clip(human_wave, -1.0, 1.0) * 32767).astype(np.int16)

    human_path = target_dir / "test_human_voice.wav"
    wavfile.write(str(human_path), sr, human_wave_int16)

    return synth_path, human_path

def main():
    service = AudioForensicService(upload_dir=Path("uploads"))
    synth_path, human_path = generate_test_audio_files(Path("uploads"))

    print("=== 1. ANALYZING SIMULATED SYNTHETIC SPEECH ===")
    with open(synth_path, "rb") as f:
        synth_res = service.analyze_audio(f.read())
    print("Verdict           :", synth_res["verdict"])
    print("Scores            :", synth_res["scores"])
    print("Acoustic Indicators:", synth_res["indicators"])
    print("Spectrogram URL   :", synth_res["spectrogram"]["spectrogram_url"])
    print("Spectrogram Data  :", synth_res["spectrogram"]["spectrogram_data_url"][:40], "...")

    print("\n=== 2. ANALYZING SIMULATED NATURAL HUMAN SPEECH ===")
    with open(human_path, "rb") as f:
        human_res = service.analyze_audio(f.read())
    print("Verdict           :", human_res["verdict"])
    print("Scores            :", human_res["scores"])
    print("Acoustic Indicators:", human_res["indicators"])
    print("Spectrogram URL   :", human_res["spectrogram"]["spectrogram_url"])

    print("\n=== AUDIO FORENSICS ENGINE TEST COMPLETED ===")

if __name__ == "__main__":
    main()

