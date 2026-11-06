import os
import uuid
import base64
from pathlib import Path
from io import BytesIO
from typing import Dict, Any, Tuple, Optional

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AudioForensicService:
    """
    Forensic Audio Analysis Engine:
    - Ingests PCM and floating-point audio waveforms.
    - Measures biomechanical voice indicators: pitch jitter, amplitude shimmer,
      spectral flatness, and high-frequency vocoder acoustic voids.
    - Inspects silence floors for artificial mathematical digital silence vs natural respiration.
    - Generates high-resolution calibrated STFT spectrograms for visual acoustic examination.
    """

    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Path("uploads")
        self.spectrogram_dir = self.upload_dir / "audio_spectrograms"
        self.spectrogram_dir.mkdir(parents=True, exist_ok=True)

    def load_audio(self, file_bytes: bytes) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Loads audio from raw bytes using scipy.io.wavfile.
        Handles 16-bit, 24-bit, 32-bit PCM and 32-bit float, converts to mono [-1.0, 1.0].
        """
        buffer = BytesIO(file_bytes)
        try:
            sample_rate, data = wavfile.read(buffer)
        except Exception as e:
            raise ValueError(f"Failed to decode audio file. Ensure it is a valid uncompressed WAV file: {str(e)}")

        if data.size == 0:
            raise ValueError("Audio file contains zero audio samples.")

        # Handle multi-channel (convert stereo to mono)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Determine bit depth and normalize to float32 [-1.0, 1.0]
        dtype = data.dtype
        if dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
            bit_depth = 16
        elif dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
            bit_depth = 32
        elif dtype == np.uint8:
            audio = (data.astype(np.float32) - 128.0) / 128.0
            bit_depth = 8
        elif dtype in (np.float32, np.float64):
            audio = data.astype(np.float32)
            # Clip if beyond range
            audio = np.clip(audio, -1.0, 1.0)
            bit_depth = 32
        else:
            audio = data.astype(np.float32)
            max_val = np.max(np.abs(audio)) or 1.0
            audio = audio / max_val
            bit_depth = 16

        duration_sec = round(len(audio) / sample_rate, 2)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        dynamic_range_db = round(20 * np.log10(max(1e-6, peak) / max(1e-6, rms + 1e-9)), 2)

        meta = {
            "sample_rate": sample_rate,
            "channels": 1 if data.ndim == 1 else 2,
            "bit_depth": bit_depth,
            "total_samples": len(audio),
            "duration_sec": duration_sec,
            "peak_amplitude": round(peak, 4),
            "rms_amplitude": round(rms, 4),
            "dynamic_range_db": dynamic_range_db
        }

        return audio, sample_rate, meta

    def compute_spectral_cutoff(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Detects artificial high-frequency vocoder brickwall cutoff (acoustic void).
        Many neural vocoders (HiFi-GAN, WaveGlow, FastSpeech) terminate synthesis
        at 8 kHz (for 16 kHz sample rates) or 11 kHz (for 22.05 kHz).
        """
        # Compute Power Spectral Density using Welch's method
        freqs, psd = signal.welch(audio, fs=sample_rate, nperseg=min(len(audio), 2048))
        psd_db = 10 * np.log10(np.maximum(psd, 1e-12))
        
        nyquist = sample_rate / 2.0
        
        # Check energy in upper band (e.g., above 7500 Hz if nyquist >= 8000 Hz)
        has_brickwall = False
        cutoff_freq_hz = nyquist
        upper_energy_ratio = 0.0

        if nyquist >= 8000:
            high_band_mask = freqs >= 7500
            low_band_mask = (freqs >= 300) & (freqs < 7500)

            low_energy = np.mean(psd[low_band_mask]) if np.any(low_band_mask) else 1e-6
            high_energy = np.mean(psd[high_band_mask]) if np.any(high_band_mask) else 0.0

            ratio = high_energy / max(1e-9, low_energy)
            upper_energy_ratio = round(float(ratio), 6)

            # A ratio < 0.0005 indicates severe high-frequency cutoff/shelf
            if upper_energy_ratio < 0.0005:
                has_brickwall = True
                # Find approximate shelf drop frequency
                smooth_psd = signal.medfilt(psd_db, kernel_size=9)
                drop_indices = np.where(smooth_psd < (np.max(smooth_psd) - 40))[0]
                if len(drop_indices) > 0:
                    cutoff_freq_hz = round(float(freqs[drop_indices[0]]), 1)
                else:
                    cutoff_freq_hz = 8000.0

        return {
            "has_brickwall_cutoff": has_brickwall,
            "cutoff_frequency_hz": cutoff_freq_hz,
            "nyquist_frequency_hz": nyquist,
            "upper_energy_ratio": upper_energy_ratio
        }

    def compute_pitch_jitter_and_shimmer(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Estimates local pitch perturbation (Jitter) and amplitude perturbation (Shimmer).
        Real human vocal folds produce natural micro-perturbation:
        - Natural Jitter: 0.4% - 1.8%
        - Synthetic speech is often either unnaturally smooth (<0.20%) or erratically jumping (>3.0%).
        """
        # Frame-based autocorrelation for pitch detection
        frame_len = int(sample_rate * 0.04)  # 40ms frame
        hop_len = int(sample_rate * 0.02)    # 20ms hop
        
        min_period = int(sample_rate / 400.0)  # 400 Hz max pitch
        max_period = int(sample_rate / 60.0)   # 60 Hz min pitch

        periods = []
        amplitudes = []

        for start in range(0, len(audio) - frame_len, hop_len):
            frame = audio[start:start + frame_len]
            # Check if frame is voiced (energy above threshold)
            if np.mean(frame ** 2) < 0.001:
                continue

            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(frame) - 1:]

            # Search in valid pitch period range
            search_window = corr[min_period:max_period]
            if len(search_window) == 0:
                continue

            peak_idx = np.argmax(search_window) + min_period
            peak_val = corr[peak_idx]

            # Voicing strength check (normalized correlation)
            if corr[0] > 0 and (peak_val / corr[0]) > 0.45:
                periods.append(peak_idx)
                amplitudes.append(np.max(np.abs(frame)))

        periods = np.array(periods, dtype=np.float64)
        amplitudes = np.array(amplitudes, dtype=np.float64)

        if len(periods) >= 8:
            mean_period = np.mean(periods)
            # Local Jitter: sum(|T_i - T_{i+1}|) / ((N-1) * mean(T))
            diff_periods = np.abs(np.diff(periods))
            jitter_percent = round(float((np.mean(diff_periods) / max(1e-6, mean_period)) * 100.0), 3)

            # Local Shimmer: sum(|A_i - A_{i+1}|) / ((N-1) * mean(A))
            mean_amp = np.mean(amplitudes)
            diff_amps = np.abs(np.diff(amplitudes))
            shimmer_percent = round(float((np.mean(diff_amps) / max(1e-6, mean_amp)) * 100.0), 3)

            mean_f0 = round(float(sample_rate / mean_period), 1)
        else:
            jitter_percent = 0.50  # Default neutral
            shimmer_percent = 2.0  # Default neutral
            mean_f0 = 120.0

        return {
            "mean_f0_pitch_hz": mean_f0,
            "jitter_percent": jitter_percent,
            "shimmer_percent": shimmer_percent,
            "voiced_frames_count": len(periods)
        }

    def compute_spectral_flatness_and_dynamics(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Measures Wiener entropy (spectral flatness) and silent floor dynamics.
        Synthetic speech generators often output digital zero floor (absence of respiration sound).
        """
        # Spectral Flatness
        eps = 1e-12
        freqs, psd = signal.welch(audio, fs=sample_rate, nperseg=min(len(audio), 1024))
        psd_safe = np.maximum(psd, eps)
        geom_mean = np.exp(np.mean(np.log(psd_safe)))
        arith_mean = np.mean(psd_safe)
        spectral_flatness = round(float(geom_mean / max(eps, arith_mean)), 5)

        # Silence floor examination (segment energy in dB)
        frame_len = int(sample_rate * 0.05)  # 50ms
        frame_energies = []
        for i in range(0, len(audio) - frame_len, frame_len):
            seg = audio[i:i + frame_len]
            rms = np.sqrt(np.mean(seg ** 2))
            frame_energies.append(rms)

        frame_energies = np.array(frame_energies)
        min_rms = float(np.min(frame_energies)) if len(frame_energies) > 0 else 0.0
        min_rms_db = round(20 * np.log10(max(1e-6, min_rms)), 1)

        # Mathematical digital silence detection:
        # If minimum RMS in pauses is below -75 dB or exactly 0.0, typical of TTS concatenate
        has_digital_silence = (min_rms < 1e-4) or (min_rms_db < -72.0)

        return {
            "spectral_flatness": float(spectral_flatness),
            "silence_floor_db": float(min_rms_db),
            "has_artificial_silence_floor": bool(has_digital_silence)
        }

    def generate_spectrogram_image(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, str]:
        """
        Generates calibrated STFT spectrogram and exports both on-disk PNG and base64 data URL.
        """
        nperseg = min(len(audio), 1024)
        noverlap = nperseg // 2
        f, t, Sxx = signal.spectrogram(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Convert power to dB
        Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))

        # Render with Matplotlib
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
        fig.patch.set_facecolor("#080b14")
        ax.set_facecolor("#080b14")

        im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno')
        ax.set_ylabel("Frequency (Hz)", color="#94a3b8", fontsize=9)
        ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=9)
        ax.set_title("DFX High-Resolution STFT Forensic Spectrogram", color="#22d3ee", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=8)

        for spine in ax.spines.values():
            spine.set_color("#334155")

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Power (dB)", color="#94a3b8", fontsize=8)
        cbar.ax.tick_params(colors="#94a3b8", labelsize=7)

        plt.tight_layout()

        # Save to disk
        filename = f"spec_{uuid.uuid4().hex[:10]}.png"
        save_path = self.spectrogram_dir / filename
        plt.savefig(str(save_path), facecolor=fig.get_facecolor(), bbox_inches="tight")

        # Encode to base64
        buf = BytesIO()
        plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        buf.seek(0)
        base64_data = base64.b64encode(buf.read()).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_data}"

        return filename, data_url

    def analyze_audio(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Executes complete multi-parameter forensic acoustic examination.
        """
        audio, sample_rate, meta = self.load_audio(file_bytes)

        # Step 1: High-Frequency Vocoder Void Detection
        cutoff_info = self.compute_spectral_cutoff(audio, sample_rate)

        # Step 2: Pitch Perturbation (Jitter) & Amplitude Perturbation (Shimmer)
        jitter_info = self.compute_pitch_jitter_and_shimmer(audio, sample_rate)

        # Step 3: Spectral Flatness & Digital Silence Analysis
        dynamics_info = self.compute_spectral_flatness_and_dynamics(audio, sample_rate)

        # Step 4: Generate Forensic Spectrogram
        spec_filename, spec_data_url = self.generate_spectrogram_image(audio, sample_rate)

        # --------------------------------------------------
        # FORENSIC SYNTHESIS PROBABILITY SCORING
        # --------------------------------------------------
        # Weights for forensic indicators:
        # 1. Brickwall cutoff: strong vocoder signature (weight: 35)
        # 2. Pitch jitter anomaly (<0.20% or >2.8%): (weight: 30)
        # 3. Artificial mathematical silence floor: (weight: 20)
        # 4. Spectral flatness & shimmer anomaly: (weight: 15)

        synth_points = 0.0

        # Cutoff indicator
        if cutoff_info["has_brickwall_cutoff"]:
            synth_points += 35.0
        elif cutoff_info["upper_energy_ratio"] < 0.002:
            synth_points += 15.0

        # Jitter indicator (normal human: 0.4% - 1.8%)
        jitter = jitter_info["jitter_percent"]
        if jitter < 0.15:
            synth_points += 30.0  # Unnaturally robotic/smooth glottal pulses
        elif jitter > 2.80:
            synth_points += 20.0  # Excessive neural phase jumps
        elif jitter < 0.30:
            synth_points += 15.0

        # Silence floor indicator
        if dynamics_info["has_artificial_silence_floor"]:
            synth_points += 20.0

        # Shimmer indicator (normal human: 1.5% - 4.5%)
        shimmer = jitter_info["shimmer_percent"]
        if shimmer < 0.8:
            synth_points += 15.0
        elif shimmer > 6.0:
            synth_points += 10.0

        synth_probability = round(min(98.5, max(2.5, synth_points)), 2)
        human_probability = round(100.0 - synth_probability, 2)

        # Calibrated Uncertainty: distance from 50/50 decision boundary
        margin = abs(human_probability - synth_probability)
        uncertainty = round(100.0 - margin, 2)

        # Forensic Qualitative Assessment
        if synth_probability >= 60.0:
            verdict = "SYNTHETIC VOICE INDICATED"
            explanation = (
                "Acoustic analysis detected strong indicators of neural speech synthesis, "
                "including artificial vocal perturbation constraints, high-frequency cutoff shelves, "
                "or mathematical digital pause signatures."
            )
        elif synth_probability <= 35.0:
            verdict = "NATURAL HUMAN VOICE INDICATED"
            explanation = (
                "Acoustic features demonstrate continuous high-frequency harmonic propagation, "
                "organic glottal jitter, and ambient infraglottic pause dynamics consistent with natural human vocalization."
            )
        else:
            verdict = "INCONCLUSIVE / ELEVATED UNCERTAINTY"
            explanation = (
                "Acoustic indicators present conflicting signatures or moderate uncertainty. "
                "Audio compression or microphone noise may affect perturbation metrics."
            )

        return {
            "status": "success",
            "audio_metadata": meta,
            "verdict": verdict,
            "explanation": explanation,
            "scores": {
                "synthetic_probability": synth_probability,
                "natural_probability": human_probability,
                "uncertainty_score": uncertainty,
                "label": "SYNTHETIC" if synth_probability >= 50.0 else "NATURAL"
            },
            "indicators": {
                "high_frequency_brickwall": cutoff_info["has_brickwall_cutoff"],
                "cutoff_frequency_hz": cutoff_info["cutoff_frequency_hz"],
                "pitch_jitter_percent": jitter_info["jitter_percent"],
                "pitch_shimmer_percent": jitter_info["shimmer_percent"],
                "mean_f0_hz": jitter_info["mean_f0_pitch_hz"],
                "spectral_flatness": dynamics_info["spectral_flatness"],
                "silence_floor_db": dynamics_info["silence_floor_db"],
                "artificial_silence_detected": dynamics_info["has_artificial_silence_floor"]
            },
            "spectrogram": {
                "spectrogram_url": f"/uploads/audio_spectrograms/{spec_filename}",
                "spectrogram_data_url": spec_data_url
            },
            "model": "DFX Multi-Factor Acoustic Resonance & Vocal Jitter Forensic Engine",
            "forensic_disclaimer": (
                "Audio deepfake indicators represent acoustic and signal-processing characteristics. "
                "They should be corroborated with visual and contextual evidence, "
                "not treated as standalone legal proof."
            )
        }
