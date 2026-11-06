# DFX — Technical Architecture & Forensic Pipeline Specification

This document provides a comprehensive technical overview of the **Deepfake Forensics AI (DFX)** architecture, data flow, mathematical model ensembles, and explainability workflows.

---

## 1. System Architecture Diagram

```mermaid
flowchart TD
    subgraph ClientLayer["Frontend Presentation Layer (Next.js 16 + Tailwind CSS)"]
        UI["Multi-Modal Web Dashboard"]
        TabImg["🖼️ Still Image Forensics"]
        TabVid["🎥 Video Temporal Forensics"]
        TabAud["🎙️ Audio Acoustic Forensics"]
        UI --> TabImg
        UI --> TabVid
        UI --> TabAud
    end

    subgraph APILayer["Backend Gateway (FastAPI + Starlette Threadpool)"]
        Router["FastAPI Application Router"]
        StaticSrv["Static Artifact Server (/uploads)"]
        PDFGen["ReportLab PDF Dossier Compiler"]
    end

    subgraph ForensicCore["Forensic Engine Services"]
        subgraph ImgEngine["Image Forensics Pipeline"]
            EXIF["EXIF & Hardware Metadata Extractor"]
            ELA["Error Level Analysis (ELA) Engine"]
            YuNet["YuNet ONNX Face Detector (50% Contextual Margin)"]
            DualVision["DFX Dual-Engine Ensemble"]
            Swin["Swin Transformer (Shifted Window, 60%)"]
            ViT["Vision Transformer (ViT-Base, 40%)"]
            GradCAM["Grad-CAM Attention Attribution (LayerNorm)"]
            
            YuNet --> DualVision
            DualVision --> Swin
            DualVision --> ViT
            DualVision --> GradCAM
        end

        subgraph VidEngine["Video Temporal Forensics Pipeline"]
            Decoder["OpenCV Video Stream Decoder"]
            Sampler["Uniform Stride Temporal Frame Sampler"]
            Tracker["Per-Frame Facial Tracking"]
            Discontinuity["Temporal Variance & Jitter Profiler"]
            SuspiciousRank["Top Suspicious Frames Selector"]

            Decoder --> Sampler --> Tracker --> Discontinuity --> SuspiciousRank
        end

        subgraph AudEngine["Audio Acoustic Forensics Pipeline"]
            WavReader["PCM Waveform Ingestion (scipy.io.wavfile)"]
            Jitter["Glottal Pitch Jitter (0.4%-1.8% baseline)"]
            Shimmer["Cycle-to-Cycle Amplitude Shimmer"]
            VoidDetect["High-Frequency Brickwall Cutoff Detector"]
            SilenceDyn["Respiration Silence Floor Profiler"]
            STFT["Calibrated STFT Time-Frequency Spectrogram"]

            WavReader --> Jitter
            WavReader --> Shimmer
            WavReader --> VoidDetect
            WavReader --> SilenceDyn
            WavReader --> STFT
        end
    end

    subgraph ArtifactStore["Storage & Artifacts Layer"]
        Uploads["uploads/"]
        Heatmaps["uploads/gradcam_*.jpg"]
        VidFrames["uploads/video_frames/"]
        Spectrograms["uploads/audio_spectrograms/"]
        PDFs["uploads/*.pdf"]
    end

    TabImg -->|POST /api/analyze/image| Router
    TabImg -->|POST /api/analyze/faces| Router
    TabVid -->|POST /api/analyze/video| Router
    TabAud -->|POST /api/analyze/audio| Router
    UI -->|POST /api/report/export-pdf| Router

    Router --> EXIF & ELA & YuNet
    Router --> Decoder
    Router --> WavReader
    Router --> PDFGen

    GradCAM --> Heatmaps
    SuspiciousRank --> VidFrames
    STFT --> Spectrograms
    PDFGen --> PDFs
```

---

## 2. End-to-End Analysis Lifecycle (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    actor User as Forensic Examiner
    participant FE as Next.js Dashboard
    participant API as FastAPI Gateway
    participant CV as Vision / YuNet Engine
    participant ML as Dual-Engine (Swin + ViT)
    participant CAM as Grad-CAM Explainer
    participant PDF as ReportLab Generator

    User->>FE: Upload Still Image / Video / Audio File
    FE->>API: POST /api/analyze/... (Multipart Payload)
    
    rect rgb(20, 25, 40)
        Note over API,CV: Spatial & Geometric Processing
        API->>CV: Detect Face Bounding Boxes (YuNet ONNX)
        CV-->>API: Extracted Faces with 50% Anatomical Margin
    end

    rect rgb(30, 25, 45)
        Note over API,ML: Dual-Engine Ensemble Inference
        API->>ML: Inference with Swin-Transformer (60%)
        API->>ML: Inference with ViT-Base (40%)
        ML-->>API: Combined Fake Score & Distance-to-Boundary Uncertainty
    end

    rect rgb(25, 35, 45)
        Note over API,CAM: Spatial Attribution & Explainability
        API->>CAM: Compute Backward Activation on model.vit.layers[-1]
        CAM-->>API: Normalized Heatmap Overlay & Focal Region Tag
    end

    API-->>FE: Return Structured JSON (Verdict, Scores, Heatmap Data URL)
    FE->>User: Display Bounding Boxes, Metrics, and Heatmaps

    opt Export PDF Report
        User->>FE: Click "Download PDF Dossier"
        FE->>API: POST /api/report/export-pdf (Dossier JSON)
        API->>PDF: Render ISO/IEC 27037 Compliant Vector Document
        PDF-->>API: Generated Binary PDF File
        API-->>FE: Stream Application/PDF Attachment
        FE-->>User: Trigger Browser File Download
    end
```

---

## 3. Dual-Engine Calibration & Decision Flowchart

DFX rejects uncalibrated binary thresholding in favor of a consensus ensemble with probabilistic uncertainty estimation:

```mermaid
flowchart TD
    Input["Input Contextual Face Crop (224x224 RGB)"] --> SwinModel["Swin Transformer (Hierarchical Shifted-Window)"]
    Input --> ViTModel["Vision Transformer (ViT-Base Patch Embeddings)"]

    SwinModel -->|P_swin_fake| Ensemble["Weighted Linear Consensus Ensemble"]
    ViTModel -->|P_vit_fake| Ensemble

    Ensemble --> CalcCombined["P_fake = 0.60 * P_swin + 0.40 * P_vit<br>P_real = 100.0 - P_fake"]
    
    CalcCombined --> CalcUncertainty["Uncertainty Metric = 100.0 - | P_real - P_fake |"]
    
    CalcCombined --> DecisionCheck{"Decision Thresholding"}

    DecisionCheck -->|P_fake >= 65%| Manip["Verdict: MANIPULATION INDICATED<br>Color: RED"]
    DecisionCheck -->|P_fake <= 35%| Auth["Verdict: AUTHENTIC INDICATED<br>Color: GREEN"]
    DecisionCheck -->|35% < P_fake < 65% or Uncertainty > 60%| Inconcl["Verdict: INCONCLUSIVE / ELEVATED UNCERTAINTY<br>Color: AMBER"]

    Manip --> OutputDossier["Output Forensic Assessment & Grad-CAM Focal Region"]
    Auth --> OutputDossier
    Inconcl --> OutputDossier
```

---

## 4. Audio Acoustic Resonance & Speech Biometrics Pipeline

```mermaid
flowchart LR
    WavInput["Raw Audio Waveform (WAV PCM)"] --> MonoNorm["Channel Mixdown & Float32 [-1, 1] Normalization"]

    MonoNorm --> WelchsMethod["Welch's Power Spectral Density"]
    MonoNorm --> PitchTrack["Frame-Based Autocorrelation F0 Tracking"]
    MonoNorm --> EnergyVAD["Short-Time Energy Voice Activity Detection"]
    MonoNorm --> STFTGen["Short-Time Fourier Transform (STFT)"]

    WelchsMethod --> BrickwallCheck{"High-Freq Drop > 30 dB before Nyquist?"}
    BrickwallCheck -->|Yes| BrickwallFlag["Flag: High-Frequency Vocoder Void (Shelf at 8kHz / 11kHz)"]
    BrickwallCheck -->|No| NaturalSpectrum["Continuous Harmonic Propagation"]

    PitchTrack --> JitterCalc["Local Jitter: Sum(|Ti - Ti+1|) / ((N-1) * T_mean) * 100%"]
    JitterCalc --> JitterEval{"Jitter in Human Range?<br>(0.4% - 1.8%)"}
    JitterEval -->|Jitter < 0.20%| RoboticSynth["Flag: Unnaturally Rigid Glottal Pulses"]
    JitterEval -->|Jitter > 2.80%| PhaseJumps["Flag: Neural Phase Discontinuities"]
    JitterEval -->|0.4% <= Jitter <= 1.8%| HumanJitter["Organic Vocal Chord Tremor"]

    EnergyVAD --> SilenceFloor{"Pause RMS < -75 dB or Exact Zeros?"}
    SilenceFloor -->|Yes| DigitalZero["Flag: Artificial Digital Silence (No Respiration)"]
    SilenceFloor -->|No| AmbientRoom["Natural Infraglottic Breath & Room Tone"]

    STFTGen --> SpecPlot["Generate Matplotlib Inferno Spectrogram Image"]
```

---

## 5. Technical Stack Matrix

| Subsystem | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend Framework** | Next.js 16 (React 19, TypeScript) | Client-side reactive interface with dark-mode forensic theme |
| **Styling** | Tailwind CSS v4 | High-density information dashboards and responsive controls |
| **Backend REST API** | FastAPI + Uvicorn | Asynchronous endpoint routing and threadpool dispatching |
| **Face Detection** | YuNet ONNX (OpenCV DNN) | Ultra-fast CPU facial landmark and bounding box tracking |
| **Primary Vision Model** | Swin Transformer (`umm-maybe/AI-image-detector`) | Hierarchical shifted-window self-attention for studio specificity |
| **Secondary Vision Model** | Vision Transformer (`dima806/deepfake_vs_real_image_detection`) | Global attention for patch-level GAN synthesis artifacts |
| **Visual Explainability** | Grad-CAM (`pytorch-grad-cam`) | LayerNorm spatial attribution maps |
| **Video Decoding** | OpenCV (`cv2.VideoCapture`) | Frame extraction with uniform temporal stride sampling |
| **Audio Digital Signal Processing**| SciPy (`scipy.signal`, `scipy.io.wavfile`) | Glottal pitch autocorrelation, spectral flatness, and STFT |
| **Dossier Compilation** | ReportLab 5.1 | Formal vector PDF forensic examination certificates |

