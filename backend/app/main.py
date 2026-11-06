import os
import uuid
import shutil
import threading
from pathlib import Path
from io import BytesIO
from typing import Dict, Any

import cv2
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.services.face_detector import detect_faces
from app.services.deepfake_detector import DeepfakeDetectionService
from app.services.metadata_extractor import extract_forensic_metadata
from app.services.explainability import ExplainabilityService
from app.services.video_processor import VideoForensicService
from app.services.audio_detector import AudioForensicService
from app.services.pdf_generator import ForensicPDFGenerator


# --------------------------------------------------
# CONFIGURATION & STORAGE
# --------------------------------------------------

MAX_IMAGE_SIZE = 15 * 1024 * 1024       # 15 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024      # 100 MB
MAX_AUDIO_SIZE = 25 * 1024 * 1024       # 25 MB

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG", "WEBP"]
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
ALLOWED_AUDIO_EXTENSIONS = {".wav"}

# Set environment defaults
os.environ.setdefault("MPLCONFIGDIR", str(UPLOAD_DIR.resolve()))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Thread-safe singletons (avoids redundant memory duplication on 8GB RAM)
detector = None
detector_lock = threading.Lock()

explainability = None
explainability_lock = threading.Lock()

video_service = None
video_service_lock = threading.Lock()

audio_service = None
audio_service_lock = threading.Lock()


def get_detector() -> DeepfakeDetectionService:
    global detector
    if detector is None:
        with detector_lock:
            if detector is None:
                detector = DeepfakeDetectionService()
    return detector


def get_explainability() -> ExplainabilityService:
    global explainability
    if explainability is None:
        with explainability_lock:
            if explainability is None:
                explainability = ExplainabilityService()
    return explainability


def get_video_service() -> VideoForensicService:
    global video_service
    if video_service is None:
        with video_service_lock:
            if video_service is None:
                video_service = VideoForensicService(
                    detector=get_detector(),
                    explainability=get_explainability()
                )
    return video_service


def get_audio_service() -> AudioForensicService:
    global audio_service
    if audio_service is None:
        with audio_service_lock:
            if audio_service is None:
                audio_service = AudioForensicService(upload_dir=UPLOAD_DIR)
    return audio_service


# --------------------------------------------------
# FASTAPI APPLICATION SETUP
# --------------------------------------------------

app = FastAPI(
    title="DFX - Deepfake Forensics AI Platform",
    description="Explainable AI-powered digital media forensics & multi-modal manipulation analysis",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory for static artifact inspection and Grad-CAM overlays
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# --------------------------------------------------
# HEALTH & SYSTEM STATUS
# --------------------------------------------------

@app.get("/")
async def home():
    return {
        "message": "DFX Deepfake Forensics API is online",
        "version": "3.0.0",
        "capabilities": [
            "image_forensics",
            "video_forensics",
            "audio_forensics",
            "gradcam_explainability",
            "exif_integrity"
        ],
        "status": "ready"
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "DFX Multi-Modal Forensic Engine",
        "models": {
            "face_detector": "YuNet ONNX (0.50 threshold, 50% contextual margin)",
            "primary_classifier": "DFX Dual-Engine (Swin-ShiftedWindow + ViT-Base)",
            "explainability": "Grad-CAM Spatial LayerNorm Attribution",
            "video_engine": "Temporal Stride Sequence Aggregator & Discontinuity Profiler",
            "audio_engine": "Multi-Factor Acoustic Resonance & Vocal Jitter Forensic Engine"
        }
    }


# --------------------------------------------------
# IMAGE FORENSIC INGESTION & METADATA ANALYSIS
# --------------------------------------------------

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    content = await file.read(MAX_IMAGE_SIZE + 1)

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum image size is 15 MB."
        )

    try:
        image = Image.open(BytesIO(content))
        image.verify()

        # Reopen after verify
        image = Image.open(BytesIO(content))

        if image.format not in ALLOWED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Only {', '.join(ALLOWED_IMAGE_FORMATS)} images are supported."
            )

        width, height = image.size
        image_format = image.format

        file_id = str(uuid.uuid4())
        filename = f"{file_id}.{image_format.lower()}"
        file_path = UPLOAD_DIR / filename
        file_path.write_bytes(content)

        # Extract deep forensic metadata & EXIF integrity tags
        metadata = extract_forensic_metadata(content)

        return {
            "status": "success",
            "file_id": file_id,
            "filename": file.filename,
            "format": image_format,
            "width": width,
            "height": height,
            "size_bytes": len(content),
            "metadata": metadata,
            "analysis_status": "pending",
            "message": "Image ingested and metadata analyzed successfully."
        }

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file or corrupted headers."
        )


# --------------------------------------------------
# IMAGE FACE DETECTION, DUAL-ENGINE CLASSIFICATION & GRAD-CAM
# --------------------------------------------------

def analyze_faces_and_explain(content: bytes):
    # Step 1: Detect faces with 50% anatomical margin
    result = detect_faces(content, margin=0.50)
    model = get_detector()
    cam_engine = get_explainability()

    analyzed_faces = []

    for face in result["faces"]:
        face_crop = face["face_crop"]

        # Convert BGR to RGB for inference
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(rgb_face)

        # Step 2: Run Dual-Engine Classification with Calibrated Uncertainty
        prediction = model.predict(pil_face)

        # Step 3: Generate Grad-CAM Spatial Heatmap targeting anomalous tokens
        cam_info = cam_engine.generate_heatmap(face_crop, target_class=1, save_to_disk=True)

        analyzed_faces.append({
            "face_id": face["face_id"],
            "x": face["x"],
            "y": face["y"],
            "width": face["width"],
            "height": face["height"],
            "detector_score": face["detector_score"],
            "classification": prediction,
            "explainability": {
                "heatmap_url": f"/uploads/{cam_info['saved_filename']}" if cam_info.get("saved_filename") else None,
                "heatmap_data_url": cam_info["heatmap_data_url"],
                "peak_intensity": cam_info["peak_intensity"],
                "focal_region": cam_info["focal_region"],
                "interpretation": cam_info["interpretation"]
            }
        })

    # Global aggregate assessment across all detected faces
    fake_count = sum(1 for f in analyzed_faces if f["classification"]["label"] == "FAKE")
    uncertain_count = sum(1 for f in analyzed_faces if f["classification"]["uncertainty_score"] > 60.0)

    if fake_count > 0:
        overall_assessment = "MANIPULATION INDICATED"
    elif uncertain_count > 0:
        overall_assessment = "INCONCLUSIVE / ELEVATED UNCERTAINTY"
    elif len(analyzed_faces) > 0:
        overall_assessment = "NO MANIPULATION DETECTED (AUTHENTIC INDICATED)"
    else:
        overall_assessment = "NO FACE DETECTED"

    return {
        "status": "success",
        "message": "Face detection, dual-engine classification, and Grad-CAM explainability completed",
        "image_width": result["image_width"],
        "image_height": result["image_height"],
        "face_count": result["face_count"],
        "overall_assessment": overall_assessment,
        "faces": analyzed_faces,
        "model": "DFX Dual-Engine (Swin-ShiftedWindow + ViT-Base)",
        "warning": (
            "Model predictions are indicative forensic scores, "
            "not definitive proof of image manipulation."
        )
    }


@app.post("/api/analyze/faces")
async def analyze_faces(file: UploadFile = File(...)):
    content = await file.read(MAX_IMAGE_SIZE + 1)

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum image size is 15 MB."
        )

    try:
        return await run_in_threadpool(
            analyze_faces_and_explain,
            content
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Face analysis & explainability error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="Face analysis or deepfake classification failed."
        )


# --------------------------------------------------
# FORENSIC DOSSIER GENERATION (IMAGE)
# --------------------------------------------------

@app.post("/api/report/generate")
async def generate_forensic_report(file: UploadFile = File(...)):
    content = await file.read(MAX_IMAGE_SIZE + 1)
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 15 MB)")

    meta = extract_forensic_metadata(content)
    analysis = await run_in_threadpool(analyze_faces_and_explain, content)

    report_id = f"DFX-IMG-{uuid.uuid4().hex[:8].upper()}"

    dossier = {
        "report_id": report_id,
        "media_type": "IMAGE",
        "filename": file.filename,
        "verdict": analysis["overall_assessment"],
        "integrity_summary": {
            "tampering_risk": meta.get("exif_tampering_risk", "UNKNOWN"),
            "editing_software_detected": meta.get("editing_detected", False),
            "software": meta.get("software"),
            "camera_hardware": f"{meta.get('camera_make', '')} {meta.get('camera_model', '')}".strip() or "None",
            "estimated_jpeg_quality": meta.get("estimated_jpeg_quality"),
            "compression_ratio": meta.get("compression_ratio")
        },
        "facial_findings": [
            {
                "face_id": f["face_id"],
                "box": {"x": f["x"], "y": f["y"], "w": f["width"], "h": f["height"]},
                "label": f["classification"]["label"],
                "confidence": f["classification"]["confidence"],
                "real_score": f["classification"]["real_score"],
                "fake_score": f["classification"]["fake_score"],
                "uncertainty_score": f["classification"]["uncertainty_score"],
                "assessment": f["classification"]["assessment"],
                "gradcam_focal_region": f["explainability"]["focal_region"],
                "gradcam_peak_intensity": f["explainability"]["peak_intensity"],
                "heatmap_url": f["explainability"]["heatmap_url"]
            }
            for f in analysis["faces"]
        ],
        "forensic_disclaimer": (
            "This report is generated by DFX Deepfake Forensics AI. "
            "Findings represent algorithmic probabilities and feature attribution maps, "
            "not judicial or legal certification of authenticity."
        )
    }

    return {
        "status": "success",
        "report_id": report_id,
        "dossier": dossier
    }


# --------------------------------------------------
# VIDEO FORENSICS ENDPOINTS (PHASE 3)
# --------------------------------------------------

@app.post("/api/analyze/video")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    sample_fps: float = Form(1.0),
    max_frames: int = Form(30)
):
    file_ext = Path(file.filename or "video.mp4").suffix.lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{file_ext}'. Allowed formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        )

    video_id = str(uuid.uuid4())
    temp_video_filename = f"{video_id}{file_ext}"
    temp_video_path = UPLOAD_DIR / temp_video_filename

    # Stream file to disk to preserve memory
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        if temp_video_path.exists():
            temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to write uploaded video: {str(e)}")

    file_size = temp_video_path.stat().st_size
    if file_size > MAX_VIDEO_SIZE:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"Video file exceeds maximum size limit of 100 MB.")

    try:
        video_engine = get_video_service()
        analysis_result = await run_in_threadpool(
            video_engine.analyze_video,
            str(temp_video_path),
            UPLOAD_DIR,
            sample_fps,
            max_frames
        )

        analysis_result["video_id"] = video_id
        analysis_result["filename"] = file.filename
        analysis_result["video_url"] = f"/uploads/{temp_video_filename}"

        return analysis_result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("Video analysis error:", repr(e))
        raise HTTPException(status_code=500, detail="Video forensic analysis pipeline encountered a processing error.")


@app.post("/api/report/video")
async def generate_video_report_endpoint(
    file: UploadFile = File(...),
    sample_fps: float = Form(1.0),
    max_frames: int = Form(30)
):
    analysis = await analyze_video_endpoint(file, sample_fps, max_frames)
    report_id = f"DFX-VID-{uuid.uuid4().hex[:8].upper()}"

    dossier = {
        "report_id": report_id,
        "media_type": "VIDEO",
        "filename": analysis.get("filename"),
        "verdict": analysis.get("verdict"),
        "video_metadata": analysis.get("video_metadata"),
        "temporal_summary": analysis.get("temporal_summary"),
        "suspicious_frames": analysis.get("suspicious_frames"),
        "timeline_sample_count": len(analysis.get("timeline", [])),
        "forensic_disclaimer": analysis.get("forensic_disclaimer")
    }

    return {
        "status": "success",
        "report_id": report_id,
        "dossier": dossier
    }


# --------------------------------------------------
# AUDIO FORENSICS ENDPOINTS (PHASE 4)
# --------------------------------------------------

@app.post("/api/analyze/audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    """
    Ingests audio files, extracts vocal tract perturbation features (jitter, shimmer),
    detects neural vocoder brickwall cutoffs, and visualizes STFT spectrograms.
    """
    file_ext = Path(file.filename or "audio.wav").suffix.lower()
    if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{file_ext}'. Uncompressed WAV (.wav) audio is supported."
        )

    content = await file.read(MAX_AUDIO_SIZE + 1)
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="Audio file exceeds maximum size limit of 25 MB.")

    try:
        service = get_audio_service()
        analysis = await run_in_threadpool(service.analyze_audio, content)
        analysis["filename"] = file.filename
        return analysis

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("Audio forensic analysis error:", repr(e))
        raise HTTPException(status_code=500, detail="Audio forensic analysis pipeline encountered an error.")


@app.post("/api/report/audio")
async def generate_audio_report_endpoint(file: UploadFile = File(...)):
    """
    Generates a consolidated audio forensic report dossier.
    """
    analysis = await analyze_audio_endpoint(file)
    report_id = f"DFX-AUD-{uuid.uuid4().hex[:8].upper()}"

    dossier = {
        "report_id": report_id,
        "media_type": "AUDIO",
        "filename": analysis.get("filename"),
        "verdict": analysis.get("verdict"),
        "scores": analysis.get("scores"),
        "explanation": analysis.get("explanation"),
        "audio_metadata": analysis.get("audio_metadata"),
        "indicators": analysis.get("indicators"),
        "spectrogram_url": analysis.get("spectrogram", {}).get("spectrogram_url"),
        "forensic_disclaimer": analysis.get("forensic_disclaimer")
    }

    return {
        "status": "success",
        "report_id": report_id,
        "dossier": dossier
    }


# --------------------------------------------------
# PDF FORENSIC REPORT GENERATION ENDPOINT (PHASE 5)
# --------------------------------------------------

@app.post("/api/report/export-pdf")
async def export_pdf_report_endpoint(dossier: Dict[str, Any] = Body(...)):
    """
    Renders an official multi-page PDF forensic laboratory report
    complete with case metadata, verdict banner, quantitative metrics,
    and formal legal/scientific disclaimers.
    """
    report_id = dossier.get("report_id", f"DFX-{uuid.uuid4().hex[:8].upper()}")
    pdf_filename = f"{report_id}.pdf"
    pdf_path = UPLOAD_DIR / pdf_filename

    try:
        generator = ForensicPDFGenerator()
        await run_in_threadpool(generator.generate_pdf, dossier, pdf_path)

        return FileResponse(
            path=str(pdf_path),
            filename=pdf_filename,
            media_type="application/pdf"
        )
    except Exception as e:
        print("PDF report generation error:", repr(e))
        raise HTTPException(status_code=500, detail="Failed to compile forensic PDF document.")

