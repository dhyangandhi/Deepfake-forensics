import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image

from app.services.face_detector import detect_faces_from_frame
from app.services.deepfake_detector import DeepfakeDetectionService
from app.services.explainability import ExplainabilityService


class VideoForensicService:
    """
    Forensic Video Analysis Engine:
    - Extracts frame-level sequences at uniform temporal intervals.
    - Tracks facial entities across time using YuNet.
    - Evaluates facial manipulation across frames via DFX Dual-Engine (Swin + ViT).
    - Measures inter-frame temporal consistency and jitter variance.
    - Generates suspicious-frame heatmaps and chronological timeline metrics.
    """

    def __init__(self, detector: Optional[DeepfakeDetectionService] = None, explainability: Optional[ExplainabilityService] = None):
        self.detector = detector
        self.explainability = explainability

    def get_detector(self) -> DeepfakeDetectionService:
        if self.detector is None:
            self.detector = DeepfakeDetectionService()
        return self.detector

    def get_explainability(self) -> ExplainabilityService:
        if self.explainability is None:
            self.explainability = ExplainabilityService()
        return self.explainability

    def extract_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Extracts stream geometry, FPS, codec, duration, and frame count.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video stream: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or np.isnan(fps):
                fps = 25.0  # Fallback standard FPS
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            duration_sec = round(total_frames / fps, 2) if total_frames > 0 else 0.0

            file_size_bytes = os.path.getsize(video_path)

            return {
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_sec": duration_sec,
                "fourcc": fourcc.strip(),
                "file_size_bytes": file_size_bytes
            }
        finally:
            cap.release()

    def analyze_video(
        self,
        video_path: str,
        output_dir: Path,
        sample_fps: float = 1.0,
        max_frames: int = 30
    ) -> Dict[str, Any]:
        """
        Executes frame-level deepfake forensics across a video stream.
        """
        meta = self.extract_metadata(video_path)
        fps = meta["fps"]
        total_frames = meta["total_frames"]

        if total_frames <= 0:
            raise ValueError("Video contains no valid readable frames.")

        # Calculate uniform sampling stride
        target_step = max(1, int(round(fps / max(0.1, sample_fps))))
        estimated_samples = total_frames // target_step

        if estimated_samples > max_frames:
            # Distribute max_frames uniformly across the entire video timeline
            step = max(1, total_frames // max_frames)
        else:
            step = target_step

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file for frame extraction.")

        frames_dir = output_dir / "video_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        timeline: List[Dict[str, Any]] = []
        frames_with_faces: List[Dict[str, Any]] = []
        
        detector_engine = self.get_detector()
        cam_engine = self.get_explainability()

        current_frame_idx = 0
        analyzed_sample_count = 0

        try:
            while cap.isOpened() and analyzed_sample_count < max_frames:
                # Seek to target frame index
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()

                if not ret or frame is None:
                    break

                timestamp_sec = round(current_frame_idx / fps, 2)

                # Execute face detection on the current frame
                detection = detect_faces_from_frame(frame, margin=0.50)
                faces = detection["faces"]

                frame_faces_data = []
                frame_fake_score = 0.0
                frame_uncertainty = 0.0
                frame_label = "NO_FACE"
                top_face_crop = None
                top_face_box = None

                if len(faces) > 0:
                    max_fake = -1.0
                    for face in faces:
                        face_crop = face["face_crop"]
                        # Convert BGR to RGB for deepfake model inference
                        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(rgb_crop)

                        # Dual-engine prediction
                        pred = detector_engine.predict(pil_crop)

                        face_entry = {
                            "face_id": face["face_id"],
                            "x": face["x"],
                            "y": face["y"],
                            "width": face["width"],
                            "height": face["height"],
                            "confidence": pred["confidence"],
                            "label": pred["label"],
                            "fake_score": pred["fake_score"],
                            "real_score": pred["real_score"],
                            "uncertainty_score": pred["uncertainty_score"],
                            "assessment": pred["assessment"]
                        }
                        frame_faces_data.append(face_entry)

                        if pred["fake_score"] > max_fake:
                            max_fake = pred["fake_score"]
                            frame_fake_score = pred["fake_score"]
                            frame_uncertainty = pred["uncertainty_score"]
                            frame_label = pred["label"]
                            top_face_crop = face_crop
                            top_face_box = (face["x"], face["y"], face["width"], face["height"])

                timeline_entry = {
                    "frame_index": current_frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "face_count": len(faces),
                    "fake_score": round(frame_fake_score, 2),
                    "label": frame_label,
                    "uncertainty_score": round(frame_uncertainty, 2),
                    "faces": frame_faces_data
                }

                timeline.append(timeline_entry)

                if len(faces) > 0:
                    frames_with_faces.append({
                        **timeline_entry,
                        "raw_frame": frame,
                        "top_face_crop": top_face_crop,
                        "top_face_box": top_face_box
                    })

                analyzed_sample_count += 1
                current_frame_idx += step

                if current_frame_idx >= total_frames:
                    break

        finally:
            cap.release()

        # --------------------------------------------------
        # TEMPORAL AGGREGATION & METRICS
        # --------------------------------------------------
        total_sampled = len(timeline)
        face_frames_count = len(frames_with_faces)

        if face_frames_count > 0:
            fake_scores = [f["fake_score"] for f in frames_with_faces]
            mean_fake_score = round(float(np.mean(fake_scores)), 2)
            peak_fake_score = round(float(np.max(fake_scores)), 2)
            temporal_variance = round(float(np.var(fake_scores)), 2)

            # Find peak timestamp
            peak_item = max(frames_with_faces, key=lambda f: f["fake_score"])
            peak_timestamp = peak_item["timestamp_sec"]
            peak_frame_index = peak_item["frame_index"]

            # Number of frames flagged as FAKE
            manipulated_count = sum(1 for f in frames_with_faces if f["fake_score"] >= 50.0)
            manipulated_ratio = round((manipulated_count / face_frames_count) * 100.0, 2)

            # High uncertainty count
            uncertain_count = sum(1 for f in frames_with_faces if f["uncertainty_score"] > 60.0)
        else:
            mean_fake_score = 0.0
            peak_fake_score = 0.0
            temporal_variance = 0.0
            peak_timestamp = 0.0
            peak_frame_index = 0
            manipulated_count = 0
            manipulated_ratio = 0.0
            uncertain_count = 0

        # Assess temporal stability
        if temporal_variance > 400.0:
            temporal_stability = "High Temporal Instability / Discontinuity Detected"
        elif temporal_variance > 150.0:
            temporal_stability = "Moderate Inter-Frame Variance"
        else:
            temporal_stability = "Temporally Consistent"

        # Determine overall forensic verdict
        if face_frames_count == 0:
            verdict = "NO FACES DETECTED IN SAMPLED FRAMES"
        elif manipulated_ratio >= 30.0 or peak_fake_score >= 70.0:
            verdict = "MANIPULATION INDICATED"
        elif peak_fake_score >= 50.0 or uncertain_count / face_frames_count >= 0.50:
            verdict = "INCONCLUSIVE / ELEVATED UNCERTAINTY"
        elif mean_fake_score <= 35.0 and peak_fake_score < 50.0:
            verdict = "NO MANIPULATION DETECTED (AUTHENTIC INDICATED)"
        else:
            verdict = "INCONCLUSIVE"

        # --------------------------------------------------
        # EXTRACT TOP SUSPICIOUS FRAMES WITH GRAD-CAM EXPLANATION
        # --------------------------------------------------
        suspicious_frames_out = []

        if face_frames_count > 0:
            # Sort descending by fake score
            ranked_frames = sorted(frames_with_faces, key=lambda f: f["fake_score"], reverse=True)
            top_candidates = ranked_frames[:4]  # Take top 4 most suspicious frames

            for rank, candidate in enumerate(top_candidates):
                frame_img = candidate["raw_frame"]
                box = candidate["top_face_box"]
                f_idx = candidate["frame_index"]
                f_time = candidate["timestamp_sec"]
                f_score = candidate["fake_score"]

                # Draw bounding box and label overlay on candidate frame
                annotated = frame_img.copy()
                if box is not None:
                    bx, by, bw, bh = box
                    color = (0, 0, 255) if f_score >= 50.0 else (0, 255, 0)
                    cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)
                    tag = f"{candidate['label']} ({f_score:.1f}%) @ {f_time}s"
                    cv2.putText(annotated, tag, (bx, max(20, by - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                frame_filename = f"vid_frame_{uuid.uuid4().hex[:8]}_{f_idx}.jpg"
                frame_save_path = frames_dir / frame_filename
                cv2.imwrite(str(frame_save_path), annotated)

                # Generate Grad-CAM on top face crop
                gradcam_info = None
                if candidate["top_face_crop"] is not None:
                    try:
                        gradcam_info = cam_engine.generate_heatmap(
                            candidate["top_face_crop"],
                            target_class=1,
                            save_to_disk=True
                        )
                    except Exception as e:
                        print(f"Video Grad-CAM generation error on frame {f_idx}:", repr(e))

                suspicious_frames_out.append({
                    "rank": rank + 1,
                    "frame_index": f_idx,
                    "timestamp_sec": f_time,
                    "fake_score": f_score,
                    "uncertainty_score": candidate["uncertainty_score"],
                    "label": candidate["label"],
                    "frame_image_url": f"/uploads/video_frames/{frame_filename}",
                    "gradcam": {
                        "heatmap_url": f"/uploads/{gradcam_info['saved_filename']}" if (gradcam_info and gradcam_info.get("saved_filename")) else None,
                        "focal_region": gradcam_info["focal_region"] if gradcam_info else "N/A",
                        "peak_intensity": gradcam_info["peak_intensity"] if gradcam_info else 0.0,
                        "interpretation": gradcam_info["interpretation"] if gradcam_info else "N/A"
                    } if gradcam_info else None
                })

        # Strip unpickleable NumPy objects from timeline before returning
        clean_timeline = [
            {
                "frame_index": t["frame_index"],
                "timestamp_sec": t["timestamp_sec"],
                "face_count": t["face_count"],
                "fake_score": t["fake_score"],
                "label": t["label"],
                "uncertainty_score": t["uncertainty_score"],
                "faces": t["faces"]
            }
            for t in timeline
        ]

        return {
            "status": "success",
            "video_metadata": meta,
            "analysis_parameters": {
                "sample_fps": sample_fps,
                "step_frames": step,
                "total_sampled_frames": total_sampled,
                "facial_frames_count": face_frames_count
            },
            "verdict": verdict,
            "temporal_summary": {
                "manipulated_frames_count": manipulated_count,
                "manipulated_frame_ratio_percent": manipulated_ratio,
                "mean_fake_score": mean_fake_score,
                "peak_fake_score": peak_fake_score,
                "peak_timestamp_sec": peak_timestamp,
                "peak_frame_index": peak_frame_index,
                "temporal_variance": temporal_variance,
                "temporal_stability": temporal_stability
            },
            "timeline": clean_timeline,
            "suspicious_frames": suspicious_frames_out,
            "model": "DFX Dual-Engine (Swin + ViT) with YuNet Temporal Facial Tracking",
            "forensic_disclaimer": (
                "Video manipulation scores represent frame-level probabilistic indicators. "
                "They should be evaluated alongside visual artifacts and audio tracks, "
                "not as standalone definitive proof."
            )
        }

