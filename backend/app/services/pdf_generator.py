import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    Image as RLImage
)


class ForensicPDFGenerator:
    """
    Generates formal forensic laboratory dossiers in PDF format conforming
    to digital forensics reporting standards (SWGDE / ISO/IEC 27037 compliant structure).
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.title_style = ParagraphStyle(
            "DocTitle",
            parent=self.styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            fontName="Helvetica-Bold",
            spaceAfter=4
        )
        self.subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=self.styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#0284c7"),
            fontName="Helvetica-Bold",
            textTransform="uppercase",
            spaceAfter=12
        )
        self.heading2_style = ParagraphStyle(
            "Heading2Custom",
            parent=self.styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1e293b"),
            fontName="Helvetica-Bold",
            spaceBefore=10,
            spaceAfter=6
        )
        self.body_style = ParagraphStyle(
            "BodyCustom",
            parent=self.styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
            fontName="Helvetica"
        )
        self.disclaimer_style = ParagraphStyle(
            "DisclaimerCustom",
            parent=self.styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748b"),
            fontName="Helvetica-Oblique"
        )

    def generate_pdf(self, dossier: Dict[str, Any], output_path: Path) -> Path:
        """
        Builds a multi-page PDF forensic report from a dossier dictionary.
        """
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        elements = []

        # 1. Header & Title
        elements.append(Paragraph("DEEPFAKE FORENSICS AI (DFX)", self.subtitle_style))
        elements.append(Paragraph("FORENSIC EXAMINATION DOSSIER & LABORATORY REPORT", self.title_style))
        elements.append(Spacer(1, 8))

        # 2. Case & File Metadata Box
        report_id = dossier.get("report_id", "DFX-CASE-UNKNOWN")
        media_type = dossier.get("media_type", "DIGITAL_MEDIA")
        filename = dossier.get("filename", "Unknown")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        meta_data = [
            [Paragraph("<b>Report ID:</b>", self.body_style), Paragraph(report_id, self.body_style),
             Paragraph("<b>Examination Date:</b>", self.body_style), Paragraph(now_str, self.body_style)],
            [Paragraph("<b>Target Media:</b>", self.body_style), Paragraph(filename, self.body_style),
             Paragraph("<b>Media Modality:</b>", self.body_style), Paragraph(media_type, self.body_style)]
        ]

        t_meta = Table(meta_data, colWidths=[90, 180, 110, 160])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(t_meta)
        elements.append(Spacer(1, 14))

        # 3. Overall Forensic Verdict Banner
        verdict = dossier.get("verdict", "INCONCLUSIVE")
        is_manip = "MANIPULATION" in verdict or "SYNTHETIC" in verdict
        is_auth = "AUTHENTIC" in verdict or "NATURAL" in verdict

        if is_manip:
            bg_color = colors.HexColor("#fee2e2")
            border_color = colors.HexColor("#ef4444")
            text_color = colors.HexColor("#b91c1c")
        elif is_auth:
            bg_color = colors.HexColor("#dcfce7")
            border_color = colors.HexColor("#22c55e")
            text_color = colors.HexColor("#15803d")
        else:
            bg_color = colors.HexColor("#fef3c7")
            border_color = colors.HexColor("#f59e0b")
            text_color = colors.HexColor("#b45309")

        verdict_style = ParagraphStyle(
            "VerdictStyle",
            parent=self.styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=text_color,
            fontName="Helvetica-Bold",
            alignment=1
        )

        verdict_table = Table(
            [[Paragraph(f"FORENSIC EXAMINATION VERDICT: {verdict}", verdict_style)]],
            colWidths=[540]
        )
        verdict_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 1.5, border_color),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(verdict_table)
        elements.append(Spacer(1, 14))

        # 4. Modality-Specific Findings
        if media_type == "IMAGE":
            # Image Findings
            elements.append(Paragraph("1. Image Integrity & Metadata Inspection", self.heading2_style))
            integ = dossier.get("integrity_summary", {})
            img_table_data = [
                ["EXIF Tampering Risk", str(integ.get("tampering_risk", "UNKNOWN"))],
                ["Software Editing Signatures", "DETECTED" if integ.get("editing_software_detected") else "None Reported"],
                ["Editing / Authoring Tool", str(integ.get("software") or "None (Clean Metadata)")],
                ["Hardware Make / Model", str(integ.get("camera_hardware") or "Not Recorded (Stripped)")],
                ["Estimated JPEG Quality", f"Q={integ.get('estimated_jpeg_quality')}" if integ.get('estimated_jpeg_quality') else "N/A (Lossless/WebP)"],
                ["Compression Ratio", f"{integ.get('compression_ratio')} : 1" if integ.get('compression_ratio') else "N/A"]
            ]
            t_img = Table(img_table_data, colWidths=[200, 340])
            t_img.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_img)
            elements.append(Spacer(1, 12))

            # Facial findings
            elements.append(Paragraph("2. Facial Manipulation & Spatial Attention Findings", self.heading2_style))
            faces = dossier.get("facial_findings", [])
            if faces:
                face_rows = [["ID", "Label", "Dual-Engine Score", "Uncertainty", "Grad-CAM Attribution"]]
                for f in faces:
                    face_rows.append([
                        f"#{f.get('face_id')}",
                        f.get('label', 'N/A'),
                        f"{f.get('confidence', 0)}% (Real: {f.get('real_score')}%, Fake: {f.get('fake_score')}%)",
                        f"{f.get('uncertainty_score', 0)}%",
                        str(f.get('gradcam_focal_region', 'N/A'))
                    ])
                t_faces = Table(face_rows, colWidths=[35, 60, 205, 75, 165])
                t_faces.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ('PADDING', (0, 0), (-1, -1), 4),
                ]))
                elements.append(t_faces)
            else:
                elements.append(Paragraph("No facial geometries identified in the target image.", self.body_style))

        elif media_type == "VIDEO":
            # Video Findings
            elements.append(Paragraph("1. Temporal Sequence & Inter-Frame Dynamics", self.heading2_style))
            temp = dossier.get("temporal_summary", {})
            vmeta = dossier.get("video_metadata", {})
            vid_table_data = [
                ["Duration / Total Frames", f"{vmeta.get('duration_sec', 0)}s / {vmeta.get('total_frames', 0)} frames"],
                ["Sampling Strategy", f"Uniform Stride ({dossier.get('timeline_sample_count', 0)} sample points)"],
                ["Manipulated Frame Ratio", f"{temp.get('manipulated_frame_ratio_percent', 0)}% ({temp.get('manipulated_frames_count', 0)} frames flagged)"],
                ["Peak Manipulation Score", f"{temp.get('peak_fake_score', 0)}% @ {temp.get('peak_timestamp_sec', 0)}s"],
                ["Mean Sequence Score", f"{temp.get('mean_fake_score', 0)}%"],
                ["Inter-Frame Temporal Stability", str(temp.get('temporal_stability', 'N/A'))],
                ["Temporal Variance", str(temp.get('temporal_variance', 0))]
            ]
            t_vid = Table(vid_table_data, colWidths=[200, 340])
            t_vid.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_vid)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("2. Top Suspicious Temporal Frames", self.heading2_style))
            sf_list = dossier.get("suspicious_frames", [])
            if sf_list:
                sf_rows = [["Rank", "Timestamp", "Frame Index", "Score / Label", "Grad-CAM Focal Region"]]
                for sf in sf_list:
                    cam_region = sf.get("gradcam", {}).get("focal_region", "N/A") if sf.get("gradcam") else "N/A"
                    sf_rows.append([
                        f"#{sf.get('rank')}",
                        f"{sf.get('timestamp_sec')}s",
                        f"Frame {sf.get('frame_index')}",
                        f"{sf.get('label')} ({sf.get('fake_score')}%)",
                        str(cam_region)
                    ])
                t_sf = Table(sf_rows, colWidths=[40, 80, 90, 130, 200])
                t_sf.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ('PADDING', (0, 0), (-1, -1), 4),
                ]))
                elements.append(t_sf)

        elif media_type == "AUDIO":
            # Audio Findings
            elements.append(Paragraph("1. Acoustic Biometrics & Vocal Resonance", self.heading2_style))
            ameta = dossier.get("audio_metadata", {})
            ind = dossier.get("indicators", {})
            scores = dossier.get("scores", {})

            aud_table_data = [
                ["Sampling Frequency / Channels", f"{ameta.get('sample_rate', 0)} Hz / {ameta.get('channels', 1)} channel(s)"],
                ["Duration / Bit Depth", f"{ameta.get('duration_sec', 0)}s / {ameta.get('bit_depth', 16)}-bit PCM"],
                ["Synthetic Voice Probability", f"{scores.get('synthetic_probability', 0)}% (Uncertainty: {scores.get('uncertainty_score', 0)}%)"],
                ["Glottal Pitch Jitter (Perturbation)", f"{ind.get('pitch_jitter_percent', 0)}% (Human Standard: 0.4%–1.8%)"],
                ["Amplitude Shimmer", f"{ind.get('pitch_shimmer_percent', 0)}% (Human Standard: 1.5%–4.5%)"],
                ["High-Frequency Brickwall Shelf", "DETECTED (Shelf: " + str(ind.get('cutoff_frequency_hz')) + " Hz)" if ind.get('high_frequency_brickwall') else "ABSENT (Continuous Full-Band)"],
                ["Respiration Silence Floor", f"{ind.get('silence_floor_db', 0)} dB ({'Digital Mathematical Zero' if ind.get('artificial_silence_detected') else 'Natural Acoustic Floor'})"],
                ["Spectral Flatness (Wiener Entropy)", str(ind.get('spectral_flatness', 0))]
            ]
            t_aud = Table(aud_table_data, colWidths=[200, 340])
            t_aud.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_aud)

        # 5. Formal Forensic Limitations & Legal Disclaimer
        elements.append(Spacer(1, 16))
        elements.append(Paragraph("3. Scientific Limitations & Legal Notice", self.heading2_style))
        disclaimer_text = (
            "NOTICE: This forensic examination dossier was produced using algorithmically calibrated "
            "deep learning and digital signal processing models (DFX v3.0). All scores, percentages, and "
            "heatmaps denote statistical feature correlations and biometric anomaly distributions. "
            "They do NOT constitute conclusive legal proof or definitive judicial certification of media "
            "authenticity. Findings should be corroborated by independent certified forensic examiners "
            "and evaluated within the full context of digital chain-of-custody."
        )
        elements.append(Paragraph(disclaimer_text, self.disclaimer_style))

        # Build document
        doc.build(elements)
        return output_path

