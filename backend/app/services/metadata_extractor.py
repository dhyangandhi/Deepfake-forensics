import io
import math
from typing import Dict, Any, Optional
from PIL import Image, ExifTags, ImageChops, ImageEnhance
import numpy as np

# Standard Independent JPEG Group (IJG) 50% Quality Luminance Quantization Table
IJG_LUMINANCE_TABLE_50 = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
]

KNOWN_EDITING_SOFTWARE = [
    "photoshop", "gimp", "lightroom", "pixlr", "canva", "snapseed",
    "vsco", "afterlight", "facetune", "procreate", "paint.net",
    "midjourney", "stable diffusion", "dall-e", "novelai", "automatic1111"
]


def estimate_jpeg_quality(image: Image.Image) -> Optional[int]:
    """
    Estimates the JPEG quality factor (1-100) from the luminance quantization table.
    """
    if not hasattr(image, "quantization") or not image.quantization:
        return None

    lum_table = image.quantization.get(0)
    if not lum_table or len(lum_table) < 64:
        return None

    # Calculate average ratio between actual table and IJG standard table at 50%
    ratios = []
    for actual, standard in zip(lum_table[:64], IJG_LUMINANCE_TABLE_50):
        if standard > 0 and actual > 0:
            ratios.append(float(actual) / float(standard))

    if not ratios:
        return None

    avg_ratio = sum(ratios) / len(ratios)

    if avg_ratio <= 0:
        return 100

    if avg_ratio < 1.0:
        # Scale for qualities 51 - 100
        quality = 100.0 - (avg_ratio * 50.0)
    else:
        # Scale for qualities 1 - 50
        quality = 50.0 / avg_ratio

    return int(max(1, min(100, round(quality))))


def compute_ela_score(image: Image.Image, quality: int = 95) -> Dict[str, Any]:
    """
    Computes Error Level Analysis (ELA) to evaluate compression inconsistency.
    """
    rgb_image = image.convert("RGB")
    buf = io.BytesIO()
    rgb_image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    diff = ImageChops.difference(rgb_image, resaved)
    diff_arr = np.array(diff, dtype=np.float32)

    mean_diff = float(np.mean(diff_arr))
    max_diff = float(np.max(diff_arr))
    std_diff = float(np.std(diff_arr))

    # A very high variance in error levels across the image suggests non-uniform compression
    inconsistency_risk = "HIGH" if std_diff > 8.0 else ("MODERATE" if std_diff > 4.5 else "LOW")

    return {
        "mean_error": round(mean_diff, 2),
        "max_error": round(max_diff, 2),
        "error_variance": round(std_diff, 2),
        "inconsistency_risk": inconsistency_risk
    }


def extract_forensic_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """
    Performs forensic metadata inspection on raw image bytes:
    - EXIF hardware tags (Camera Make, Model, Lens)
    - Software and editor footprints
    - Compression ratio and estimated quality
    - ELA consistency metrics
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Unable to parse image metadata: {str(e)}"}

    width, height = image.size
    image_format = image.format or "UNKNOWN"
    raw_size_bytes = len(image_bytes)
    uncompressed_bytes = width * height * 3
    compression_ratio = round(uncompressed_bytes / max(1, raw_size_bytes), 2)

    estimated_quality = estimate_jpeg_quality(image) if image_format == "JPEG" else None
    ela_metrics = compute_ela_score(image) if image_format in ["JPEG", "PNG", "WEBP"] else None

    # Parse EXIF tags
    exif_data = {}
    software_tag = None
    camera_make = None
    camera_model = None
    date_original = None

    try:
        raw_exif = image.getexif()
        if raw_exif:
            for tag_id, val in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                str_val = str(val).strip()

                if tag_name == "Software":
                    software_tag = str_val
                elif tag_name == "Make":
                    camera_make = str_val
                elif tag_name == "Model":
                    camera_model = str_val
                elif tag_name == "DateTimeOriginal" or tag_name == "DateTime":
                    date_original = str_val

                # Capture compact serializable tags
                if len(str_val) < 80:
                    exif_data[tag_name] = str_val
    except Exception:
        pass

    # Check for known editing software
    editing_detected = False
    detected_software_name = None
    if software_tag:
        software_lower = software_tag.lower()
        for editor in KNOWN_EDITING_SOFTWARE:
            if editor in software_lower:
                editing_detected = True
                detected_software_name = software_tag
                break

    # Determine EXIF integrity risk
    has_camera_hardware = bool(camera_make or camera_model)
    has_exif = bool(exif_data)

    if editing_detected:
        exif_risk = "HIGH_RISK_EDITED"
        risk_explanation = f"Image metadata indicates editing in '{detected_software_name}'."
    elif not has_exif:
        exif_risk = "MODERATE_RISK_STRIPPED"
        risk_explanation = "Metadata is completely stripped. Typical of generative AI engines, web downloads, or messaging apps."
    elif has_camera_hardware:
        exif_risk = "LOW_RISK_HARDWARE_CONFIRMED"
        risk_explanation = f"Valid physical camera hardware tags detected ({camera_make} {camera_model})."
    else:
        exif_risk = "NEUTRAL"
        risk_explanation = "Basic metadata present, but no specific camera hardware signature found."

    return {
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": raw_size_bytes,
        "compression_ratio": compression_ratio,
        "estimated_jpeg_quality": estimated_quality,
        "exif_present": has_exif,
        "camera_make": camera_make,
        "camera_model": camera_model,
        "software": software_tag,
        "date_original": date_original,
        "editing_detected": editing_detected,
        "exif_tampering_risk": exif_risk,
        "risk_explanation": risk_explanation,
        "ela_metrics": ela_metrics,
        "exif_tags": exif_data
    }

