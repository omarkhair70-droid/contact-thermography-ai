from __future__ import annotations

import cv2
import numpy as np


QC_SEMANTICS = "engineering client-device acquisition QC; not a clinical assessment"


def assess_client_device_quality(
    bgr: np.ndarray,
    response_mask: np.ndarray,
    valid_mask: np.ndarray,
    edge_band_px: int = 3,
) -> dict:
    """QC for non-circular real-device photographs using the full valid frame.

    Publication-plate QC assumes a central circular disk, which is not appropriate
    for the first client mouse cohort. This gate measures luminance clipping and
    low-saturation specular glare over the actual letterboxed image area, and it
    reports when selected TLC response reaches the acquisition boundary.
    """

    image = np.asarray(bgr, dtype=np.uint8)
    response = np.asarray(response_mask) > 0
    valid = np.asarray(valid_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image")
    if response.shape != image.shape[:2] or valid.shape != image.shape[:2]:
        raise ValueError("response_mask and valid_mask must match image height/width")
    if not valid.any():
        raise ValueError("valid_mask cannot be empty")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Luminance clipping deliberately uses grayscale rather than HSV Value.
    # A saturated green/cyan TLC response can legitimately have V~=255 because
    # one colour channel is high; treating that as white clipping would mark the
    # biological signal itself as an acquisition defect.
    bright_clip = float((gray[valid] >= 252).mean())
    specular = valid & (value >= 245) & (saturation <= 30)
    specular_fraction = float(specular.sum() / valid.sum())
    dynamic_range = float(
        np.percentile(gray[valid], 95) - np.percentile(gray[valid], 5)
    )
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F)[valid].var())

    kernel_size = max(1, int(edge_band_px) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1) > 0
    valid_edge = valid & ~eroded_valid
    response_total = int(response.sum())
    response_edge_pixels = int((response & valid_edge).sum())
    response_edge_fraction = (
        float(response_edge_pixels / response_total) if response_total else 0.0
    )
    response_touches_valid_edge = response_edge_pixels > 0

    response_specular_overlap = int((response & specular).sum())
    response_specular_fraction = (
        float(response_specular_overlap / response_total) if response_total else 0.0
    )

    flags = []
    if specular_fraction > 0.04:
        flags.append("SPECULAR_GLARE_PRESENT")
    if bright_clip > 0.08:
        flags.append("BRIGHT_CLIPPING")
    if response_touches_valid_edge:
        flags.append("RESPONSE_TOUCHES_FRAME")
    if response_specular_fraction > 0.02:
        flags.append("RESPONSE_OVERLAPS_SPECULAR_HIGHLIGHT")
    if dynamic_range < 12:
        flags.append("LOW_CONTRAST")
    if laplacian_variance < 20:
        flags.append("LOW_SHARPNESS")

    severe = {
        "BRIGHT_CLIPPING",
        "LOW_CONTRAST",
        "RESPONSE_OVERLAPS_SPECULAR_HIGHLIGHT",
    }
    severe_count = sum(flag in severe for flag in flags)
    if severe_count >= 2:
        status = "REVIEW_REQUIRED"
    elif flags:
        status = "REVIEW"
    else:
        status = "PASS"

    return {
        "status": status,
        "flags": flags,
        "metrics": {
            "valid_frame_fraction": round(float(valid.mean()), 6),
            "laplacian_variance": round(laplacian_variance, 4),
            "gray_dynamic_range_p95_p5": round(dynamic_range, 4),
            "bright_clip_fraction": round(bright_clip, 6),
            "specular_glare_fraction": round(specular_fraction, 6),
            "response_touches_valid_edge": response_touches_valid_edge,
            "response_edge_fraction": round(response_edge_fraction, 6),
            "response_specular_overlap_fraction": round(response_specular_fraction, 6),
        },
        "semantics": QC_SEMANTICS,
        "clinical_claim": "NONE",
    }
