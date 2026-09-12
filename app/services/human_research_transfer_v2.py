from __future__ import annotations

import cv2
import numpy as np


# v0.2 deliberately removes both additive offset and positive multiplicative
# scale before extracting morphology. That makes the contract portable between
# source-domain Celsius thermography and MumGuard's relative Contact-TLC signal
# without pretending that the two numeric scales are interchangeable.
BILATERAL_SPATIAL_FEATURES = (
    "max_side_hotspot_peak_z",
    "max_side_local_contrast_p95",
    "max_side_gradient_p90",
    "max_side_hotspot_coherence",
    "max_side_hotspot_center_distance",
    "bilateral_mean_abs_diff",
    "bilateral_std_abs_diff",
    "bilateral_p90_abs_diff",
    "bilateral_hotspot_peak_abs_diff",
    "bilateral_local_contrast_abs_diff",
    "bilateral_gradient_abs_diff",
    "bilateral_coherence_abs_diff",
    "bilateral_pattern_mae",
    "bilateral_pattern_corr_distance",
)

FEATURE_CONTRACT = "dimensionless_bilateral_spatial_v0"


def resample_supported_field(
    field: np.ndarray,
    support: np.ndarray | None = None,
    *,
    size: int = 64,
) -> np.ndarray:
    """Resize a partially-observed 2-D field without turning missing pixels into data."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("bilateral transfer fields must be 2-D")
    valid = np.isfinite(values)
    if support is not None:
        mask = np.asarray(support, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("support mask must match field shape")
        valid &= mask
    if int(valid.sum()) < 64:
        raise ValueError("at least 64 supported pixels are required per side")
    if size < 16:
        raise ValueError("resampled field size must be at least 16")

    numerator = np.where(valid, values, 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    target = (int(size), int(size))
    num_r = cv2.resize(numerator, target, interpolation=cv2.INTER_AREA)
    weight_r = cv2.resize(weights, target, interpolation=cv2.INTER_AREA)
    out = np.full((size, size), np.nan, dtype=np.float64)
    keep = weight_r >= 0.25
    out[keep] = num_r[keep] / np.maximum(weight_r[keep], 1e-6)
    if int(np.isfinite(out).sum()) < 64:
        raise ValueError("resampling left too little supported tissue")
    return out


def _joint_affine_normalize(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    left_values = left[np.isfinite(left)]
    right_values = right[np.isfinite(right)]
    pooled = np.concatenate([left_values, right_values])
    if pooled.size < 128:
        raise ValueError("bilateral session has insufficient supported pixels")

    center = float(np.median(pooled))
    p10, p90 = np.percentile(pooled, [10, 90])
    scale = float(p90 - p10)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(pooled))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("bilateral session has degenerate thermal spread")

    return (left - center) / scale, (right - center) / scale, {
        "center": center,
        "scale": scale,
        "normalization": "pooled_median_and_p90_minus_p10",
    }


def _supported_smooth(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = np.isfinite(values)
    filled = np.where(support, values, 0.0).astype(np.float32)
    weights = support.astype(np.float32)
    numerator = cv2.GaussianBlur(filled, (0, 0), sigmaX=1.5, sigmaY=1.5)
    denominator = cv2.GaussianBlur(weights, (0, 0), sigmaX=1.5, sigmaY=1.5)
    smooth = np.full(values.shape, np.nan, dtype=np.float64)
    keep = support & (denominator >= 0.35)
    smooth[keep] = numerator[keep] / np.maximum(denominator[keep], 1e-6)
    return smooth, keep


def _largest_component_stats(binary: np.ndarray) -> tuple[float, float]:
    mask = np.asarray(binary, dtype=np.uint8)
    count = int(mask.sum())
    if count == 0:
        return 0.0, 1.0
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if labels_count <= 1:
        return 0.0, 1.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    area = float(stats[index, cv2.CC_STAT_AREA])
    coherence = area / float(count)
    cx, cy = centroids[index]
    h, w = mask.shape
    dx = (float(cx) - (w - 1) / 2.0) / max(w - 1, 1)
    dy = (float(cy) - (h - 1) / 2.0) / max(h - 1, 1)
    center_distance = float(np.sqrt(dx * dx + dy * dy) / np.sqrt(0.5))
    return coherence, min(center_distance, 1.0)


def _side_metrics(z: np.ndarray) -> dict[str, float]:
    support = np.isfinite(z)
    selected = z[support]
    if selected.size < 64:
        raise ValueError("each side needs at least 64 normalized supported pixels")

    p90 = float(np.percentile(selected, 90))
    p99 = float(np.percentile(selected, 99))
    smooth, smooth_support = _supported_smooth(z)

    residual = z - smooth
    residual_values = residual[smooth_support & np.isfinite(residual)]
    if residual_values.size < 32:
        raise ValueError("insufficient support for local contrast")
    positive_residual = np.maximum(residual_values, 0.0)
    local_contrast_p95 = float(np.percentile(positive_residual, 95))

    smooth_filled = np.where(np.isfinite(smooth), smooth, 0.0).astype(np.float32)
    gx = cv2.Sobel(smooth_filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth_filled, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    interior = cv2.erode(smooth_support.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    gradient_values = gradient[interior]
    if gradient_values.size < 32:
        gradient_values = gradient[smooth_support]
    gradient_p90 = float(np.percentile(gradient_values, 90))

    hotspot = support & (z >= p90)
    coherence, center_distance = _largest_component_stats(hotspot)

    return {
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "p90": p90,
        "hotspot_peak_z": p99,
        "local_contrast_p95": local_contrast_p95,
        "gradient_p90": gradient_p90,
        "hotspot_coherence": float(coherence),
        "hotspot_center_distance": float(center_distance),
    }


def _pattern_metrics(left_z: np.ndarray, right_z: np.ndarray) -> tuple[float, float]:
    # Mirror the right side into a common anatomical orientation before comparison.
    right_mirrored = np.fliplr(right_z)
    joint = np.isfinite(left_z) & np.isfinite(right_mirrored)
    if int(joint.sum()) < 32:
        raise ValueError("insufficient bilateral overlap for pattern comparison")
    left_values = left_z[joint]
    right_values = right_mirrored[joint]
    mae = float(np.mean(np.abs(left_values - right_values)))
    if float(np.std(left_values)) <= 1e-8 or float(np.std(right_values)) <= 1e-8:
        corr_distance = 1.0
    else:
        corr = float(np.corrcoef(left_values, right_values)[0, 1])
        corr_distance = float(np.clip(1.0 - corr, 0.0, 2.0))
    return mae, corr_distance


def extract_dimensionless_bilateral_spatial_features(
    left_field: np.ndarray,
    right_field: np.ndarray,
    *,
    left_support: np.ndarray | None = None,
    right_support: np.ndarray | None = None,
    size: int = 64,
) -> dict[str, float]:
    """Extract source/target-shared spatial features from a bilateral human exam.

    The same feature vector is invariant to `field -> a*field+b` for any positive
    scalar `a`, because both sides are jointly normalized before morphology is
    measured. No feature is Celsius-specific and no disease label is produced.
    """

    left = resample_supported_field(left_field, left_support, size=size)
    right = resample_supported_field(right_field, right_support, size=size)
    left_z, right_z, _ = _joint_affine_normalize(left, right)

    left_metrics = _side_metrics(left_z)
    right_metrics = _side_metrics(right_z)
    pattern_mae, pattern_corr_distance = _pattern_metrics(left_z, right_z)

    features = {
        "max_side_hotspot_peak_z": max(left_metrics["hotspot_peak_z"], right_metrics["hotspot_peak_z"]),
        "max_side_local_contrast_p95": max(left_metrics["local_contrast_p95"], right_metrics["local_contrast_p95"]),
        "max_side_gradient_p90": max(left_metrics["gradient_p90"], right_metrics["gradient_p90"]),
        "max_side_hotspot_coherence": max(left_metrics["hotspot_coherence"], right_metrics["hotspot_coherence"]),
        "max_side_hotspot_center_distance": max(left_metrics["hotspot_center_distance"], right_metrics["hotspot_center_distance"]),
        "bilateral_mean_abs_diff": abs(left_metrics["mean"] - right_metrics["mean"]),
        "bilateral_std_abs_diff": abs(left_metrics["std"] - right_metrics["std"]),
        "bilateral_p90_abs_diff": abs(left_metrics["p90"] - right_metrics["p90"]),
        "bilateral_hotspot_peak_abs_diff": abs(left_metrics["hotspot_peak_z"] - right_metrics["hotspot_peak_z"]),
        "bilateral_local_contrast_abs_diff": abs(left_metrics["local_contrast_p95"] - right_metrics["local_contrast_p95"]),
        "bilateral_gradient_abs_diff": abs(left_metrics["gradient_p90"] - right_metrics["gradient_p90"]),
        "bilateral_coherence_abs_diff": abs(left_metrics["hotspot_coherence"] - right_metrics["hotspot_coherence"]),
        "bilateral_pattern_mae": pattern_mae,
        "bilateral_pattern_corr_distance": pattern_corr_distance,
    }
    if tuple(features) != BILATERAL_SPATIAL_FEATURES:
        raise RuntimeError("bilateral spatial feature contract drifted")
    if not np.isfinite(list(features.values())).all():
        raise ValueError("bilateral spatial transfer features contain NaN/Inf")
    return {name: float(features[name]) for name in BILATERAL_SPATIAL_FEATURES}
