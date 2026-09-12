from __future__ import annotations

import cv2
import numpy as np


# DMR-IR cancer cases are not bilaterally segmented in the same way as healthy
# controls: most cancer subjects expose only one breast side in the source
# segmentation set. Using side presence/absence or requiring both sides would
# therefore create label-confounded selection. v0.2.1 trains only on local
# spatial morphology that is available on both classes, while MumGuard keeps
# its bilateral asymmetry channel as an independent target-domain evidence lane.
LOCAL_SPATIAL_FEATURES = (
    "std_z",
    "hotspot_peak_z",
    "local_contrast_p95",
    "gradient_p90",
    "hotspot_coherence",
    "hotspot_center_distance",
    "center_periphery_abs_diff",
    "left_right_abs_diff",
    "top_bottom_abs_diff",
)

FEATURE_CONTRACT = "dimensionless_local_spatial_v0_2_1"


def resample_supported_field(
    field: np.ndarray,
    support: np.ndarray | None = None,
    *,
    size: int = 64,
) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("local transfer fields must be 2-D")
    valid = np.isfinite(values)
    if support is not None:
        mask = np.asarray(support, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("support mask must match field shape")
        valid &= mask
    if int(valid.sum()) < 64:
        raise ValueError("at least 64 supported pixels are required")
    if int(size) < 16:
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


def _affine_normalize(field: np.ndarray) -> tuple[np.ndarray, dict]:
    values = field[np.isfinite(field)]
    if values.size < 64:
        raise ValueError("field has insufficient supported pixels")
    center = float(np.median(values))
    p10, p90 = np.percentile(values, [10, 90])
    scale = float(p90 - p10)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("field has degenerate thermal spread")
    z = (field - center) / scale
    return z, {
        "center": center,
        "scale": scale,
        "normalization": "per_field_median_and_p90_minus_p10",
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


def _region_mean(z: np.ndarray, region: np.ndarray) -> float:
    values = z[region & np.isfinite(z)]
    if values.size < 8:
        return 0.0
    return float(values.mean())


def extract_dimensionless_local_spatial_features(
    field: np.ndarray,
    support: np.ndarray | None = None,
    *,
    size: int = 64,
) -> dict[str, float]:
    """Extract positive-affine-invariant 2-D morphology from one breast field.

    The feature vector is invariant to ``field -> a*field+b`` for positive ``a``.
    It deliberately excludes bilateral availability, absolute Celsius and source
    acquisition metadata so DMR-IR side coverage cannot become a disease proxy.
    """

    resized = resample_supported_field(field, support, size=size)
    z, _ = _affine_normalize(resized)
    valid = np.isfinite(z)
    selected = z[valid]
    if selected.size < 64:
        raise ValueError("insufficient normalized support")

    p90 = float(np.percentile(selected, 90))
    p99 = float(np.percentile(selected, 99))
    smooth, smooth_support = _supported_smooth(z)
    residual = z - smooth
    residual_values = residual[smooth_support & np.isfinite(residual)]
    if residual_values.size < 32:
        raise ValueError("insufficient support for local contrast")
    local_contrast_p95 = float(np.percentile(np.maximum(residual_values, 0.0), 95))

    smooth_filled = np.where(np.isfinite(smooth), smooth, 0.0).astype(np.float32)
    gx = cv2.Sobel(smooth_filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth_filled, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    interior = cv2.erode(smooth_support.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    gradient_values = gradient[interior]
    if gradient_values.size < 32:
        gradient_values = gradient[smooth_support]
    gradient_p90 = float(np.percentile(gradient_values, 90))

    hotspot = valid & (z >= p90)
    coherence, center_distance = _largest_component_stats(hotspot)

    h, w = z.shape
    yy, xx = np.mgrid[:h, :w]
    center = valid & (xx >= 0.25 * w) & (xx < 0.75 * w) & (yy >= 0.25 * h) & (yy < 0.75 * h)
    periphery = valid & ~center
    left = valid & (xx < w / 2.0)
    right = valid & ~left
    top = valid & (yy < h / 2.0)
    bottom = valid & ~top

    features = {
        "std_z": float(np.std(selected)),
        "hotspot_peak_z": p99,
        "local_contrast_p95": local_contrast_p95,
        "gradient_p90": gradient_p90,
        "hotspot_coherence": float(coherence),
        "hotspot_center_distance": float(center_distance),
        "center_periphery_abs_diff": abs(_region_mean(z, center) - _region_mean(z, periphery)),
        "left_right_abs_diff": abs(_region_mean(z, left) - _region_mean(z, right)),
        "top_bottom_abs_diff": abs(_region_mean(z, top) - _region_mean(z, bottom)),
    }
    if tuple(features) != LOCAL_SPATIAL_FEATURES:
        raise RuntimeError("local spatial feature contract drifted")
    if not np.isfinite(list(features.values())).all():
        raise ValueError("local spatial transfer features contain NaN/Inf")
    return {name: float(features[name]) for name in LOCAL_SPATIAL_FEATURES}
