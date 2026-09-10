from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

COLOR_FEATURES = [
    "hue_mean",
    "hue_std",
    "saturation_mean",
    "value_mean",
    "lab_a_mean",
    "lab_b_mean",
]


@dataclass(frozen=True)
class ClientDeviceSegmentationConfig:
    """Provisional selector learned from the first real client-device mouse cohort.

    These values identify visible thermochromic response in the current acquisition
    setup. They are not a temperature calibration and must not be interpreted as
    degrees Celsius or a clinical decision boundary.
    """

    saturation_min: int = 55
    pixel_value_min: int = 85
    component_area_min_px: int = 150
    component_mean_value_min: float = 160.0
    open_kernel_px: int = 3
    close_kernel_px: int = 5


DEFAULT_CLIENT_CONFIG = ClientDeviceSegmentationConfig()


def letterbox_square(bgr: np.ndarray, out_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image")
    h, w = bgr.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    valid = np.zeros((side, side), dtype=np.uint8)
    oy = (side - h) // 2
    ox = (side - w) // 2
    canvas[oy : oy + h, ox : ox + w] = bgr
    valid[oy : oy + h, ox : ox + w] = 255
    canvas = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(valid, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return canvas, valid > 0


def _branch_end_points(skel: np.ndarray) -> tuple[int, int]:
    s = skel.astype(np.uint8)
    padded = np.pad(s, 1)
    neighbours = np.zeros_like(s, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dx == 1 and dy == 1:
                continue
            neighbours += padded[dy : dy + s.shape[0], dx : dx + s.shape[1]]
    return int(((s == 1) & (neighbours >= 3)).sum()), int(
        ((s == 1) & (neighbours == 1)).sum()
    )


def segment_client_response(
    bgr: np.ndarray,
    valid_mask: np.ndarray | None = None,
    config: ClientDeviceSegmentationConfig = DEFAULT_CLIENT_CONFIG,
) -> tuple[np.ndarray, list[dict]]:
    """Segment high-confidence visible TLC response for the current client setup.

    The first client cohort exposed a failure mode in the generic reference mask:
    dim, saturated background texture was being accepted as response. The client
    selector therefore keeps only sufficiently large components whose mean value
    is bright enough, while white glare is naturally suppressed by saturation.
    """

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if valid_mask is None:
        valid_mask = np.ones(hsv.shape[:2], dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != hsv.shape[:2]:
            raise ValueError("valid_mask must match the image height/width")

    raw = (
        valid_mask
        & (hsv[:, :, 1] > config.saturation_min)
        & (hsv[:, :, 2] > config.pixel_value_min)
    ).astype(np.uint8) * 255

    open_kernel = np.ones((config.open_kernel_px, config.open_kernel_px), np.uint8)
    close_kernel = np.ones((config.close_kernel_px, config.close_kernel_px), np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_kernel)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, 8)
    clean = np.zeros_like(raw)
    selected = []
    height, width = raw.shape
    for component_id in range(1, n_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < config.component_area_min_px:
            continue
        component = labels == component_id
        mean_value = float(hsv[:, :, 2][component].mean())
        if mean_value < config.component_mean_value_min:
            continue
        clean[component] = 255
        selected.append(
            {
                "area_px": area,
                "mean_value": round(mean_value, 6),
                "mean_saturation": round(float(hsv[:, :, 1][component].mean()), 6),
                "centroid_x_norm": round(float(centroids[component_id, 0] / width), 6),
                "centroid_y_norm": round(float(centroids[component_id, 1] / height), 6),
            }
        )
    selected.sort(key=lambda item: item["area_px"], reverse=True)
    return clean, selected


def summarize_response(
    bgr: np.ndarray,
    response_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict:
    mask = np.asarray(response_mask) > 0
    if valid_mask is None:
        valid_mask = np.ones(mask.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
    valid_area = max(int(valid_mask.sum()), 1)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    regions = regionprops(label(mask))
    largest = max(regions, key=lambda region: region.area) if regions else None
    area = int(mask.sum())

    if largest is None:
        largest_fraction = eccentricity = solidity = perimeter = elongation = 0.0
    else:
        largest_fraction = float(largest.area / valid_area)
        eccentricity = float(largest.eccentricity)
        solidity = float(largest.solidity)
        perimeter = float(largest.perimeter)
        major = float(getattr(largest, "axis_major_length", 0.0))
        minor = float(getattr(largest, "axis_minor_length", 0.0))
        elongation = major / max(minor, 1e-6)

    skeleton = skeletonize(mask)
    branches, endpoints = _branch_end_points(skeleton)

    if area:
        ys, xs = np.where(mask)
        hue = hsv[:, :, 0][mask].astype(float)
        sat = hsv[:, :, 1][mask].astype(float)
        value = hsv[:, :, 2][mask].astype(float)
        lab_a = lab[:, :, 1][mask].astype(float)
        lab_b = lab[:, :, 2][mask].astype(float)
        centroid_x = float(xs.mean() / bgr.shape[1])
        centroid_y = float(ys.mean() / bgr.shape[0])
        hue_mean, hue_std = float(hue.mean()), float(hue.std())
        saturation_mean = float(sat.mean())
        value_mean = float(value.mean())
        lab_a_mean = float(lab_a.mean())
        lab_b_mean = float(lab_b.mean())
    else:
        centroid_x = centroid_y = 0.0
        hue_mean = hue_std = saturation_mean = value_mean = 0.0
        lab_a_mean = lab_b_mean = 0.0

    area_fraction = float(area / valid_area)
    if area_fraction < 0.02:
        morphology = "minimal-response"
    elif len(regions) >= 5 and largest_fraction < 0.04:
        morphology = "scattered"
    elif elongation >= 3.0 and area_fraction < 0.22:
        morphology = "linear-like"
    elif branches >= 8 and elongation >= 1.6:
        morphology = "branched/complex"
    elif area_fraction >= 0.12:
        morphology = "diffuse/large-region"
    elif largest_fraction >= 0.04:
        morphology = "focal/irregular-region"
    else:
        morphology = "mixed"

    return {
        "response_area_fraction": area_fraction,
        "component_count": len(regions),
        "largest_component_fraction": largest_fraction,
        "largest_eccentricity": eccentricity,
        "largest_solidity": solidity,
        "largest_perimeter_px": perimeter,
        "largest_elongation": elongation,
        "skeleton_length_px": int(skeleton.sum()),
        "branch_pixels": branches,
        "endpoint_pixels": endpoints,
        "response_centroid_x_norm": centroid_x,
        "response_centroid_y_norm": centroid_y,
        "hue_mean": hue_mean,
        "hue_std": hue_std,
        "saturation_mean": saturation_mean,
        "value_mean": value_mean,
        "lab_a_mean": lab_a_mean,
        "lab_b_mean": lab_b_mean,
        "morphology_descriptor": morphology,
    }


def analyze_client_image(
    path: str | Path,
    config: ClientDeviceSegmentationConfig = DEFAULT_CLIENT_CONFIG,
) -> tuple[dict, np.ndarray, np.ndarray]:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    normalized, valid = letterbox_square(image)
    mask, components = segment_client_response(normalized, valid, config)
    features = summarize_response(normalized, mask, valid)
    features.update(
        {
            "source_image": path.name,
            "selected_components": len(components),
            "segmentation_config": asdict(config),
            "clinical_claim": "NONE",
            "absolute_temperature_interpretation": "DISABLED_UNCALIBRATED_TLC",
        }
    )
    return features, normalized, mask


def robust_colour_domain_shift(
    reference_features: pd.DataFrame,
    client_features: pd.DataFrame,
) -> dict:
    """Compare profile colour summaries in reference-IQR units.

    This is a domain-shift indicator only. It is intentionally not a disease score.
    """

    rows = []
    for column in COLOR_FEATURES:
        reference = pd.to_numeric(reference_features[column], errors="coerce").dropna()
        client = pd.to_numeric(client_features[column], errors="coerce").dropna()
        q1, q3 = reference.quantile([0.25, 0.75])
        iqr = max(float(q3 - q1), 1e-6)
        ref_median = float(reference.median())
        client_median = float(client.median())
        rows.append(
            {
                "feature": column,
                "reference_median": ref_median,
                "client_median": client_median,
                "client_minus_reference_iqr": (client_median - ref_median) / iqr,
            }
        )
    max_abs = max(abs(row["client_minus_reference_iqr"]) for row in rows)
    return {
        "metric": "median colour-feature shift measured in reference IQR units",
        "max_abs_shift_iqr": max_abs,
        "features": rows,
        "clinical_claim": "NONE",
    }


def threshold_sweep(
    image_paths: Iterable[str | Path],
    saturation_values: Iterable[int] = (45, 50, 55, 60, 65),
    value_values: Iterable[int] = (70, 85, 100),
    component_mean_values: Iterable[float] = (150.0, 160.0, 170.0),
) -> pd.DataFrame:
    records = []
    for saturation_min in saturation_values:
        for pixel_value_min in value_values:
            for component_mean_value_min in component_mean_values:
                config = ClientDeviceSegmentationConfig(
                    saturation_min=saturation_min,
                    pixel_value_min=pixel_value_min,
                    component_mean_value_min=component_mean_value_min,
                )
                cohort = []
                for path in image_paths:
                    features, _, _ = analyze_client_image(path, config)
                    cohort.append(
                        {
                            "source_image": features["source_image"],
                            "response_area_fraction": features["response_area_fraction"],
                        }
                    )
                frame = pd.DataFrame(cohort)
                frame["response_area_rank_desc"] = frame["response_area_fraction"].rank(
                    ascending=False, method="average"
                )
                for row in frame.to_dict("records"):
                    records.append(
                        {
                            "saturation_min": saturation_min,
                            "pixel_value_min": pixel_value_min,
                            "component_mean_value_min": component_mean_value_min,
                            **row,
                        }
                    )
    return pd.DataFrame(records)


def summarize_threshold_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_image, group in sweep.groupby("source_image"):
        area = group["response_area_fraction"].astype(float)
        rank = group["response_area_rank_desc"].astype(float)
        rows.append(
            {
                "source_image": source_image,
                "median_response_area_fraction": float(area.median()),
                "response_area_q25": float(area.quantile(0.25)),
                "response_area_q75": float(area.quantile(0.75)),
                "median_blind_rank": float(rank.median()),
                "rank_q25": float(rank.quantile(0.25)),
                "rank_q75": float(rank.quantile(0.75)),
                "rank_min": float(rank.min()),
                "rank_max": float(rank.max()),
                "clinical_claim": "NONE",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["median_blind_rank", "median_response_area_fraction"],
        ascending=[True, False],
    )
