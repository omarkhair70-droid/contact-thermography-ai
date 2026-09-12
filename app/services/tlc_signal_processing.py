from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


class TLCCalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class HueTemperaturePoint:
    hue_deg: float
    temperature_c: float


@dataclass(frozen=True)
class TLCSignalMapResult:
    mode: str
    signal_map: np.ndarray
    active_mask: np.ndarray
    hue_deg: np.ndarray
    saturation: np.ndarray
    value: np.ndarray
    provenance: dict


def _validate_image_and_mask(
    bgr: np.ndarray,
    response_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with 3 channels")
    mask = np.asarray(response_mask) > 0
    if mask.shape != image.shape[:2]:
        raise ValueError("response_mask must match image height/width")
    return image, mask


def bgr_to_hsv_physical_channels(
    bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hue in degrees plus saturation/value normalized to [0, 1]."""
    image = np.asarray(bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with 3 channels")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_deg = hsv[:, :, 0].astype(np.float32) * 2.0
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    return hue_deg, saturation, value


def _validate_hue_span(low_hue_deg: float, high_hue_deg: float) -> tuple[float, float]:
    low = float(low_hue_deg)
    high = float(high_hue_deg)
    if not np.isfinite([low, high]).all():
        raise TLCCalibrationError("Hue bounds must be finite")
    if high <= low:
        raise TLCCalibrationError("high_hue_deg must be greater than low_hue_deg")
    if low < 0.0 or high > 360.0:
        raise TLCCalibrationError("Hue bounds must stay within [0, 360] degrees")
    return low, high


def build_relative_thermal_map(
    bgr: np.ndarray,
    response_mask: np.ndarray,
    *,
    low_hue_deg: float = 0.0,
    high_hue_deg: float = 240.0,
    tlc_profile_id: str | None = None,
) -> TLCSignalMapResult:
    """Build a relative chromatic thermal index from visible TLC response.

    This is deliberately not Celsius. The configured colour-play span is mapped
    to [0, 1] only inside the active response mask, which lets the rest of the
    pipeline perform spatial/temporal analysis before absolute calibration is
    available.
    """
    image, mask = _validate_image_and_mask(bgr, response_mask)
    low, high = _validate_hue_span(low_hue_deg, high_hue_deg)
    hue_deg, saturation, value = bgr_to_hsv_physical_channels(image)
    relative = np.full(hue_deg.shape, np.nan, dtype=np.float32)
    if mask.any():
        clipped = np.clip(hue_deg[mask], low, high)
        relative[mask] = (clipped - low) / (high - low)
    return TLCSignalMapResult(
        mode="relative_hue_index",
        signal_map=relative,
        active_mask=mask,
        hue_deg=hue_deg,
        saturation=saturation,
        value=value,
        provenance={
            "tlc_profile_id": tlc_profile_id,
            "mapping": "relative_hue_index",
            "low_hue_deg": low,
            "high_hue_deg": high,
            "absolute_temperature": False,
        },
    )


def _normalize_calibration_points(
    points: Sequence[HueTemperaturePoint] | Iterable[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    normalized: list[tuple[float, float]] = []
    for point in points:
        if isinstance(point, HueTemperaturePoint):
            hue, temp = point.hue_deg, point.temperature_c
        else:
            hue, temp = point
        hue = float(hue)
        temp = float(temp)
        if not np.isfinite([hue, temp]).all():
            raise TLCCalibrationError("Calibration points must be finite")
        if hue < 0.0 or hue > 360.0:
            raise TLCCalibrationError("Calibration hue must be within [0, 360] degrees")
        normalized.append((hue, temp))
    if len(normalized) < 2:
        raise TLCCalibrationError("At least two Hue-temperature points are required")
    normalized.sort(key=lambda item: item[0])
    hues = np.asarray([item[0] for item in normalized], dtype=np.float32)
    temps = np.asarray([item[1] for item in normalized], dtype=np.float32)
    if np.any(np.diff(hues) <= 0):
        raise TLCCalibrationError("Calibration Hue values must be strictly increasing")
    return hues, temps


def build_calibrated_temperature_map(
    bgr: np.ndarray,
    response_mask: np.ndarray,
    calibration_points: Sequence[HueTemperaturePoint] | Iterable[tuple[float, float]],
    *,
    tlc_profile_id: str | None = None,
) -> TLCSignalMapResult:
    """Map active-response Hue to Celsius using empirical calibration points.

    Values between measured calibration points use piecewise linear
    interpolation. Values outside the calibrated Hue span are clipped to the
    nearest endpoint rather than extrapolated.
    """
    image, mask = _validate_image_and_mask(bgr, response_mask)
    hues, temps = _normalize_calibration_points(calibration_points)
    hue_deg, saturation, value = bgr_to_hsv_physical_channels(image)
    temperature = np.full(hue_deg.shape, np.nan, dtype=np.float32)
    if mask.any():
        active_hue = np.clip(hue_deg[mask], float(hues[0]), float(hues[-1]))
        temperature[mask] = np.interp(active_hue, hues, temps).astype(np.float32)
    return TLCSignalMapResult(
        mode="absolute_temperature_c",
        signal_map=temperature,
        active_mask=mask,
        hue_deg=hue_deg,
        saturation=saturation,
        value=value,
        provenance={
            "tlc_profile_id": tlc_profile_id,
            "mapping": "empirical_piecewise_linear",
            "calibration_point_count": int(hues.size),
            "calibrated_hue_min_deg": float(hues[0]),
            "calibrated_hue_max_deg": float(hues[-1]),
            "temperature_min_c": float(np.min(temps)),
            "temperature_max_c": float(np.max(temps)),
            "absolute_temperature": True,
        },
    )


def summarize_signal_map(result: TLCSignalMapResult) -> dict:
    """Summarize the active TLC signal without assigning biological meaning."""
    mask = np.asarray(result.active_mask, dtype=bool)
    values = np.asarray(result.signal_map, dtype=np.float32)[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mode": result.mode,
            "observable_fraction": 0.0,
            "count": 0,
            "mean": None,
            "std": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "range_p95_p05": None,
            "upper_quartile_fraction": None,
            "lower_quartile_fraction": None,
            "clinical_claim": "NONE",
        }
    p05, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "mode": result.mode,
        "observable_fraction": float(mask.mean()),
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(p05),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "range_p95_p05": float(p95 - p05),
        "upper_quartile_fraction": float(np.mean(values >= p75)),
        "lower_quartile_fraction": float(np.mean(values <= p25)),
        "clinical_claim": "NONE",
    }


def summarize_spatial_signal(result: TLCSignalMapResult) -> dict:
    """Return geometry-aware signal summaries for static TLC captures."""
    mask = np.asarray(result.active_mask, dtype=bool)
    signal = np.asarray(result.signal_map, dtype=np.float32)
    h, w = mask.shape
    if not mask.any():
        return {
            "left_mean": None,
            "right_mean": None,
            "left_minus_right": None,
            "top_mean": None,
            "bottom_mean": None,
            "top_minus_bottom": None,
            "center_mean": None,
            "periphery_mean": None,
            "center_minus_periphery": None,
            "clinical_claim": "NONE",
        }
    yy, xx = np.mgrid[:h, :w]
    left = mask & (xx < (w / 2.0))
    right = mask & ~left
    top = mask & (yy < (h / 2.0))
    bottom = mask & ~top
    cx0, cx1 = 0.25 * w, 0.75 * w
    cy0, cy1 = 0.25 * h, 0.75 * h
    center = mask & (xx >= cx0) & (xx < cx1) & (yy >= cy0) & (yy < cy1)
    periphery = mask & ~center

    def mean(region: np.ndarray) -> float | None:
        values = signal[region]
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else None

    left_mean = mean(left)
    right_mean = mean(right)
    top_mean = mean(top)
    bottom_mean = mean(bottom)
    center_mean = mean(center)
    periphery_mean = mean(periphery)

    def delta(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else float(a - b)

    return {
        "left_mean": left_mean,
        "right_mean": right_mean,
        "left_minus_right": delta(left_mean, right_mean),
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "top_minus_bottom": delta(top_mean, bottom_mean),
        "center_mean": center_mean,
        "periphery_mean": periphery_mean,
        "center_minus_periphery": delta(center_mean, periphery_mean),
        "clinical_claim": "NONE",
    }


def summarize_transient_sequence(
    signal_maps: Sequence[np.ndarray],
    active_masks: Sequence[np.ndarray],
    timestamps_s: Sequence[float],
) -> dict:
    """Deterministic temporal baseline for ordered TLC captures."""
    if len(signal_maps) != len(active_masks) or len(signal_maps) != len(timestamps_s):
        raise ValueError("signal_maps, active_masks, and timestamps_s must have equal length")
    if len(signal_maps) < 2:
        raise ValueError("Transient analysis requires at least two timepoints")
    t = np.asarray(timestamps_s, dtype=np.float64)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    means: list[float] = []
    for signal_map, active_mask in zip(signal_maps, active_masks):
        signal = np.asarray(signal_map, dtype=np.float32)
        mask = np.asarray(active_mask, dtype=bool)
        if signal.shape != mask.shape:
            raise ValueError("Each signal map and active mask must have matching shape")
        values = signal[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Each transient timepoint must contain observable signal")
        means.append(float(values.mean()))
    y = np.asarray(means, dtype=np.float64)
    slope = float(np.polyfit(t, y, deg=1)[0])
    peak_index = int(np.argmax(y))
    return {
        "timepoint_count": int(t.size),
        "start_mean": float(y[0]),
        "end_mean": float(y[-1]),
        "delta_mean": float(y[-1] - y[0]),
        "linear_rate_per_s": slope,
        "peak_mean": float(y[peak_index]),
        "time_to_peak_s": float(t[peak_index] - t[0]),
        "temporal_std": float(y.std()),
        "clinical_claim": "NONE",
    }
