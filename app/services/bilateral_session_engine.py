from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from app.services.thermal_anomaly_engine import (
    ThermalAnomalyResult,
    build_multiscale_thermal_anomaly,
)


@dataclass(frozen=True)
class FrameRegistration:
    dx_px: float
    dy_px: float
    method: str
    match_count: int
    confidence: float


@dataclass(frozen=True)
class SideFieldResult:
    signal_map: np.ndarray
    observable_mask: np.ndarray
    coverage_count: np.ndarray
    frame_offsets_xy: tuple[tuple[float, float], ...]
    provenance: dict


@dataclass(frozen=True)
class BilateralResult:
    asymmetry_map: np.ndarray
    signed_difference_map: np.ndarray
    joint_mask: np.ndarray
    score: float
    features: dict
    provenance: dict


@dataclass(frozen=True)
class SessionEvidenceResult:
    left_field: SideFieldResult
    right_field: SideFieldResult
    left_anomaly: ThermalAnomalyResult
    right_anomaly: ThermalAnomalyResult
    bilateral: BilateralResult
    scores: dict
    status: str
    provenance: dict


def _as_bgr(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")
    return array


def _registration_gray(image: np.ndarray) -> np.ndarray:
    bgr = _as_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def estimate_translation_from_overlap(
    anchor_bgr: np.ndarray,
    moving_bgr: np.ndarray,
    *,
    max_features: int = 1500,
    min_matches: int = 8,
) -> FrameRegistration:
    """Estimate translation that maps ``moving`` into ``anchor`` coordinates.

    ORB feature displacement is used first because successive MumGuard captures
    may only partially overlap. Phase correlation is retained as a deterministic
    fallback when feature support is weak.
    """
    anchor = _registration_gray(anchor_bgr)
    moving = _registration_gray(moving_bgr)
    if anchor.shape != moving.shape:
        moving = cv2.resize(moving, (anchor.shape[1], anchor.shape[0]), interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(nfeatures=int(max_features), fastThreshold=8)
    key_a, des_a = orb.detectAndCompute(anchor, None)
    key_b, des_b = orb.detectAndCompute(moving, None)
    if des_a is not None and des_b is not None and len(key_a) >= 4 and len(key_b) >= 4:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(des_b, des_a), key=lambda item: item.distance)
        good = matches[: min(len(matches), 200)]
        if len(good) >= int(min_matches):
            deltas = []
            for match in good:
                bx, by = key_b[match.queryIdx].pt
                ax, ay = key_a[match.trainIdx].pt
                deltas.append((ax - bx, ay - by))
            delta = np.asarray(deltas, dtype=np.float32)
            center = np.median(delta, axis=0)
            residual = np.linalg.norm(delta - center, axis=1)
            mad = float(np.median(np.abs(residual - np.median(residual))))
            inlier_limit = max(2.5, 3.0 * 1.4826 * mad)
            inliers = residual <= inlier_limit
            if int(inliers.sum()) >= int(min_matches):
                robust = np.median(delta[inliers], axis=0)
                confidence = float(min(1.0, inliers.sum() / max(len(good), 1)))
                return FrameRegistration(
                    dx_px=float(robust[0]),
                    dy_px=float(robust[1]),
                    method="orb_median_translation",
                    match_count=int(inliers.sum()),
                    confidence=confidence,
                )

    shift, response = cv2.phaseCorrelate(
        anchor.astype(np.float32), moving.astype(np.float32)
    )
    return FrameRegistration(
        dx_px=float(shift[0]),
        dy_px=float(shift[1]),
        method="phase_correlation_fallback",
        match_count=0,
        confidence=float(np.clip(response, 0.0, 1.0)),
    )


def infer_sequential_offsets(images_bgr: Sequence[np.ndarray]) -> tuple[tuple[float, float], ...]:
    """Infer cumulative frame positions for a sequential overlapping side scan."""
    if not images_bgr:
        raise ValueError("At least one frame is required")
    offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    for index in range(1, len(images_bgr)):
        registration = estimate_translation_from_overlap(
            images_bgr[index - 1], images_bgr[index]
        )
        prev_x, prev_y = offsets[-1]
        offsets.append((prev_x + registration.dx_px, prev_y + registration.dy_px))
    return tuple(offsets)


def compose_side_signal(
    signal_maps: Sequence[np.ndarray],
    observable_masks: Sequence[np.ndarray],
    offsets_xy: Sequence[tuple[float, float]],
    *,
    frame_weights: Sequence[float] | None = None,
) -> SideFieldResult:
    """Compose overlapping captures into one de-duplicated side field.

    Overlap is averaged once on the common canvas and ``coverage_count`` records
    how many source captures support each pixel, preventing repeated tissue from
    being counted as independent evidence downstream.
    """
    if not signal_maps:
        raise ValueError("At least one signal map is required")
    if len(signal_maps) != len(observable_masks) or len(signal_maps) != len(offsets_xy):
        raise ValueError("signal_maps, observable_masks, and offsets_xy must have equal length")
    if frame_weights is None:
        weights = [1.0] * len(signal_maps)
    else:
        if len(frame_weights) != len(signal_maps):
            raise ValueError("frame_weights must match signal_maps length")
        weights = [float(value) for value in frame_weights]
        if any((not np.isfinite(value)) or value <= 0.0 for value in weights):
            raise ValueError("frame_weights must be positive finite values")

    shapes = [np.asarray(signal).shape for signal in signal_maps]
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError("Every signal map must be 2-D")
    for signal, mask in zip(signal_maps, observable_masks):
        if np.asarray(signal).shape != np.asarray(mask).shape:
            raise ValueError("Each signal map and observable mask must have matching shape")

    min_x = min(float(offset[0]) for offset in offsets_xy)
    min_y = min(float(offset[1]) for offset in offsets_xy)
    max_x = max(float(offset[0]) + shape[1] for offset, shape in zip(offsets_xy, shapes))
    max_y = max(float(offset[1]) + shape[0] for offset, shape in zip(offsets_xy, shapes))
    width = int(np.ceil(max_x - min_x))
    height = int(np.ceil(max_y - min_y))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid composite canvas size")

    weighted_sum = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    coverage = np.zeros((height, width), dtype=np.uint16)

    for signal_raw, mask_raw, offset, weight in zip(signal_maps, observable_masks, offsets_xy, weights):
        signal = np.asarray(signal_raw, dtype=np.float32)
        mask = np.asarray(mask_raw, dtype=bool) & np.isfinite(signal)
        x0 = int(round(float(offset[0]) - min_x))
        y0 = int(round(float(offset[1]) - min_y))
        h, w = signal.shape
        target_sum = weighted_sum[y0 : y0 + h, x0 : x0 + w]
        target_weight = weight_sum[y0 : y0 + h, x0 : x0 + w]
        target_coverage = coverage[y0 : y0 + h, x0 : x0 + w]
        target_sum[mask] += signal[mask] * weight
        target_weight[mask] += weight
        target_coverage[mask] += 1

    observed = weight_sum > 0.0
    composite = np.full((height, width), np.nan, dtype=np.float32)
    composite[observed] = (weighted_sum[observed] / weight_sum[observed]).astype(np.float32)
    overlap_pixels = int(np.sum(coverage > 1))
    observed_pixels = int(np.sum(observed))

    return SideFieldResult(
        signal_map=composite,
        observable_mask=observed,
        coverage_count=coverage,
        frame_offsets_xy=tuple((float(x), float(y)) for x, y in offsets_xy),
        provenance={
            "algorithm": "translated_weighted_overlap_composition_v1",
            "frame_count": len(signal_maps),
            "observed_pixels": observed_pixels,
            "overlap_pixels": overlap_pixels,
            "overlap_fraction_of_observed": float(overlap_pixels / max(observed_pixels, 1)),
            "deduplicates_overlap": True,
        },
    )


def _robust_standardize(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = signal[mask]
    values = values[np.isfinite(values)]
    result = np.full(signal.shape, np.nan, dtype=np.float32)
    if values.size == 0:
        return result
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = max(1.4826 * mad, float(np.std(values)) * 0.1, 1e-6)
    result[mask] = (signal[mask] - median) / sigma
    return result


def _normalized_field(field: SideFieldResult, output_size: int, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(field.signal_map, dtype=np.float32)
    mask = np.asarray(field.observable_mask, dtype=np.uint8)
    finite = np.nan_to_num(signal, nan=0.0).astype(np.float32)
    target = (int(output_size), int(output_size))
    signal_resized = cv2.resize(finite, target, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, target, interpolation=cv2.INTER_NEAREST) > 0
    if mirror:
        signal_resized = np.fliplr(signal_resized).copy()
        mask_resized = np.fliplr(mask_resized).copy()
    signal_resized[~mask_resized] = np.nan
    return signal_resized, mask_resized


def analyze_bilateral_asymmetry(
    left: SideFieldResult,
    right: SideFieldResult,
    *,
    output_size: int = 256,
    mirror_right: bool = True,
    min_joint_fraction: float = 0.05,
) -> BilateralResult:
    """Compare left and right breasts after side-level reconstruction.

    Each side is robustly standardized before comparison so the asymmetry channel
    measures distribution/behaviour disagreement rather than a global colour or
    exposure offset. Right-side mirroring makes homologous lateral anatomy face
    the same normalized coordinate system.
    """
    if int(output_size) < 32:
        raise ValueError("output_size must be at least 32")
    left_signal, left_mask = _normalized_field(left, output_size, False)
    right_signal, right_mask = _normalized_field(right, output_size, bool(mirror_right))
    joint = left_mask & right_mask & np.isfinite(left_signal) & np.isfinite(right_signal)
    joint_fraction = float(joint.mean())
    if joint_fraction < float(min_joint_fraction):
        asymmetry = np.full(left_signal.shape, np.nan, dtype=np.float32)
        signed = np.full(left_signal.shape, np.nan, dtype=np.float32)
        return BilateralResult(
            asymmetry_map=asymmetry,
            signed_difference_map=signed,
            joint_mask=joint,
            score=0.0,
            features={
                "joint_fraction": joint_fraction,
                "asymmetry_p90": 0.0,
                "asymmetry_p95": 0.0,
                "asymmetry_p99": 0.0,
                "high_asymmetry_fraction": 0.0,
                "status": "INSUFFICIENT_BILATERAL_OVERLAP",
                "clinical_claim": "NONE",
            },
            provenance={
                "algorithm": "robust_bilateral_distribution_difference_v1",
                "mirror_right": bool(mirror_right),
                "species_specific": False,
            },
        )

    left_z = _robust_standardize(left_signal, left_mask)
    right_z = _robust_standardize(right_signal, right_mask)
    signed = np.full(left_signal.shape, np.nan, dtype=np.float32)
    signed[joint] = left_z[joint] - right_z[joint]
    asymmetry = np.full(left_signal.shape, np.nan, dtype=np.float32)
    asymmetry[joint] = np.abs(signed[joint])
    values = asymmetry[joint]
    p90, p95, p99 = np.percentile(values, [90, 95, 99])
    score = float(np.mean(np.sort(values)[-max(1, values.size // 10) :]))
    high_fraction = float(np.mean(values >= p95))
    return BilateralResult(
        asymmetry_map=asymmetry,
        signed_difference_map=signed,
        joint_mask=joint,
        score=score,
        features={
            "joint_fraction": joint_fraction,
            "asymmetry_p90": float(p90),
            "asymmetry_p95": float(p95),
            "asymmetry_p99": float(p99),
            "high_asymmetry_fraction": high_fraction,
            "status": "OK",
            "clinical_claim": "NONE",
        },
        provenance={
            "algorithm": "robust_bilateral_distribution_difference_v1",
            "mirror_right": bool(mirror_right),
            "species_specific": False,
            "uses_patient_internal_control": True,
        },
    )


def _positive_core_score(anomaly: ThermalAnomalyResult, hotter_is_higher: bool) -> float:
    mask = np.asarray(anomaly.observable_mask, dtype=bool)
    evidence = np.asarray(anomaly.evidence_map, dtype=np.float32)
    signed = np.asarray(anomaly.signed_contrast_map, dtype=np.float32)
    valid = mask & np.isfinite(evidence) & np.isfinite(signed)
    if not valid.any():
        return 0.0
    direction = signed if hotter_is_higher else -signed
    positive = valid & (direction > 0.0)
    if not positive.any():
        return 0.0
    strength = evidence[positive] * np.clip(direction[positive] / max(float(np.std(direction[valid])), 1e-6), 0.0, 4.0) / 4.0
    return float(np.mean(np.sort(strength)[-max(1, strength.size // 10) :]))


def _abnormal_pattern_score(anomaly: ThermalAnomalyResult) -> float:
    features = anomaly.features
    return float(
        np.clip(
            0.55 * float(features.get("global_anomaly_score", 0.0))
            + 0.25 * float(features.get("mean_persistence", 0.0))
            + 0.20 * min(1.0, 10.0 * float(features.get("high_evidence_fraction", 0.0))),
            0.0,
            1.0,
        )
    )


def build_three_channel_session_evidence(
    left: SideFieldResult,
    right: SideFieldResult,
    *,
    hotter_is_higher: bool = True,
    anomaly_radii: Sequence[int] = (3, 7, 15),
    minimum_side_observable_fraction: float = 0.03,
) -> SessionEvidenceResult:
    """Build MumGuard's three technique channels from one bilateral session.

    Channels:
    1. focal/core hyperthermia-like local elevation;
    2. right-versus-left thermal asymmetry;
    3. abnormal spatial thermal behaviour.

    The engine is label-free and human-first: mouse labels are not used anywhere
    in the calculation and can be reserved for downstream experiment scoring.
    """
    left_anomaly = build_multiscale_thermal_anomaly(
        left.signal_map,
        left.observable_mask,
        radii=anomaly_radii,
        signal_mode="session_side_field",
    )
    right_anomaly = build_multiscale_thermal_anomaly(
        right.signal_map,
        right.observable_mask,
        radii=anomaly_radii,
        signal_mode="session_side_field",
    )
    bilateral = analyze_bilateral_asymmetry(left, right)

    left_core = _positive_core_score(left_anomaly, hotter_is_higher)
    right_core = _positive_core_score(right_anomaly, hotter_is_higher)
    hyperthermia_score = max(left_core, right_core)
    pattern_score = max(
        _abnormal_pattern_score(left_anomaly),
        _abnormal_pattern_score(right_anomaly),
    )
    bilateral_score = float(np.clip(bilateral.score / 4.0, 0.0, 1.0))

    left_observable = float(np.asarray(left.observable_mask, dtype=bool).mean())
    right_observable = float(np.asarray(right.observable_mask, dtype=bool).mean())
    bilateral_ok = bilateral.features.get("status") == "OK"
    side_ok = min(left_observable, right_observable) >= float(minimum_side_observable_fraction)
    status = "OK" if side_ok and bilateral_ok else "INCONCLUSIVE_MEASUREMENT_SUPPORT"

    available_weights = [0.38, 0.34, 0.28]
    raw_scores = [hyperthermia_score, bilateral_score, pattern_score]
    overall = float(np.average(raw_scores, weights=available_weights))

    return SessionEvidenceResult(
        left_field=left,
        right_field=right,
        left_anomaly=left_anomaly,
        right_anomaly=right_anomaly,
        bilateral=bilateral,
        scores={
            "core_hyperthermia_score": hyperthermia_score,
            "left_core_hyperthermia_score": left_core,
            "right_core_hyperthermia_score": right_core,
            "bilateral_asymmetry_score": bilateral_score,
            "abnormal_skin_behavior_score": pattern_score,
            "overall_measurement_evidence_score": overall,
            "clinical_claim": "NONE",
        },
        status=status,
        provenance={
            "algorithm": "mumguard_three_channel_session_evidence_v1",
            "human_first": True,
            "label_free": True,
            "species_specific": False,
            "channels": [
                "core_hyperthermia",
                "bilateral_asymmetry",
                "abnormal_skin_thermal_behavior",
            ],
            "overlap_deduplication_expected_upstream": True,
        },
    )
