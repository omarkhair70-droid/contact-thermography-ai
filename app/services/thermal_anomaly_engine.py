from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ThermalAnomalyResult:
    evidence_map: np.ndarray
    signed_contrast_map: np.ndarray
    persistence_map: np.ndarray
    observable_mask: np.ndarray
    features: dict
    provenance: dict


def _validate_signal(
    signal_map: np.ndarray,
    observable_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal_map, dtype=np.float32)
    mask = np.asarray(observable_mask, dtype=bool)
    if signal.ndim != 2:
        raise ValueError("signal_map must be a 2-D array")
    if mask.shape != signal.shape:
        raise ValueError("observable_mask must match signal_map shape")
    mask = mask & np.isfinite(signal)
    return signal, mask


def _masked_box_sum(values: np.ndarray, mask: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    if radius < 1:
        raise ValueError("radius must be >= 1")
    kernel = 2 * int(radius) + 1
    weighted = np.where(mask, values, 0.0).astype(np.float32)
    counts = mask.astype(np.float32)
    value_sum = cv2.boxFilter(
        weighted,
        ddepth=cv2.CV_32F,
        ksize=(kernel, kernel),
        normalize=False,
        borderType=cv2.BORDER_REFLECT,
    )
    count_sum = cv2.boxFilter(
        counts,
        ddepth=cv2.CV_32F,
        ksize=(kernel, kernel),
        normalize=False,
        borderType=cv2.BORDER_REFLECT,
    )
    return value_sum, count_sum


def _ring_reference(
    signal: np.ndarray,
    mask: np.ndarray,
    inner_radius: int,
    outer_radius: int,
    min_ring_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be greater than inner_radius")
    outer_sum, outer_count = _masked_box_sum(signal, mask, outer_radius)
    inner_sum, inner_count = _masked_box_sum(signal, mask, inner_radius)
    ring_sum = outer_sum - inner_sum
    ring_count = outer_count - inner_count
    usable = mask & (ring_count >= float(min_ring_pixels))
    reference = np.full(signal.shape, np.nan, dtype=np.float32)
    reference[usable] = ring_sum[usable] / np.maximum(ring_count[usable], 1.0)
    return reference, usable


def _robust_signed_z(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = values[mask]
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return np.full(values.shape, np.nan, dtype=np.float32), 0.0, 1.0
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    robust_sigma = max(1.4826 * mad, float(np.std(valid)) * 0.1, 1e-6)
    z = np.full(values.shape, np.nan, dtype=np.float32)
    z[mask] = (values[mask] - median) / robust_sigma
    return z, median, robust_sigma


def build_multiscale_thermal_anomaly(
    signal_map: np.ndarray,
    observable_mask: np.ndarray,
    *,
    radii: Sequence[int] = (3, 7, 15),
    outer_multiplier: float = 2.0,
    z_clip: float = 6.0,
    persistence_z: float = 1.5,
    min_ring_pixels: int = 12,
    signal_mode: str = "relative_or_temperature",
) -> ThermalAnomalyResult:
    """Compare every observable point with disjoint surrounding tissue at multiple scales.

    This is a deterministic, label-free measurement algorithm. It does not learn
    cancer appearance. For every scale it compares the local signal with an
    annulus outside the local neighborhood, robustly normalizes the signed
    contrast, then aggregates magnitude and cross-scale persistence into an
    evidence map.
    """
    signal, base_mask = _validate_signal(signal_map, observable_mask)
    clean_radii = tuple(sorted({int(r) for r in radii if int(r) >= 1}))
    if not clean_radii:
        raise ValueError("At least one positive radius is required")

    z_maps: list[np.ndarray] = []
    contrast_maps: list[np.ndarray] = []
    usable_masks: list[np.ndarray] = []
    scale_metadata: list[dict] = []

    for inner_radius in clean_radii:
        outer_radius = max(inner_radius + 1, int(round(inner_radius * outer_multiplier)))
        reference, usable = _ring_reference(
            signal,
            base_mask,
            inner_radius,
            outer_radius,
            min_ring_pixels,
        )
        contrast = np.full(signal.shape, np.nan, dtype=np.float32)
        contrast[usable] = signal[usable] - reference[usable]
        z, median, robust_sigma = _robust_signed_z(contrast, usable)
        z_maps.append(z)
        contrast_maps.append(contrast)
        usable_masks.append(usable)
        scale_metadata.append(
            {
                "inner_radius_px": inner_radius,
                "outer_radius_px": outer_radius,
                "usable_fraction": float(usable.mean()),
                "contrast_median": median,
                "contrast_robust_sigma": robust_sigma,
            }
        )

    z_stack = np.stack(z_maps, axis=0)
    contrast_stack = np.stack(contrast_maps, axis=0)
    usable_stack = np.stack(usable_masks, axis=0)
    usable_count = usable_stack.sum(axis=0)
    aggregate_mask = base_mask & (usable_count > 0)

    clipped_abs_z = np.where(
        usable_stack,
        np.minimum(np.abs(z_stack), float(z_clip)) / float(z_clip),
        np.nan,
    )
    magnitude = np.full(signal.shape, np.nan, dtype=np.float32)
    signed_contrast = np.full(signal.shape, np.nan, dtype=np.float32)
    if aggregate_mask.any():
        # Every selected column has at least one usable scale, avoiding all-NaN
        # reductions outside the observable/comparable field.
        magnitude[aggregate_mask] = np.nanmedian(
            clipped_abs_z[:, aggregate_mask], axis=0
        ).astype(np.float32)
        signed_contrast[aggregate_mask] = np.nanmedian(
            contrast_stack[:, aggregate_mask], axis=0
        ).astype(np.float32)

    persistence = np.zeros(signal.shape, dtype=np.float32)
    if clean_radii:
        persistence[aggregate_mask] = (
            (usable_stack & (np.abs(z_stack) >= float(persistence_z))).sum(axis=0)[aggregate_mask]
            / usable_count[aggregate_mask]
        ).astype(np.float32)

    evidence = np.full(signal.shape, np.nan, dtype=np.float32)
    evidence[aggregate_mask] = (
        0.72 * np.nan_to_num(magnitude[aggregate_mask], nan=0.0)
        + 0.28 * persistence[aggregate_mask]
    )
    evidence[aggregate_mask] = np.clip(evidence[aggregate_mask], 0.0, 1.0)

    finite_evidence = evidence[aggregate_mask]
    finite_evidence = finite_evidence[np.isfinite(finite_evidence)]
    finite_signed = signed_contrast[aggregate_mask]
    finite_signed = finite_signed[np.isfinite(finite_signed)]

    if finite_evidence.size:
        p90, p95, p99 = np.percentile(finite_evidence, [90, 95, 99])
        top_threshold = float(p95)
        high_fraction = float(np.mean(finite_evidence >= top_threshold))
        global_score = float(np.mean(np.sort(finite_evidence)[-max(1, finite_evidence.size // 20) :]))
    else:
        p90 = p95 = p99 = 0.0
        high_fraction = 0.0
        global_score = 0.0

    if finite_signed.size:
        signed_p05, signed_p50, signed_p95 = np.percentile(finite_signed, [5, 50, 95])
    else:
        signed_p05 = signed_p50 = signed_p95 = 0.0

    features = {
        "observable_fraction": float(base_mask.mean()),
        "comparable_fraction": float(aggregate_mask.mean()),
        "scale_count": len(clean_radii),
        "evidence_p90": float(p90),
        "evidence_p95": float(p95),
        "evidence_p99": float(p99),
        "high_evidence_fraction": high_fraction,
        "global_anomaly_score": global_score,
        "signed_contrast_p05": float(signed_p05),
        "signed_contrast_p50": float(signed_p50),
        "signed_contrast_p95": float(signed_p95),
        "mean_persistence": float(persistence[aggregate_mask].mean()) if aggregate_mask.any() else 0.0,
        "clinical_claim": "NONE",
    }

    return ThermalAnomalyResult(
        evidence_map=evidence,
        signed_contrast_map=signed_contrast.astype(np.float32),
        persistence_map=persistence,
        observable_mask=aggregate_mask,
        features=features,
        provenance={
            "algorithm": "multiscale_ring_local_contrast_v1",
            "signal_mode": signal_mode,
            "radii_px": list(clean_radii),
            "outer_multiplier": float(outer_multiplier),
            "persistence_z": float(persistence_z),
            "z_clip": float(z_clip),
            "label_free": True,
            "species_specific": False,
            "scales": scale_metadata,
        },
    )


def fuse_evidence_maps(
    thermal_evidence: np.ndarray,
    observable_mask: np.ndarray,
    *,
    visual_evidence: np.ndarray | None = None,
    thermal_weight: float = 0.7,
) -> np.ndarray:
    """Fuse deterministic thermal evidence with an optional visual/AI evidence map.

    The optional visual map is deliberately an adapter boundary so DINO patch
    evidence (or another representation model) can be merged without coupling
    this physics/signal module to a specific backbone.
    """
    thermal = np.asarray(thermal_evidence, dtype=np.float32)
    mask = np.asarray(observable_mask, dtype=bool)
    if thermal.shape != mask.shape:
        raise ValueError("thermal_evidence and observable_mask must have matching shape")
    if visual_evidence is None:
        result = np.full(thermal.shape, np.nan, dtype=np.float32)
        result[mask] = np.clip(np.nan_to_num(thermal[mask], nan=0.0), 0.0, 1.0)
        return result

    visual = np.asarray(visual_evidence, dtype=np.float32)
    if visual.shape != thermal.shape:
        raise ValueError("visual_evidence must match thermal_evidence shape")
    weight = float(thermal_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("thermal_weight must be within [0, 1]")
    joint = mask & np.isfinite(thermal) & np.isfinite(visual)
    result = np.full(thermal.shape, np.nan, dtype=np.float32)
    result[joint] = np.clip(
        weight * thermal[joint] + (1.0 - weight) * visual[joint],
        0.0,
        1.0,
    )
    return result
