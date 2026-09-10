from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from app.services.client_device_domain import (
    DEFAULT_CLIENT_CONFIG,
    letterbox_square,
    segment_client_response,
    summarize_response,
)
from app.services.client_device_qc import assess_client_device_quality
from app.services.tlc_profiles import normalize_plate_bgr, resolve_tlc_profile

CLIENT_TARGET_TLC_PROFILE_ID = "client-device-tlc-pending"
EMBEDDING_DIM = 384
FEATURE_CONTRACT_VERSION = "lct-target-v1"

SIGNAL_FEATURES = [
    "response_area_fraction",
    "component_count",
    "largest_component_fraction",
    "largest_eccentricity",
    "largest_solidity",
    "largest_perimeter_px",
    "largest_elongation",
    "skeleton_length_px",
    "branch_pixels",
    "endpoint_pixels",
    "response_centroid_x_norm",
    "response_centroid_y_norm",
    "hue_mean",
    "hue_std",
    "saturation_mean",
    "value_mean",
    "lab_a_mean",
    "lab_b_mean",
]

QC_FEATURES = [
    "valid_frame_fraction",
    "laplacian_variance",
    "gray_dynamic_range_p95_p5",
    "bright_clip_fraction",
    "specular_glare_fraction",
    "response_touches_valid_edge",
    "response_edge_fraction",
    "response_specular_overlap_fraction",
]

MORPHOLOGY_CATEGORIES = [
    "minimal-response",
    "scattered",
    "linear-like",
    "branched/complex",
    "diffuse/large-region",
    "focal/irregular-region",
    "mixed",
]


@dataclass(frozen=True)
class TargetFeatureResult:
    vector: np.ndarray
    morphology_descriptor: str
    qc_status: str
    qc_flags: tuple[str, ...]
    signal_features: dict
    qc_metrics: dict
    normalized_bgr: np.ndarray
    response_mask: np.ndarray


def feature_schema() -> dict:
    signal_start = 0
    qc_start = signal_start + len(SIGNAL_FEATURES)
    morphology_start = qc_start + len(QC_FEATURES)
    dino_start = morphology_start + len(MORPHOLOGY_CATEGORIES)
    return {
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "total_features": dino_start + EMBEDDING_DIM,
        "groups": {
            "signal_features": {
                "start": signal_start,
                "count": len(SIGNAL_FEATURES),
                "names": SIGNAL_FEATURES,
            },
            "qc_features": {
                "start": qc_start,
                "count": len(QC_FEATURES),
                "names": QC_FEATURES,
            },
            "morphology_one_hot": {
                "start": morphology_start,
                "count": len(MORPHOLOGY_CATEGORIES),
                "names": MORPHOLOGY_CATEGORIES,
            },
            "dinov2": {
                "start": dino_start,
                "count": EMBEDDING_DIM,
                "backbone": "dinov2_vits14",
            },
        },
        "semantics": (
            "Profile-specific contact-LCT target features: visible-response morphology, "
            "engineering QC, morphology one-hot encoding, and normalized DINOv2 embedding. "
            "Colour values are not absolute temperature without formulation calibration."
        ),
        "clinical_claim": "NONE",
    }


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"DINOv2 embedding must have shape ({EMBEDDING_DIM},), received {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ValueError("DINOv2 embedding must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("DINOv2 embedding must have non-zero norm")
    return vector / norm


def _live_dino_encoder(rgb: np.ndarray) -> np.ndarray:
    from app.services import dinov2_service

    return dinov2_service.runtime.encode_rgb(rgb)


def build_target_feature_vector(
    bgr: np.ndarray,
    tlc_profile_id: str,
    *,
    embedding: np.ndarray | None = None,
    encoder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> TargetFeatureResult:
    """Build the first client-device target-domain feature contract.

    This version is intentionally restricted to the current client TLC profile.
    A future calibrated formulation must get its own selector/profile dispatch
    rather than silently reusing this provisional client-device threshold set.
    """

    image = np.asarray(bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with 3 channels")

    profile = resolve_tlc_profile(tlc_profile_id)
    if profile.id != CLIENT_TARGET_TLC_PROFILE_ID:
        raise ValueError(
            "lct-target-v1 currently supports only the client-device TLC profile; "
            f"received {profile.id}"
        )

    normalized, valid = letterbox_square(image)
    normalized = normalize_plate_bgr(normalized, profile)
    response_mask, _ = segment_client_response(
        normalized, valid, DEFAULT_CLIENT_CONFIG
    )
    signal = summarize_response(normalized, response_mask, valid)
    morphology = str(signal["morphology_descriptor"])
    if morphology not in MORPHOLOGY_CATEGORIES:
        raise ValueError(f"Unsupported morphology descriptor: {morphology}")

    qc = assess_client_device_quality(normalized, response_mask, valid)
    qc_metrics = dict(qc["metrics"])

    if embedding is None:
        selected_encoder = encoder or _live_dino_encoder
        embedding = selected_encoder(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    dino = _normalize_embedding(embedding)

    values: list[float] = []
    values.extend(float(signal[name]) for name in SIGNAL_FEATURES)
    for name in QC_FEATURES:
        value = qc_metrics[name]
        values.append(float(bool(value)) if isinstance(value, (bool, np.bool_)) else float(value))
    values.extend(1.0 if morphology == category else 0.0 for category in MORPHOLOGY_CATEGORIES)
    values.extend(float(value) for value in dino)

    vector = np.asarray(values, dtype=np.float32)
    expected = int(feature_schema()["total_features"])
    if vector.shape != (expected,) or not np.isfinite(vector).all():
        raise RuntimeError(
            f"Invalid target feature vector: expected {expected} finite values, got {vector.shape}"
        )

    return TargetFeatureResult(
        vector=vector,
        morphology_descriptor=morphology,
        qc_status=str(qc["status"]),
        qc_flags=tuple(str(flag) for flag in qc["flags"]),
        signal_features={k: v for k, v in signal.items() if k != "morphology_descriptor"},
        qc_metrics=qc_metrics,
        normalized_bgr=normalized,
        response_mask=response_mask,
    )


def feature_columns() -> list[str]:
    return [f"feature_{index:03d}" for index in range(int(feature_schema()["total_features"]))]
