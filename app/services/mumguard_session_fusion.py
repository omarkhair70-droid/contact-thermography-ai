from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from app.services.bilateral_session_engine import (
    SideFieldResult,
    build_three_channel_session_evidence,
    compose_side_signal,
    infer_sequential_offsets,
)
from app.services.client_device_domain import segment_client_response
from app.services.contact_field import normalize_field
from app.services.thermal_anomaly_engine import fuse_evidence_maps
from app.services.tlc_signal_processing import build_relative_thermal_map


@dataclass(frozen=True)
class SessionFrameInput:
    source: bytes
    source_name: str
    side: str
    sequence_index: int


def _normalized_bgr(source: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    rgb, _, _, frame, transform = normalize_field(source, size=224)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, frame.astype(bool), transform


def _patch_support(mask: np.ndarray, token_shape: tuple[int, int]) -> np.ndarray:
    th, tw = token_shape
    h, w = mask.shape
    support = np.zeros((th, tw), dtype=bool)
    for ty in range(th):
        y0, y1 = round(ty * h / th), round((ty + 1) * h / th)
        for tx in range(tw):
            x0, x1 = round(tx * w / tw), round((tx + 1) * w / tw)
            region = mask[y0:y1, x0:x1]
            support[ty, tx] = bool(region.size and region.mean() >= 0.5)
    return support


def _dino_novelty_map(rgb: np.ndarray, response_mask: np.ndarray) -> tuple[np.ndarray | None, dict | None]:
    """Return label-free patch novelty relative to the same frame's response field."""
    try:
        from app.services.dinov2_patch_encoder import encode_patches
        from app.services.dinov2_service import DINOv2UnavailableError
        tokens, provenance = encode_patches(rgb)
    except DINOv2UnavailableError as exc:
        return None, {"status": "DINO_UNAVAILABLE", "reason": str(exc)}

    th, tw, dim = tokens.shape
    support = _patch_support(response_mask, (th, tw))
    flat = tokens.reshape(-1, dim)
    selected = flat[support.reshape(-1)]
    if selected.shape[0] < 4:
        return None, {**provenance, "status": "INSUFFICIENT_RESPONSE_PATCHES"}

    center = selected.mean(axis=0)
    norm = float(np.linalg.norm(center))
    if norm <= 1e-8:
        return None, {**provenance, "status": "DEGENERATE_REFERENCE"}
    center /= norm
    distance = 1.0 - np.clip(flat @ center, -1.0, 1.0)
    active_values = distance[support.reshape(-1)]
    lo, hi = np.percentile(active_values, [50, 95])
    denom = max(float(hi - lo), 1e-6)
    novelty = np.clip((distance.reshape(th, tw) - float(lo)) / denom, 0.0, 1.0)
    up = cv2.resize(novelty.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    up[~response_mask] = np.nan
    return up, {**provenance, "status": "OK", "reference": "same_frame_response_mean"}


def _prepare_frame(frame: SessionFrameInput, tlc_profile_id: str, include_dino: bool) -> dict:
    side = str(frame.side).upper()
    if side not in {"LEFT", "RIGHT"}:
        raise ValueError("Each session frame side must be LEFT or RIGHT")
    bgr, valid, transform = _normalized_bgr(frame.source)
    response_mask_u8, components = segment_client_response(bgr, valid)
    response_mask = response_mask_u8 > 0
    signal = build_relative_thermal_map(
        bgr,
        response_mask,
        tlc_profile_id=tlc_profile_id,
    )
    visual_map = visual_provenance = None
    if include_dino and response_mask.any():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        visual_map, visual_provenance = _dino_novelty_map(rgb, response_mask)
    return {
        "input": frame,
        "bgr": bgr,
        "response_mask": response_mask,
        "signal": signal,
        "visual_map": visual_map,
        "visual_provenance": visual_provenance,
        "component_count": len(components),
        "transform": transform,
    }


def _build_side(prepared: Sequence[dict]) -> tuple[SideFieldResult, SideFieldResult | None, tuple[tuple[float, float], ...]]:
    if not prepared:
        raise ValueError("Each bilateral session requires at least one frame per side")
    ordered = sorted(prepared, key=lambda item: (int(item["input"].sequence_index), item["input"].source_name))
    offsets = infer_sequential_offsets([item["bgr"] for item in ordered])
    signal_field = compose_side_signal(
        [item["signal"].signal_map for item in ordered],
        [item["signal"].active_mask for item in ordered],
        offsets,
    )
    visual_field = None
    if all(item["visual_map"] is not None for item in ordered):
        visual_field = compose_side_signal(
            [item["visual_map"] for item in ordered],
            [np.isfinite(item["visual_map"]) for item in ordered],
            offsets,
        )
    return signal_field, visual_field, offsets


def _serialize_side(field: SideFieldResult) -> dict:
    return {
        "shape": list(field.signal_map.shape),
        "observable_fraction": float(field.observable_mask.mean()),
        "max_coverage": int(field.coverage_count.max()) if field.coverage_count.size else 0,
        "frame_offsets_xy": [list(item) for item in field.frame_offsets_xy],
        "provenance": dict(field.provenance),
    }


def analyze_bilateral_session(
    frames: Sequence[SessionFrameInput],
    *,
    tlc_profile_id: str = "client-device-tlc-pending",
    hotter_is_higher: bool = True,
    include_dino: bool = True,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run the unified human-first MumGuard measurement/evidence pipeline.

    Independent contact annotations remain useful provenance but are not required
    to execute the TLC response measurement channel. The response mask here means
    visible, high-confidence TLC response only; it is never relabeled as confirmed
    tissue contact or healthy tissue.
    """
    if not frames:
        raise ValueError("At least one session frame is required")
    prepared = [_prepare_frame(frame, tlc_profile_id, include_dino) for frame in frames]
    left = [item for item in prepared if item["input"].side.upper() == "LEFT"]
    right = [item for item in prepared if item["input"].side.upper() == "RIGHT"]
    if not left or not right:
        raise ValueError("A bilateral session requires LEFT and RIGHT frames")

    left_field, left_visual, left_offsets = _build_side(left)
    right_field, right_visual, right_offsets = _build_side(right)
    session = build_three_channel_session_evidence(
        left_field,
        right_field,
        hotter_is_higher=hotter_is_higher,
    )

    left_fused = fuse_evidence_maps(
        session.left_anomaly.evidence_map,
        session.left_anomaly.observable_mask,
        visual_evidence=left_visual.signal_map if left_visual is not None else None,
    )
    right_fused = fuse_evidence_maps(
        session.right_anomaly.evidence_map,
        session.right_anomaly.observable_mask,
        visual_evidence=right_visual.signal_map if right_visual is not None else None,
    )

    result = {
        "status": session.status,
        "architecture": "mumguard_session_fusion_v1",
        "target_species": "human",
        "measurement_mode": "relative_tlc_signal",
        "tlc_profile_id": tlc_profile_id,
        "hotter_is_higher_working_assumption": bool(hotter_is_higher),
        "contact_annotation_required_for_measurement": False,
        "response_support_semantics": "PROVISIONAL_VISIBLE_TLC_RESPONSE_NOT_CONFIRMED_TISSUE_CONTACT",
        "three_channel_scores": dict(session.scores),
        "left": _serialize_side(left_field),
        "right": _serialize_side(right_field),
        "bilateral": {
            "score": float(session.bilateral.score),
            "features": dict(session.bilateral.features),
            "provenance": dict(session.bilateral.provenance),
        },
        "frames": [
            {
                "source_name": item["input"].source_name,
                "side": item["input"].side.upper(),
                "sequence_index": int(item["input"].sequence_index),
                "response_fraction": float(item["response_mask"].mean()),
                "selected_response_components": int(item["component_count"]),
                "signal_provenance": dict(item["signal"].provenance),
                "visual_provenance": item["visual_provenance"],
                "geometry": dict(item["transform"]),
            }
            for item in prepared
        ],
        "left_offsets_xy": [list(value) for value in left_offsets],
        "right_offsets_xy": [list(value) for value in right_offsets],
        "ai_evidence_available": left_visual is not None and right_visual is not None,
        "clinical_claim": "NONE",
    }
    arrays = {
        "left_signal": left_field.signal_map,
        "right_signal": right_field.signal_map,
        "left_coverage": left_field.coverage_count,
        "right_coverage": right_field.coverage_count,
        "left_thermal_evidence": session.left_anomaly.evidence_map,
        "right_thermal_evidence": session.right_anomaly.evidence_map,
        "left_fused_evidence": left_fused,
        "right_fused_evidence": right_fused,
        "bilateral_asymmetry": session.bilateral.asymmetry_map,
        "bilateral_signed_difference": session.bilateral.signed_difference_map,
        "bilateral_joint_mask": session.bilateral.joint_mask.astype(np.uint8),
    }
    return result, arrays


def persist_session_evidence(result: dict, arrays: dict[str, np.ndarray], directory: str | Path) -> dict:
    """Persist numerical session evidence and return filenames for API/report wiring."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "session_maps.npz", **arrays)
    (out / "session_evidence.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "evidence_filename": "session_evidence.json",
        "maps_filename": "session_maps.npz",
    }
