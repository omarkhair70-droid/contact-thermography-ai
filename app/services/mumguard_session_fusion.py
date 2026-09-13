from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from app.services.acquisition_context import (
    acquisition_context_completeness,
    normalize_acquisition_context,
)
from app.services.bilateral_session_engine import (
    SideFieldResult,
    build_three_channel_session_evidence,
    compose_side_signal,
    infer_sequential_offsets,
)
from app.services.client_device_domain import segment_client_response
from app.services.contact_field import normalize_field
from app.services.human_decision import HumanDecisionInput, build_human_decision
from app.services.human_transfer_domain_support import evaluate_session_source_support
from app.services.thermal_anomaly_engine import fuse_evidence_maps
from app.services.tlc_signal_processing import build_relative_thermal_map


@dataclass(frozen=True)
class SessionFrameInput:
    source: bytes
    source_name: str
    side: str
    sequence_index: int
    acquisition_context: dict | None = None


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


def _dino_novelty_from_tokens(
    tokens: np.ndarray,
    provenance: dict,
    response_mask: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray | None, dict]:
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
    height, width = output_shape
    up = cv2.resize(novelty.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    up[~response_mask] = np.nan
    return up, {**provenance, "status": "OK", "reference": "same_frame_response_mean"}


def _attach_batched_dino(prepared: list[dict]) -> dict:
    candidates = [item for item in prepared if item["response_mask"].any()]
    if not candidates:
        return {
            "requested_frames": 0,
            "encoded_frames": 0,
            "batched_inference": True,
            "elapsed_s": 0.0,
        }

    started = time.perf_counter()
    try:
        from app.services.dinov2_patch_encoder import encode_patches_batch

        outputs = encode_patches_batch(
            [cv2.cvtColor(item["bgr"], cv2.COLOR_BGR2RGB) for item in candidates]
        )
        if len(outputs) != len(candidates):
            raise RuntimeError("DINO batch output count did not match input frame count")
        encoded = 0
        for item, (tokens, provenance) in zip(candidates, outputs):
            visual_map, visual_provenance = _dino_novelty_from_tokens(
                tokens,
                provenance,
                item["response_mask"],
                item["bgr"].shape[:2],
            )
            item["visual_map"] = visual_map
            item["visual_provenance"] = visual_provenance
            if visual_map is not None:
                encoded += 1
        return {
            "requested_frames": len(candidates),
            "encoded_frames": encoded,
            "batched_inference": True,
            "elapsed_s": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:
        for item in candidates:
            item["visual_map"] = None
            item["visual_provenance"] = {
                "status": "DINO_UNAVAILABLE",
                "reason": str(exc),
                "batched_inference": True,
            }
        return {
            "requested_frames": len(candidates),
            "encoded_frames": 0,
            "batched_inference": True,
            "elapsed_s": round(time.perf_counter() - started, 4),
            "status": "DINO_UNAVAILABLE",
            "reason": str(exc),
        }


def _prepare_frame(frame: SessionFrameInput, tlc_profile_id: str) -> dict:
    side = str(frame.side).upper()
    if side not in {"LEFT", "RIGHT"}:
        raise ValueError("Each session frame side must be LEFT or RIGHT")
    acquisition_context = normalize_acquisition_context(frame.acquisition_context)
    bgr, valid, transform = _normalized_bgr(frame.source)
    response_mask_u8, components = segment_client_response(bgr, valid)
    response_mask = response_mask_u8 > 0
    signal = build_relative_thermal_map(
        bgr,
        response_mask,
        tlc_profile_id=tlc_profile_id,
    )
    return {
        "input": frame,
        "bgr": bgr,
        "response_mask": response_mask,
        "signal": signal,
        "visual_map": None,
        "visual_provenance": None,
        "component_count": len(components),
        "transform": transform,
        "acquisition_context": acquisition_context,
        "acquisition_context_completeness": acquisition_context_completeness(acquisition_context),
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


def _peak_location(array: np.ndarray, side: str) -> dict | None:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    masked = np.where(finite, values, -np.inf)
    y, x = np.unravel_index(int(np.argmax(masked)), values.shape)
    h, w = values.shape
    return {
        "side": side,
        "x_fraction": round(float(x / max(w - 1, 1)), 4),
        "y_fraction": round(float(y / max(h - 1, 1)), 4),
        "evidence": round(float(values[y, x]), 4),
        "coordinate_semantics": "normalized reconstructed-side field; not anatomical coordinates",
    }


def _research_finding(source_support: dict, scores: dict, left_fused: np.ndarray, right_fused: np.ndarray) -> dict:
    channel_items = [
        ("focal/core hyperthermia", float(scores.get("core_hyperthermia_score", 0.0) or 0.0)),
        ("bilateral asymmetry", float(scores.get("bilateral_asymmetry_score", 0.0) or 0.0)),
        ("abnormal thermal distribution", float(scores.get("abnormal_skin_behavior_score", 0.0) or 0.0)),
    ]
    dominant = [
        {"channel": name, "evidence": round(value, 4)}
        for name, value in sorted(channel_items, key=lambda item: item[1], reverse=True)
    ]
    left_peak = _peak_location(left_fused, "LEFT")
    right_peak = _peak_location(right_fused, "RIGHT")
    candidates = [item for item in (left_peak, right_peak) if item is not None]
    peak = max(candidates, key=lambda item: item["evidence"]) if candidates else None
    return {
        "status": source_support.get("research_decision", "INCONCLUSIVE"),
        "research_concern_score": source_support.get("research_concern_score"),
        "domain_status": source_support.get("status"),
        "decision_model_executed": bool(source_support.get("decision_model_executed")),
        "dominant_channels": dominant,
        "likely_side": peak.get("side") if peak else None,
        "normalized_peak": peak,
        "reason": source_support.get("reason"),
        "score_semantics": source_support.get(
            "score_semantics",
            "research concern only; not a cancer probability or clinical diagnosis",
        ),
        "clinical_risk": None,
        "clinical_claim": "NONE",
    }


def analyze_bilateral_session(
    frames: Sequence[SessionFrameInput],
    *,
    tlc_profile_id: str = "client-device-tlc-pending",
    hotter_is_higher: bool = True,
    include_dino: bool = True,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run the unified human-first MumGuard measurement/evidence pipeline."""
    if not frames:
        raise ValueError("At least one session frame is required")

    total_started = time.perf_counter()
    prepared_started = time.perf_counter()
    prepared = [_prepare_frame(frame, tlc_profile_id) for frame in frames]
    frame_preparation_s = time.perf_counter() - prepared_started

    dino_meta = {
        "requested_frames": 0,
        "encoded_frames": 0,
        "batched_inference": True,
        "elapsed_s": 0.0,
    }
    if include_dino:
        dino_meta = _attach_batched_dino(prepared)

    left = [item for item in prepared if item["input"].side.upper() == "LEFT"]
    right = [item for item in prepared if item["input"].side.upper() == "RIGHT"]
    if not left or not right:
        raise ValueError("A bilateral session requires LEFT and RIGHT frames")

    reconstruction_started = time.perf_counter()
    left_field, left_visual, left_offsets = _build_side(left)
    right_field, right_visual, right_offsets = _build_side(right)
    reconstruction_s = time.perf_counter() - reconstruction_started

    evidence_started = time.perf_counter()
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

    ai_evidence_available = left_visual is not None and right_visual is not None
    human_decision = build_human_decision(
        HumanDecisionInput(
            measurement_status=session.status,
            measurement_scores=dict(session.scores),
            ai_evidence_available=ai_evidence_available,
        )
    ).as_dict()
    source_support = evaluate_session_source_support(
        left_field.signal_map,
        right_field.signal_map,
        left_support=left_field.observable_mask,
        right_support=right_field.observable_mask,
        measurement_status=session.status,
    )
    finding = _research_finding(
        source_support,
        dict(session.scores),
        left_fused,
        right_fused,
    )
    evidence_s = time.perf_counter() - evidence_started
    total_s = time.perf_counter() - total_started

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
        "research_finding": finding,
        "human_decision": human_decision,
        "human_transfer_source_support": source_support,
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
                "acquisition_context": item["acquisition_context"],
                "acquisition_context_completeness": item["acquisition_context_completeness"],
            }
            for item in prepared
        ],
        "left_offsets_xy": [list(value) for value in left_offsets],
        "right_offsets_xy": [list(value) for value in right_offsets],
        "ai_evidence_available": ai_evidence_available,
        "performance": {
            "frame_count": len(frames),
            "frame_preparation_s": round(frame_preparation_s, 4),
            "dino_batch_s": round(float(dino_meta.get("elapsed_s", 0.0)), 4),
            "side_reconstruction_s": round(reconstruction_s, 4),
            "evidence_and_decision_s": round(evidence_s, 4),
            "total_analysis_s": round(total_s, 4),
            "dino_frames_requested": int(dino_meta.get("requested_frames", 0)),
            "dino_frames_encoded": int(dino_meta.get("encoded_frames", 0)),
            "dino_batched_inference": bool(dino_meta.get("batched_inference", False)),
            "dino_status": dino_meta.get("status", "OK" if include_dino else "NOT_REQUESTED"),
        },
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


def _render_map_preview(array: np.ndarray, path: Path) -> None:
    """Render a viewable PNG for an evidence array while keeping NaN support black."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Session map previews require 2-D arrays")
    finite = np.isfinite(values)
    gray = np.zeros(values.shape, dtype=np.uint8)
    if finite.any():
        selected = values[finite]
        lo, hi = np.percentile(selected, [5, 95])
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            lo = float(np.min(selected))
            hi = float(np.max(selected))
        if hi > lo:
            normalized = np.clip((values - float(lo)) / float(hi - lo), 0.0, 1.0)
            gray[finite] = np.round(normalized[finite] * 255.0).astype(np.uint8)
        else:
            gray[finite] = 127
    preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    preview[~finite] = 0
    if not cv2.imwrite(str(path), preview):
        raise OSError(f"Could not write session map preview: {path}")


def persist_session_evidence(result: dict, arrays: dict[str, np.ndarray], directory: str | Path) -> dict:
    """Persist numerical evidence plus viewable session-map previews."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "session_maps.npz", **arrays)

    preview_keys = (
        "left_thermal_evidence",
        "right_thermal_evidence",
        "left_fused_evidence",
        "right_fused_evidence",
        "bilateral_asymmetry",
    )
    preview_filenames = {}
    for key in preview_keys:
        if key not in arrays:
            continue
        filename = f"{key}.png"
        _render_map_preview(arrays[key], out / filename)
        preview_filenames[key] = filename

    result["preview_filenames"] = dict(preview_filenames)
    (out / "session_evidence.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    return {
        "evidence_filename": "session_evidence.json",
        "maps_filename": "session_maps.npz",
        "preview_filenames": preview_filenames,
    }
