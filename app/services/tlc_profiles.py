from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATH = ROOT / "config" / "tlc_profiles.json"


class UnknownTLCProfileError(ValueError):
    pass


@dataclass(frozen=True)
class TLCProfile:
    id: str
    label: str
    calibration_status: str
    saturation_min: int
    value_min: int
    min_component_area_px: int
    normalization_mode: str
    qc_dark_clip_value: int
    qc_bright_clip_value: int
    qc_saturation_clip_value: int

    def provenance(self) -> dict:
        return {
            "tlc_profile_id": self.id,
            "label": self.label,
            "calibration_status": self.calibration_status,
            "normalization_mode": self.normalization_mode,
            "colour_interpretation_calibrated": False,
        }


@lru_cache(maxsize=8)
def _load_profiles(path_text: str) -> dict[str, TLCProfile]:
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = {}
    for raw in payload.get("profiles", []):
        segmentation = raw.get("active_response_segmentation", {})
        normalization = raw.get("normalization", {})
        profile = TLCProfile(
            id=str(raw["id"]),
            label=str(raw.get("label") or raw["id"]),
            calibration_status=str(raw.get("calibration_status") or "UNKNOWN"),
            saturation_min=int(segmentation["saturation_min"]),
            value_min=int(segmentation["value_min"]),
            min_component_area_px=int(segmentation["min_component_area_px"]),
            normalization_mode=str(normalization.get("mode") or "identity"),
            qc_dark_clip_value=int(raw["engineering_qc"]["dark_clip_value"]),
            qc_bright_clip_value=int(raw["engineering_qc"]["bright_clip_value"]),
            qc_saturation_clip_value=int(
                raw["engineering_qc"]["saturation_clip_value"]
            ),
        )
        profiles[profile.id] = profile
    if not profiles:
        raise RuntimeError(f"No TLC profiles were configured in {path}")
    return profiles


def profile_path() -> Path:
    configured = os.getenv("TLC_PROFILES_PATH")
    return Path(configured) if configured else DEFAULT_PROFILE_PATH


def resolve_tlc_profile(profile_id: str) -> TLCProfile:
    profiles = _load_profiles(str(profile_path().resolve()))
    try:
        return profiles[profile_id]
    except KeyError as exc:
        allowed = ", ".join(sorted(profiles))
        raise UnknownTLCProfileError(
            f"Unknown tlc_profile_id '{profile_id}'. Configured profiles: {allowed}"
        ) from exc


def configured_profile_ids() -> list[str]:
    return sorted(_load_profiles(str(profile_path().resolve())))


def normalize_plate_bgr(bgr: np.ndarray, profile: TLCProfile) -> np.ndarray:
    """Apply the selected formulation profile before colour-derived analysis.

    Both bundled profiles intentionally remain identity transforms until their
    formulation-specific calibration data exists. Keeping this dispatch here
    prevents a future profile from silently changing every domain.
    """
    if profile.normalization_mode in {"identity", "identity-pending-calibration"}:
        return np.asarray(bgr, dtype=np.uint8).copy()
    raise ValueError(
        f"Unsupported normalization mode '{profile.normalization_mode}' "
        f"for TLC profile '{profile.id}'"
    )
