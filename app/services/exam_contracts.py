from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

Side = Literal["LEFT", "RIGHT", "UNKNOWN"]

DEFAULT_TLC_PROFILE = "client-device-tlc-pending"

@dataclass
class UploadPlateMeta:
    filename: str
    side: Side = "UNKNOWN"
    position: Optional[str] = None
    sequence_index: Optional[int] = None
    tlc_profile_id: str = DEFAULT_TLC_PROFILE
    device_profile_id: Optional[str] = None


def normalize_side(value: str | None) -> Side:
    if not value:
        return "UNKNOWN"
    v = value.strip().upper()
    if v in {"L", "LEFT"}:
        return "LEFT"
    if v in {"R", "RIGHT"}:
        return "RIGHT"
    return "UNKNOWN"


def normalize_tlc_profile(value: str | None) -> str:
    """Keep TLC formulation/domain explicit in every analysis contract."""
    if not value:
        return DEFAULT_TLC_PROFILE
    return value.strip() or DEFAULT_TLC_PROFILE
