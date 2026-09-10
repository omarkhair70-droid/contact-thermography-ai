from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

Side = Literal["LEFT", "RIGHT", "UNKNOWN"]

@dataclass
class UploadPlateMeta:
    filename: str
    side: Side = "UNKNOWN"
    position: Optional[str] = None
    sequence_index: Optional[int] = None

def normalize_side(value: str | None) -> Side:
    if not value:
        return "UNKNOWN"
    v = value.strip().upper()
    if v in {"L", "LEFT"}:
        return "LEFT"
    if v in {"R", "RIGHT"}:
        return "RIGHT"
    return "UNKNOWN"
