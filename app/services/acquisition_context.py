from __future__ import annotations

import math
from typing import Any


CAPTURE_ROLES = {"SPATIAL_TILE", "TEMPORAL_FRAME", "UNKNOWN"}
PREPARATION_FLAGS = {
    "RECENT_PHYSICAL_ACTIVITY",
    "TOPICAL_PRODUCT_OR_COSMETIC",
    "HOT_OR_COLD_DRINK_OR_LARGE_MEAL",
    "NICOTINE",
    "ALCOHOL",
    "CIRCULATION_AFFECTING_MEDICATION",
    "THERMAL_TREATMENT_OR_EXTERNAL_HEATING_COOLING",
    "UNKNOWN_PREPARATION",
}


class AcquisitionContextError(ValueError):
    pass


def _optional_text(value: Any, *, max_length: int = 160) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_length:
        raise AcquisitionContextError(f"metadata text exceeds {max_length} characters")
    return text


def _optional_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AcquisitionContextError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise AcquisitionContextError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise AcquisitionContextError(f"{field} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise AcquisitionContextError(f"{field} must be <= {maximum}")
    return number


def normalize_preparation_flags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise AcquisitionContextError("preparation_flags must be a list or comma-separated string")
    normalized = []
    for flag in raw:
        key = flag.upper()
        if key not in PREPARATION_FLAGS:
            raise AcquisitionContextError(f"unsupported preparation flag: {flag}")
        if key not in normalized:
            normalized.append(key)
    return normalized


def normalize_acquisition_context(item: dict | None) -> dict:
    """Normalize capture/session metadata without turning protocol deviations into diagnoses.

    These fields are provenance/QC context. Their presence does not imply that a
    clinical protocol was satisfied, and missing values remain explicit ``None``.
    """
    raw = dict(item or {})
    role = str(raw.get("capture_role") or "UNKNOWN").upper()
    if role not in CAPTURE_ROLES:
        raise AcquisitionContextError(
            f"capture_role must be one of {', '.join(sorted(CAPTURE_ROLES))}"
        )
    return {
        "capture_role": role,
        "timestamp": _optional_text(raw.get("timestamp"), max_length=80),
        "room_temperature_c": _optional_number(
            raw.get("room_temperature_c"), field="room_temperature_c", minimum=-50.0, maximum=80.0
        ),
        "room_humidity_percent": _optional_number(
            raw.get("room_humidity_percent"), field="room_humidity_percent", minimum=0.0, maximum=100.0
        ),
        "acclimatization_minutes": _optional_number(
            raw.get("acclimatization_minutes"), field="acclimatization_minutes", minimum=0.0, maximum=240.0
        ),
        "lighting_profile_id": _optional_text(raw.get("lighting_profile_id")),
        "camera_profile_id": _optional_text(raw.get("camera_profile_id")),
        "tlc_batch_id": _optional_text(raw.get("tlc_batch_id")),
        "protocol_revision": _optional_text(raw.get("protocol_revision")),
        "operator_id": _optional_text(raw.get("operator_id")),
        "preparation_flags": normalize_preparation_flags(raw.get("preparation_flags")),
    }


def acquisition_context_completeness(context: dict) -> dict:
    """Return transparent metadata completeness only; this is not a medical QC score."""
    keys = (
        "capture_role",
        "room_temperature_c",
        "room_humidity_percent",
        "acclimatization_minutes",
        "lighting_profile_id",
        "camera_profile_id",
        "tlc_batch_id",
    )
    known = 0
    missing = []
    for key in keys:
        value = context.get(key)
        if key == "capture_role":
            present = value not in (None, "", "UNKNOWN")
        else:
            present = value not in (None, "")
        if present:
            known += 1
        else:
            missing.append(key)
    return {
        "known_fields": known,
        "field_count": len(keys),
        "fraction": known / len(keys),
        "missing_fields": missing,
        "semantics": "PROVENANCE_COMPLETENESS_NOT_CLINICAL_QUALITY",
    }
