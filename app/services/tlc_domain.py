from __future__ import annotations

REFERENCE_TLC_PROFILE = "reference-publication-unknown"


def normalize_device_profile(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_profile_provenance(tlc_profile_id: str, device_profile_ids: list[str] | None = None) -> dict:
    devices = sorted({str(item).strip() for item in (device_profile_ids or []) if str(item).strip()})
    in_reference_domain = tlc_profile_id == REFERENCE_TLC_PROFILE
    return {
        "tlc_profile_id": tlc_profile_id,
        "device_profile_ids": devices,
        "reference_profile_id": REFERENCE_TLC_PROFILE,
        "domain_status": "REFERENCE_PUBLICATION_DOMAIN" if in_reference_domain else "SEPARATE_TLC_DOMAIN",
        "matches_reference_colour_domain": in_reference_domain,
        "colour_calibration_status": "UNKNOWN_REFERENCE_CALIBRATION" if in_reference_domain else "PROFILE_CALIBRATION_REQUIRED",
        "absolute_temperature_interpretation": "DISABLED_UNTIL_PROFILE_CALIBRATED",
        "clinical_inference": "DISABLED_UNTIL_VALIDATED_MODEL",
    }
