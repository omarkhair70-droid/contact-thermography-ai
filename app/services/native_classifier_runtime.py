from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "artifacts" / "model_registry.json"
MODEL_NAME = "lct-native-binary"
ACTIVE_STATUS = "ACTIVE_RESEARCH"
EXPECTED_FEATURE_CONTRACT = "lct-target-v1"
EXPECTED_TLC_PROFILE = "client-device-tlc-pending"


def _load_registry(path: Path = REGISTRY_PATH) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("model registry must contain a models list")
    return payload


def _model_entry(path: Path = REGISTRY_PATH) -> dict | None:
    registry = _load_registry(path)
    for entry in registry["models"]:
        if isinstance(entry, dict) and entry.get("name") == MODEL_NAME:
            return dict(entry)
    return None


def native_classifier_status(path: Path = REGISTRY_PATH) -> dict:
    """Return the user/API-safe runtime state for the target-native binary head.

    This function is intentionally fail-closed. A registry entry alone never
    enables inference: the model must be explicitly marked ACTIVE_RESEARCH,
    remain non-clinical, match the target feature/profile contract and point
    at a real artifact before the runtime can be considered available.
    """
    entry = _model_entry(path)
    base = {
        "name": MODEL_NAME,
        "available": False,
        "research_binary_class": None,
        "clinical_claim": "NONE",
        "clinical_use": False,
        "feature_contract_version": EXPECTED_FEATURE_CONTRACT,
        "tlc_profile_id": EXPECTED_TLC_PROFILE,
        "semantics": (
            "research-only TUMOR_LIKE / NO_TUMOR_LIKE classification; "
            "not a diagnosis or cancer probability"
        ),
    }
    if entry is None:
        return {
            **base,
            "status": "MISSING_REGISTRY_ENTRY",
            "reason": "native classifier registry entry is missing",
        }

    status = str(entry.get("status") or "UNSPECIFIED")
    if entry.get("clinical_use") is not False or entry.get("clinical_claim", "NONE") != "NONE":
        return {
            **base,
            "status": "INVALID_REGISTRY_CONTRACT",
            "reason": "native classifier must remain non-clinical with clinical_claim=NONE",
        }
    if entry.get("target_feature_contract_version") != EXPECTED_FEATURE_CONTRACT:
        return {
            **base,
            "status": "FEATURE_CONTRACT_MISMATCH",
            "reason": f"expected {EXPECTED_FEATURE_CONTRACT}",
        }
    if entry.get("tlc_profile_id") != EXPECTED_TLC_PROFILE:
        return {
            **base,
            "status": "TLC_PROFILE_MISMATCH",
            "reason": f"expected {EXPECTED_TLC_PROFILE}",
        }
    if status != ACTIVE_STATUS:
        return {
            **base,
            "status": status,
            "reason": str(entry.get("blocked_reason") or "native classifier is not activated"),
            "activation_scope": entry.get("activation_scope"),
        }

    artifact = entry.get("artifact")
    if not artifact:
        return {
            **base,
            "status": "MISSING_MODEL_ARTIFACT",
            "reason": "ACTIVE_RESEARCH requires an artifact path",
        }
    artifact_path = ROOT / str(artifact)
    if not artifact_path.is_file():
        return {
            **base,
            "status": "MISSING_MODEL_ARTIFACT",
            "reason": f"configured artifact does not exist: {artifact}",
        }

    return {
        **base,
        "available": True,
        "status": ACTIVE_STATUS,
        "version": entry.get("version"),
        "artifact": str(artifact),
        "activation_scope": entry.get("activation_scope"),
        "reason": None,
    }
