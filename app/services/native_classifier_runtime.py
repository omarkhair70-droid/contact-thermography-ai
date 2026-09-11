from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "artifacts" / "model_registry.json"
MODEL_NAME = "lct-native-binary"
ACTIVE_STATUS = "ACTIVE_RESEARCH"
EXPECTED_FEATURE_CONTRACT = "lct-target-v1"
EXPECTED_TLC_PROFILE = "client-device-tlc-pending"
EXPECTED_FEATURE_COUNT = 417


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


def _artifact_path(entry: dict, registry_path: Path) -> Path:
    artifact = str(entry.get("artifact") or "").strip()
    if not artifact:
        raise ValueError("ACTIVE_RESEARCH requires an artifact path")
    candidate = Path(artifact)
    if candidate.is_absolute():
        return candidate
    # Production registry paths are repository-root relative. For isolated tests,
    # also allow an artifact beside the temporary registry.
    rooted = ROOT / candidate
    local = registry_path.parent / candidate
    if rooted.is_file() or registry_path == REGISTRY_PATH:
        return rooted
    return local


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
            "model score is not a diagnosis or cancer probability"
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
            "validation_status": entry.get("validation_status"),
            "warning": entry.get("warning"),
        }

    try:
        artifact_path = _artifact_path(entry, path)
    except ValueError as exc:
        return {
            **base,
            "status": "MISSING_MODEL_ARTIFACT",
            "reason": str(exc),
        }
    if not artifact_path.is_file():
        return {
            **base,
            "status": "MISSING_MODEL_ARTIFACT",
            "reason": f"configured artifact does not exist: {entry.get('artifact')}",
        }

    return {
        **base,
        "available": True,
        "status": ACTIVE_STATUS,
        "version": entry.get("version"),
        "artifact": str(entry.get("artifact")),
        "activation_scope": entry.get("activation_scope"),
        "validation_status": entry.get("validation_status"),
        "warning": entry.get("warning"),
        "reason": None,
    }


def _load_linear_json(path: Path = REGISTRY_PATH) -> tuple[dict, dict]:
    status = native_classifier_status(path)
    if not status["available"]:
        raise RuntimeError(str(status.get("reason") or status.get("status")))
    entry = _model_entry(path)
    assert entry is not None
    artifact_path = _artifact_path(entry, path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    if payload.get("model_type") != "logistic_regression_linear_json":
        raise ValueError("unsupported native classifier artifact type")
    if payload.get("feature_contract_version") != EXPECTED_FEATURE_CONTRACT:
        raise ValueError("native artifact feature contract mismatch")
    if payload.get("tlc_profile_id") != EXPECTED_TLC_PROFILE:
        raise ValueError("native artifact TLC profile mismatch")
    if payload.get("clinical_use") is not False or payload.get("clinical_claim") != "NONE":
        raise ValueError("native artifact must remain non-clinical")
    if int(payload.get("full_feature_count", -1)) != EXPECTED_FEATURE_COUNT:
        raise ValueError("native artifact full feature count mismatch")

    offset = int(payload.get("feature_offset", -1))
    count = int(payload.get("feature_count", -1))
    coef = np.asarray(payload.get("coefficients", []), dtype=np.float64)
    if offset < 0 or count <= 0 or coef.shape != (count,) or offset + count > EXPECTED_FEATURE_COUNT:
        raise ValueError("native artifact coefficient layout is invalid")
    if not np.isfinite(coef).all() or not math.isfinite(float(payload.get("intercept", float("nan")))):
        raise ValueError("native artifact contains non-finite parameters")
    return payload, status


def predict_native_binary(feature_vector: np.ndarray, path: Path = REGISTRY_PATH) -> dict:
    """Score one lct-target-v1 feature vector with the active research head.

    The returned `model_score` is a research model score only. It is deliberately
    not named or presented as a clinical probability.
    """
    payload, status = _load_linear_json(path)
    vector = np.asarray(feature_vector, dtype=np.float64).reshape(-1)
    if vector.shape != (EXPECTED_FEATURE_COUNT,) or not np.isfinite(vector).all():
        raise ValueError(f"expected {EXPECTED_FEATURE_COUNT} finite target features")

    offset = int(payload["feature_offset"])
    count = int(payload["feature_count"])
    block = vector[offset : offset + count]
    if payload.get("feature_block") == "dinov2":
        norm = float(np.linalg.norm(block))
        if norm <= 1e-12:
            raise ValueError("DINOv2 feature block has zero norm")
        block = block / norm

    coef = np.asarray(payload["coefficients"], dtype=np.float64)
    logit = float(payload["intercept"]) + float(np.dot(coef, block))
    if logit >= 0:
        score = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        score = exp_logit / (1.0 + exp_logit)

    threshold = float(payload.get("decision_threshold", 0.5))
    binary_class = "TUMOR_LIKE" if score >= threshold else "NO_TUMOR_LIKE"
    return {
        **status,
        "research_binary_class": binary_class,
        "model_score": round(score, 6),
        "decision_threshold": threshold,
        "validation_status": payload.get("training", {}).get(
            "validation_status", status.get("validation_status")
        ),
        "feature_block": payload.get("feature_block"),
        "training_subjects": payload.get("training", {}).get("subjects"),
        "positive_subjects": payload.get("training", {}).get("positive_subjects"),
        "negative_subjects": payload.get("training", {}).get("negative_subjects"),
        "semantics": (
            "experimental research-only TUMOR_LIKE / NO_TUMOR_LIKE score; "
            "trained from 8 tumor-bearing and 1 no-tumor mouse; not independently validated, "
            "not tumor size, diagnosis, or cancer probability"
        ),
        "clinical_claim": "NONE",
        "clinical_use": False,
    }
