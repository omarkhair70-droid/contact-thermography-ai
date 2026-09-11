from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.services.native_classifier_runtime import (
    native_classifier_status,
    predict_native_binary,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_registry(tmp_path: Path, entry: dict) -> Path:
    path = tmp_path / "model_registry.json"
    path.write_text(json.dumps({"registry_version": 1, "models": [entry]}), encoding="utf-8")
    return path


def _base_entry() -> dict:
    return {
        "name": "lct-native-binary",
        "version": None,
        "status": "BLOCKED_ON_CLIENT_CLASS_MAP",
        "clinical_use": False,
        "clinical_claim": "NONE",
        "target_feature_contract_version": "lct-target-v1",
        "tlc_profile_id": "client-device-tlc-pending",
        "activation_scope": "INTERNAL_RESEARCH_CROSS_VALIDATION",
        "blocked_reason": "Need exact WA0030-WA0038 image-to-class mapping",
        "artifact": None,
    }


def _linear_artifact() -> dict:
    return {
        "name": "lct-native-binary",
        "version": "test",
        "model_type": "logistic_regression_linear_json",
        "feature_contract_version": "lct-target-v1",
        "feature_block": "custom-test-block",
        "feature_offset": 33,
        "feature_count": 2,
        "full_feature_count": 417,
        "tlc_profile_id": "client-device-tlc-pending",
        "decision_threshold": 0.5,
        "intercept": 0.0,
        "coefficients": [2.0, -1.0],
        "training": {
            "subjects": 9,
            "positive_subjects": 8,
            "negative_subjects": 1,
            "validation_status": "NOT_VALIDATED_SINGLE_NEGATIVE",
        },
        "clinical_claim": "NONE",
        "clinical_use": False,
    }


def test_committed_registry_is_active_research_demo_only() -> None:
    status = native_classifier_status(ROOT / "artifacts" / "model_registry.json")
    assert status["available"] is True
    assert status["status"] == "ACTIVE_RESEARCH"
    assert status["research_binary_class"] is None
    assert status["feature_contract_version"] == "lct-target-v1"
    assert status["tlc_profile_id"] == "client-device-tlc-pending"
    assert status["activation_scope"] == "INTERNAL_RESEARCH_DEMO_ONLY"
    assert status["validation_status"] == "NOT_VALIDATED_SINGLE_NEGATIVE"
    assert status["clinical_claim"] == "NONE"
    assert status["clinical_use"] is False


def test_registry_cannot_activate_clinical_model(tmp_path: Path) -> None:
    entry = _base_entry()
    entry.update({"status": "ACTIVE_RESEARCH", "clinical_use": True, "artifact": "fake.joblib"})
    status = native_classifier_status(_write_registry(tmp_path, entry))
    assert status["available"] is False
    assert status["status"] == "INVALID_REGISTRY_CONTRACT"
    assert status["research_binary_class"] is None


def test_registry_rejects_wrong_target_feature_contract(tmp_path: Path) -> None:
    entry = _base_entry()
    entry["target_feature_contract_version"] = "wrong-contract"
    status = native_classifier_status(_write_registry(tmp_path, entry))
    assert status["available"] is False
    assert status["status"] == "FEATURE_CONTRACT_MISMATCH"


def test_registry_rejects_wrong_tlc_profile(tmp_path: Path) -> None:
    entry = _base_entry()
    entry["tlc_profile_id"] = "reference-publication-unknown"
    status = native_classifier_status(_write_registry(tmp_path, entry))
    assert status["available"] is False
    assert status["status"] == "TLC_PROFILE_MISMATCH"


def test_active_research_requires_real_artifact(tmp_path: Path) -> None:
    entry = _base_entry()
    entry.update({"status": "ACTIVE_RESEARCH", "artifact": "models/does-not-exist.joblib"})
    status = native_classifier_status(_write_registry(tmp_path, entry))
    assert status["available"] is False
    assert status["status"] == "MISSING_MODEL_ARTIFACT"
    assert status["research_binary_class"] is None


def test_linear_json_runtime_emits_research_class_and_score(tmp_path: Path) -> None:
    artifact = tmp_path / "model.json"
    artifact.write_text(json.dumps(_linear_artifact()), encoding="utf-8")
    entry = _base_entry()
    entry.update(
        {
            "version": "test",
            "status": "ACTIVE_RESEARCH",
            "artifact": "model.json",
            "activation_scope": "INTERNAL_RESEARCH_DEMO_ONLY",
            "validation_status": "NOT_VALIDATED_SINGLE_NEGATIVE",
        }
    )
    registry = _write_registry(tmp_path, entry)

    vector = np.zeros(417, dtype=float)
    vector[33] = 1.0
    vector[34] = 0.0
    result = predict_native_binary(vector, registry)
    assert result["available"] is True
    assert result["research_binary_class"] == "TUMOR_LIKE"
    assert result["model_score"] > 0.5
    assert result["clinical_claim"] == "NONE"
    assert result["clinical_use"] is False


def test_linear_json_runtime_rejects_wrong_vector_length(tmp_path: Path) -> None:
    artifact = tmp_path / "model.json"
    artifact.write_text(json.dumps(_linear_artifact()), encoding="utf-8")
    entry = _base_entry()
    entry.update({"status": "ACTIVE_RESEARCH", "artifact": "model.json"})
    registry = _write_registry(tmp_path, entry)

    try:
        predict_native_binary(np.zeros(416, dtype=float), registry)
    except ValueError as exc:
        assert "417" in str(exc)
    else:
        raise AssertionError("expected wrong feature vector length to fail closed")
