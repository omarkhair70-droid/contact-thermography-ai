from __future__ import annotations

import json
from pathlib import Path

from app.services.native_classifier_runtime import native_classifier_status


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


def test_committed_registry_is_class_map_blocked_and_emits_no_class() -> None:
    status = native_classifier_status(ROOT / "artifacts" / "model_registry.json")
    assert status["available"] is False
    assert status["status"] == "BLOCKED_ON_CLIENT_CLASS_MAP"
    assert status["research_binary_class"] is None
    assert status["feature_contract_version"] == "lct-target-v1"
    assert status["tlc_profile_id"] == "client-device-tlc-pending"
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
