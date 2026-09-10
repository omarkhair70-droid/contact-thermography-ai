from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_lct_native_binary import readiness, train


def _row(i: int, label: str, *, train_eligible: bool = True, species: str = "mouse", tlc: str = "client-tlc", device: str = "client-device") -> dict:
    return {
        "subject_id": f"S{i}",
        "image_path": f"client://S{i}.jpg",
        "source_id": "synthetic-test-fixture",
        "modality": "contact-LCT",
        "species": species,
        "label": label,
        "label_provenance": "test fixture",
        "split_group": "TRAIN_POOL",
        "license_tag": "TEST_ONLY",
        "tlc_profile_id": tlc,
        "device_profile_id": device,
        "use_role": "TRAIN_CANDIDATE" if train_eligible else "REFERENCE_ONLY",
        "train_eligible": train_eligible,
    }


def _write_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_features(tmp_path: Path, rows: list[dict]) -> Path:
    rng = np.random.default_rng(7)
    items = []
    for row in rows:
        if not row["train_eligible"]:
            continue
        positive = row["label"] in {"CANCER", "MALIGNANT", "TUMOR_BEARING"}
        base = 2.0 if positive else -2.0
        items.append({
            "subject_id": row["subject_id"],
            "feature_000": base + rng.normal(scale=0.2),
            "feature_001": base + rng.normal(scale=0.2),
            "feature_002": rng.normal(),
        })
    path = tmp_path / "features.csv"
    pd.DataFrame(items).to_csv(path, index=False)
    return path


def test_current_positive_only_style_pool_is_not_ready(tmp_path: Path) -> None:
    rows = [_row(i, "TUMOR_BEARING") for i in range(1, 10)]
    result = readiness(_write_manifest(tmp_path, rows), min_per_class=5)
    assert result["binary_training_ready"] is False
    assert result["positive_subjects"] == 9
    assert result["negative_subjects"] == 0
    assert any("negative" in reason for reason in result["reasons"])


def test_mixed_species_pool_is_rejected(tmp_path: Path) -> None:
    rows = [_row(i, "TUMOR_BEARING") for i in range(1, 6)]
    rows += [_row(i, "HEALTHY", species="human") for i in range(6, 11)]
    result = readiness(_write_manifest(tmp_path, rows), min_per_class=5)
    assert result["binary_training_ready"] is False
    assert any("one species" in reason for reason in result["reasons"])


def test_mixed_tlc_profile_pool_is_rejected(tmp_path: Path) -> None:
    rows = [_row(i, "TUMOR_BEARING") for i in range(1, 6)]
    rows += [_row(i, "HEALTHY", tlc="other-tlc") for i in range(6, 11)]
    result = readiness(_write_manifest(tmp_path, rows), min_per_class=5)
    assert result["binary_training_ready"] is False
    assert any("one TLC profile" in reason for reason in result["reasons"])


def test_ready_same_domain_pool_can_fit_baseline(tmp_path: Path) -> None:
    rows = [_row(i, "TUMOR_BEARING") for i in range(1, 7)]
    rows += [_row(i, "HEALTHY") for i in range(7, 13)]
    manifest = _write_manifest(tmp_path, rows)
    features = _write_features(tmp_path, rows)
    output = tmp_path / "out"
    result = train(manifest, features, output, min_per_class=5, folds=3)
    assert result["readiness"]["binary_training_ready"] is True
    assert result["clinical_claim"] == "NONE"
    assert (output / "lct_native_metrics.json").exists()
    assert (output / "lct_native_oof_predictions.csv").exists()
    assert (output / "logistic.joblib").exists()


def test_training_fails_closed_when_not_ready(tmp_path: Path) -> None:
    rows = [_row(i, "TUMOR_BEARING") for i in range(1, 6)]
    manifest = _write_manifest(tmp_path, rows)
    features = _write_features(tmp_path, rows)
    with pytest.raises(ValueError, match="training gate closed"):
        train(manifest, features, tmp_path / "out", min_per_class=5)
