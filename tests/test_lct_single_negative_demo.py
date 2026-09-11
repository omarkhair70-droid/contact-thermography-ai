from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_lct_single_negative_demo import train_demo


def _fixture(tmp_path: Path, negatives: int = 1):
    rng = np.random.default_rng(42)
    count = 9
    matrix = rng.normal(size=(count, 384)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    npy = tmp_path / "embeddings.npy"
    np.save(npy, matrix)

    names = [f"IMG-{i:04d}.jpg" for i in range(count)]
    index = tmp_path / "index.csv"
    pd.DataFrame({"source_image": names, "embedding_row": range(count)}).to_csv(index, index=False)

    labels = ["HEALTHY"] * negatives + ["TUMOR_BEARING"] * (count - negatives)
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "subject_id": [f"MOUSE-{i:04d}" for i in range(count)],
            "image_id": names,
            "tumor_status": labels,
        }
    ).to_csv(truth, index=False)
    return npy, index, truth


def test_demo_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    npy, index, truth = _fixture(tmp_path)
    with pytest.raises(ValueError, match="acknowledgement"):
        train_demo(
            npy,
            index,
            truth,
            tmp_path / "model.json",
            acknowledge_single_negative=False,
        )


def test_demo_requires_exactly_one_negative(tmp_path: Path) -> None:
    npy, index, truth = _fixture(tmp_path, negatives=2)
    with pytest.raises(ValueError, match="exactly one negative"):
        train_demo(
            npy,
            index,
            truth,
            tmp_path / "model.json",
            acknowledge_single_negative=True,
        )


def test_demo_writes_portable_nonclinical_artifact(tmp_path: Path) -> None:
    npy, index, truth = _fixture(tmp_path)
    out = tmp_path / "model.json"
    payload = train_demo(
        npy,
        index,
        truth,
        out,
        acknowledge_single_negative=True,
    )
    disk = json.loads(out.read_text(encoding="utf-8"))
    assert payload == disk
    assert disk["model_type"] == "logistic_regression_linear_json"
    assert disk["feature_contract_version"] == "lct-target-v1"
    assert disk["feature_block"] == "dinov2"
    assert disk["feature_offset"] == 33
    assert disk["feature_count"] == 384
    assert disk["full_feature_count"] == 417
    assert len(disk["coefficients"]) == 384
    assert disk["training"]["subjects"] == 9
    assert disk["training"]["positive_subjects"] == 8
    assert disk["training"]["negative_subjects"] == 1
    assert disk["training"]["validation_status"] == "NOT_VALIDATED_SINGLE_NEGATIVE"
    assert disk["clinical_claim"] == "NONE"
    assert disk["clinical_use"] is False
    assert "cancer probability" in disk["semantics"]
