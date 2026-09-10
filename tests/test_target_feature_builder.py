from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from app.services.target_feature_builder import (
    EMBEDDING_DIM,
    FEATURE_CONTRACT_VERSION,
    build_target_feature_vector,
    feature_columns,
    feature_schema,
)
from scripts.build_lct_target_features import build_feature_table


def _synthetic_client_image() -> np.ndarray:
    image = np.zeros((180, 280, 3), dtype=np.uint8)
    cv2.ellipse(image, (140, 95), (70, 35), 12, 0, 360, (40, 235, 160), -1)
    cv2.circle(image, (210, 130), 18, (20, 190, 245), -1)
    return image


def _embedding() -> np.ndarray:
    return np.linspace(0.01, 1.0, EMBEDDING_DIM, dtype=np.float32)


def test_target_feature_vector_is_finite_and_versioned() -> None:
    result = build_target_feature_vector(
        _synthetic_client_image(),
        "client-device-tlc-pending",
        embedding=_embedding(),
    )
    assert result.vector.shape == (417,)
    assert np.isfinite(result.vector).all()
    assert len(feature_columns()) == 417
    assert feature_schema()["total_features"] == 417
    assert feature_schema()["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert result.morphology_descriptor in feature_schema()["groups"]["morphology_one_hot"]["names"]
    morph_start = feature_schema()["groups"]["morphology_one_hot"]["start"]
    morph_count = feature_schema()["groups"]["morphology_one_hot"]["count"]
    assert result.vector[morph_start : morph_start + morph_count].sum() == pytest.approx(1.0)


def test_wrong_embedding_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        build_target_feature_vector(
            _synthetic_client_image(),
            "client-device-tlc-pending",
            embedding=np.ones(12, dtype=np.float32),
        )


def test_reference_profile_is_not_silently_processed_with_client_selector() -> None:
    with pytest.raises(ValueError, match="supports only the client-device TLC profile"):
        build_target_feature_vector(
            _synthetic_client_image(),
            "reference-publication-unknown",
            embedding=_embedding(),
        )


def test_manifest_builder_preserves_frozen_target_semantics(tmp_path: Path) -> None:
    image_path = tmp_path / "mouse-001.jpg"
    assert cv2.imwrite(str(image_path), _synthetic_client_image())

    manifest = pd.DataFrame(
        [
            {
                "subject_id": "MOUSE-001",
                "image_path": "client-private://mouse-001.jpg",
                "source_id": "client-mice-2026",
                "species": "mouse",
                "label": "TUMOR_BEARING",
                "label_provenance": "experimentally tumor-injected",
                "tlc_profile_id": "client-device-tlc-pending",
                "device_profile_id": "client-device-unknown",
                "use_role": "FROZEN_TARGET_EVAL",
                "train_eligible": False,
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    frame = build_feature_table(
        manifest_path,
        tmp_path,
        source_id="client-mice-2026",
        embedding_lookup={"mouse-001.jpg": _embedding()},
    )
    assert len(frame) == 1
    assert frame.iloc[0]["label"] == "TUMOR_BEARING"
    assert bool(frame.iloc[0]["train_eligible"]) is False
    assert frame.iloc[0]["use_role"] == "FROZEN_TARGET_EVAL"
    assert frame.iloc[0]["clinical_claim"] == "NONE"
    assert frame.iloc[0]["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert len([c for c in frame.columns if c.startswith("feature_")]) == 417
