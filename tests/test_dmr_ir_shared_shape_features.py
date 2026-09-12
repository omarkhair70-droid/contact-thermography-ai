from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.services.human_research_transfer import SHARED_THERMAL_SHAPE_FEATURES
from scripts.build_dmr_ir_shared_shape_features import build, features_from_values


def _write_case(root: Path, group: str, patient: str, side: str, offset: float) -> None:
    patient_root = root / group / f"PAC_{patient}"
    segdir = patient_root / "Segmentadas"
    matrix_dir = patient_root / "Matrizes"
    segdir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)

    name = f"PAC_{patient}_T01-{side}.png"
    matrix_name = f"PAC_{patient}_T01.txt"
    mask = np.ones((20, 20), dtype=np.uint8) * 255
    values = np.linspace(25.0 + offset, 35.0 + offset, 400, dtype=np.float32).reshape(20, 20)
    Image.fromarray(mask).save(segdir / name)
    np.savetxt(matrix_dir / matrix_name, values)


def test_features_are_offset_invariant():
    base = np.linspace(20.0, 38.0, 400, dtype=np.float64)
    a = features_from_values(base)
    b = features_from_values(base + 11.5)
    assert tuple(a) == SHARED_THERMAL_SHAPE_FEATURES
    for name in SHARED_THERMAL_SHAPE_FEATURES:
        assert np.isclose(a[name], b[name], atol=1e-10)


def test_build_recovers_both_labels_and_shared_contract(tmp_path: Path):
    _write_case(tmp_path, "Saudaveis", "001", "dir", 0.0)
    _write_case(tmp_path, "Doentes", "002", "esq", 1.0)

    manifest, features, errors, summary = build(tmp_path)

    assert len(manifest) == 2
    assert len(features) == 2
    assert errors.empty
    assert set(features["label"]) == {"HEALTHY", "CANCER"}
    assert set(SHARED_THERMAL_SHAPE_FEATURES).issubset(features.columns)
    assert summary["subjects_usable"] == 2
    assert summary["absolute_temperature_used"] is False
    assert summary["clinical_claim"] == "NONE"
