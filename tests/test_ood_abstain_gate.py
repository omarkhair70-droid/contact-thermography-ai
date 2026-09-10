import numpy as np

from scripts.ood_abstain_gate import apply_cosine_ood_gate, fit_cosine_ood_gate


def test_cosine_ood_gate_marks_far_target_as_abstain():
    source = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.20, 0.0],
            [0.98, -0.20, 0.0],
            [0.97, 0.0, 0.24],
        ],
        dtype=np.float32,
    )
    gate = fit_cosine_ood_gate(source, quantile=0.95)
    target = np.asarray([[1.0, 0.05, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    rows = apply_cosine_ood_gate(source, target, gate["threshold"])

    assert rows[0]["domain_status"] == "IN_DOMAIN"
    assert rows[0]["classification_allowed"] is True
    assert rows[1]["domain_status"] == "ABSTAIN_OOD"
    assert rows[1]["classification_allowed"] is False
    assert all(row["clinical_claim"] == "NONE" for row in rows)


def test_cosine_ood_gate_rejects_nonfinite_embeddings():
    source = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    bad = source.copy()
    bad[0, 0] = np.nan
    try:
        fit_cosine_ood_gate(bad)
    except ValueError as exc:
        assert "NaN/Inf" in str(exc)
    else:
        raise AssertionError("expected non-finite embeddings to be rejected")
