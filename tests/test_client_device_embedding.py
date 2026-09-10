from __future__ import annotations

import numpy as np

from app.services.client_device_embedding import (
    pairwise_cosine_distance_frame,
    summarize_reference_domain_shift,
)


def _unit_rows(rows):
    matrix = np.asarray(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_pairwise_client_distance_matrix_is_symmetric_and_zero_diagonal():
    # Use 384-d rows so this matches the live DINOv2 contract without loading a model.
    embeddings = np.zeros((3, 384), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    embeddings[2, 0] = 1.0
    embeddings[2, 1] = 1.0
    frame = pairwise_cosine_distance_frame(["a", "b", "c"], embeddings)
    assert frame.shape == (3, 3)
    assert np.allclose(frame.to_numpy(), frame.to_numpy().T)
    assert np.allclose(np.diag(frame.to_numpy()), 0.0)


def test_embedding_domain_summary_uses_reference_self_neighbour_baseline():
    reference = np.zeros((4, 384), dtype=np.float32)
    reference[0, 0] = 1.0
    reference[1, 0] = 0.98
    reference[1, 1] = 0.2
    reference[2, 1] = 1.0
    reference[3, 1] = 0.98
    reference[3, 0] = 0.2
    reference = _unit_rows(reference)

    client = np.zeros((2, 384), dtype=np.float32)
    client[0, 2] = 1.0
    client[1, 3] = 1.0

    summary, nearest = summarize_reference_domain_shift(
        ["client-a", "client-b"],
        client,
        ["ref-a", "ref-b", "ref-c", "ref-d"],
        reference,
    )
    assert summary["clinical_claim"] == "NONE"
    assert "diagnosis" not in summary
    assert summary["client_count"] == 2
    assert summary["reference_count"] == 4
    assert summary["client_to_reference_nearest_distance_ratio"] > 1.0
    assert len(nearest) == 2
    assert set(nearest["clinical_claim"]) == {"NONE"}
    assert nearest["percentile_vs_reference_self_nn"].between(0.0, 1.0).all()


def test_embedding_domain_rejects_nonfinite_and_wrong_dimension():
    wrong = np.ones((2, 10), dtype=np.float32)
    try:
        pairwise_cosine_distance_frame(["a", "b"], wrong)
    except ValueError as exc:
        assert "384" in str(exc)
    else:
        raise AssertionError("wrong embedding dimension should fail")

    broken = np.ones((2, 384), dtype=np.float32)
    broken[0, 0] = np.nan
    try:
        pairwise_cosine_distance_frame(["a", "b"], broken)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite embeddings should fail")
