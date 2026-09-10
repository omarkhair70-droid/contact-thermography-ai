from __future__ import annotations

import numpy as np
import pandas as pd


EMBEDDING_SEMANTICS = (
    "research embedding-domain comparison only; not tumor size, probability, diagnosis, or cancer risk"
)


def _normalize_rows(matrix: np.ndarray, expected_dim: int | None = 384) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Embedding matrix must be 2-dimensional")
    if expected_dim is not None and values.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding matrix must have {expected_dim} columns, received {values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Embedding matrix must contain only finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Embedding matrix contains a zero-norm row")
    return values / norms


def pairwise_cosine_distance_frame(
    ids: list[str],
    embeddings: np.ndarray,
) -> pd.DataFrame:
    normalized = _normalize_rows(embeddings)
    if len(ids) != len(normalized):
        raise ValueError("ids length must match embedding row count")
    distances = 1.0 - np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(distances, 0.0)
    return pd.DataFrame(distances, index=ids, columns=ids)


def summarize_reference_domain_shift(
    client_ids: list[str],
    client_embeddings: np.ndarray,
    reference_ids: list[str],
    reference_embeddings: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """Compare real-device embeddings against the publication reference domain.

    The baseline is each reference plate's nearest *other* reference plate. Client
    nearest-reference distances are then expressed relative to that empirical
    reference-only distribution. This quantifies visual/domain shift without
    converting the distance into a disease or tumor score.
    """

    client = _normalize_rows(client_embeddings)
    reference = _normalize_rows(reference_embeddings)
    if len(client_ids) != len(client):
        raise ValueError("client_ids length must match client embedding row count")
    if len(reference_ids) != len(reference):
        raise ValueError("reference_ids length must match reference embedding row count")
    if len(reference) < 2:
        raise ValueError("At least two reference embeddings are required")

    reference_pairwise = 1.0 - np.clip(reference @ reference.T, -1.0, 1.0)
    np.fill_diagonal(reference_pairwise, np.inf)
    reference_nn = reference_pairwise.min(axis=1)

    client_to_reference = 1.0 - np.clip(client @ reference.T, -1.0, 1.0)
    nearest_idx = client_to_reference.argmin(axis=1)
    nearest_distance = client_to_reference[np.arange(len(client)), nearest_idx]

    ref_median = float(np.median(reference_nn))
    ref_q1 = float(np.quantile(reference_nn, 0.25))
    ref_q3 = float(np.quantile(reference_nn, 0.75))
    ref_iqr = max(ref_q3 - ref_q1, 1e-12)

    nearest_rows = []
    for row, client_id in enumerate(client_ids):
        distance = float(nearest_distance[row])
        nearest_rows.append(
            {
                "source_image": client_id,
                "nearest_reference_plate": reference_ids[int(nearest_idx[row])],
                "cosine_distance": distance,
                "percentile_vs_reference_self_nn": float((reference_nn <= distance).mean()),
                "distance_minus_reference_median_iqr": (distance - ref_median) / ref_iqr,
                "clinical_claim": "NONE",
            }
        )

    client_centroid = client.mean(axis=0)
    reference_centroid = reference.mean(axis=0)
    client_centroid /= max(float(np.linalg.norm(client_centroid)), 1e-12)
    reference_centroid /= max(float(np.linalg.norm(reference_centroid)), 1e-12)
    centroid_distance = float(
        1.0 - np.clip(np.dot(client_centroid, reference_centroid), -1.0, 1.0)
    )

    client_median = float(np.median(nearest_distance))
    summary = {
        "metric": "DINOv2 client-vs-publication embedding domain shift",
        "client_count": len(client_ids),
        "reference_count": len(reference_ids),
        "reference_self_nearest_cosine_distance_median": ref_median,
        "reference_self_nearest_cosine_distance_iqr": ref_iqr,
        "client_nearest_reference_cosine_distance_median": client_median,
        "client_to_reference_nearest_distance_ratio": (
            client_median / max(ref_median, 1e-12)
        ),
        "client_reference_centroid_cosine_distance": centroid_distance,
        "client_fraction_beyond_all_reference_self_nn": float(
            (nearest_distance > float(reference_nn.max())).mean()
        ),
        "semantics": EMBEDDING_SEMANTICS,
        "clinical_claim": "NONE",
    }
    return summary, pd.DataFrame(nearest_rows)
