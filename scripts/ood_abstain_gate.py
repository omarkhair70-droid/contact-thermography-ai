from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("expected a non-empty 2D embedding matrix")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain NaN/Inf")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("zero-norm embedding encountered")
    return x / norms


def fit_cosine_ood_gate(source_embeddings: np.ndarray, quantile: float = 0.95) -> dict:
    if not 0.5 <= quantile < 1.0:
        raise ValueError("quantile must be in [0.5, 1.0)")
    source = _row_normalize(source_embeddings)
    if len(source) < 3:
        raise ValueError("at least 3 source embeddings are required")

    dist = 1.0 - np.clip(source @ source.T, -1.0, 1.0)
    np.fill_diagonal(dist, np.inf)
    loo_nn = dist.min(axis=1)
    threshold = float(np.quantile(loo_nn, quantile))
    return {
        "metric": "cosine_nearest_source_distance",
        "threshold": threshold,
        "threshold_quantile": float(quantile),
        "source_count": int(len(source)),
        "source_nn_median": float(np.median(loo_nn)),
        "source_nn_q95": float(np.quantile(loo_nn, 0.95)),
        "clinical_claim": "NONE",
    }


def apply_cosine_ood_gate(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    threshold: float,
) -> list[dict]:
    source = _row_normalize(source_embeddings)
    target = _row_normalize(target_embeddings)
    dist = 1.0 - np.clip(target @ source.T, -1.0, 1.0)
    nn = dist.min(axis=1)
    nearest_idx = dist.argmin(axis=1)
    rows = []
    for i, (distance, ref_idx) in enumerate(zip(nn, nearest_idx)):
        is_ood = bool(float(distance) > float(threshold))
        rows.append(
            {
                "target_index": int(i),
                "nearest_source_index": int(ref_idx),
                "nearest_source_cosine_distance": float(distance),
                "ood_threshold": float(threshold),
                "domain_status": "ABSTAIN_OOD" if is_ood else "IN_DOMAIN",
                "classification_allowed": not is_ood,
                "clinical_claim": "NONE",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_embeddings", type=Path)
    parser.add_argument("target_embeddings", type=Path)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    source = np.load(args.source_embeddings, allow_pickle=False)
    target = np.load(args.target_embeddings, allow_pickle=False)
    gate = fit_cosine_ood_gate(source, args.quantile)
    rows = apply_cosine_ood_gate(source, target, gate["threshold"])
    payload = {"gate": gate, "targets": rows, "clinical_claim": "NONE"}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
