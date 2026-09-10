from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ASSOCIATION_SEMANTICS = (
    "post-freeze descriptive association between thermochromic response and independently supplied "
    "tumor size; not model training, clinical validation, probability, diagnosis, or cancer risk"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    yr = pd.to_numeric(y, errors="coerce").rank(method="average")
    valid = xr.notna() & yr.notna()
    if int(valid.sum()) < 3:
        return None
    return _pearson(xr[valid].to_numpy(), yr[valid].to_numpy())


def evaluate_tumor_size_mapping(
    ranking: pd.DataFrame,
    labels: pd.DataFrame,
    size_column: str = "tumor_size",
) -> tuple[dict, pd.DataFrame]:
    required_ranking = {
        "source_image",
        "median_response_area_fraction",
        "median_blind_rank",
    }
    missing = required_ranking - set(ranking.columns)
    if missing:
        raise ValueError(f"Ranking is missing required columns: {', '.join(sorted(missing))}")
    if "source_image" not in labels.columns or size_column not in labels.columns:
        raise ValueError(f"Labels must contain source_image and {size_column}")

    if ranking["source_image"].duplicated().any() or labels["source_image"].duplicated().any():
        raise ValueError("source_image values must be unique in ranking and labels")

    ranking_ids = set(ranking["source_image"].astype(str))
    label_ids = set(labels["source_image"].astype(str))
    if ranking_ids != label_ids:
        missing_labels = sorted(ranking_ids - label_ids)
        extra_labels = sorted(label_ids - ranking_ids)
        raise ValueError(
            "Tumor-size mapping must match the frozen cohort exactly. "
            f"Missing labels: {missing_labels}; extra labels: {extra_labels}"
        )

    joined = ranking.merge(labels[["source_image", size_column]], on="source_image", how="inner")
    joined[size_column] = pd.to_numeric(joined[size_column], errors="coerce")
    if joined[size_column].isna().any() or not np.isfinite(joined[size_column]).all():
        raise ValueError("Tumor-size values must be finite numbers")
    if (joined[size_column] < 0).any():
        raise ValueError("Tumor-size values cannot be negative")

    area = joined["median_response_area_fraction"].astype(float)
    size = joined[size_column].astype(float)
    spearman_area_size = _spearman(area, size)
    pearson_area_size = _pearson(area.to_numpy(), size.to_numpy())

    joined["response_rank_desc"] = area.rank(ascending=False, method="average")
    joined["tumor_size_rank_desc"] = size.rank(ascending=False, method="average")
    rank_agreement = _spearman(joined["response_rank_desc"], joined["tumor_size_rank_desc"])

    # Descriptive straight-line fit only; do not call this validation on n=9.
    if len(joined) >= 2 and float(np.std(area)) > 1e-12:
        slope, intercept = np.polyfit(area.to_numpy(), size.to_numpy(), 1)
        predicted = slope * area.to_numpy() + intercept
        ss_res = float(np.sum((size.to_numpy() - predicted) ** 2))
        ss_tot = float(np.sum((size.to_numpy() - float(size.mean())) ** 2))
        r_squared = None if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    else:
        slope = intercept = r_squared = None

    summary = {
        "analysis_type": "post-freeze tumor-size association",
        "n": len(joined),
        "size_column": size_column,
        "spearman_response_area_vs_tumor_size": spearman_area_size,
        "pearson_response_area_vs_tumor_size": pearson_area_size,
        "spearman_descending_rank_agreement": rank_agreement,
        "descriptive_linear_fit": {
            "tumor_size_per_response_area_fraction": None if slope is None else float(slope),
            "intercept": None if intercept is None else float(intercept),
            "r_squared": r_squared,
        },
        "semantics": ASSOCIATION_SEMANTICS,
        "clinical_claim": "NONE",
    }
    return summary, joined.sort_values("median_blind_rank")
