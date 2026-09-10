from __future__ import annotations

import pandas as pd
import pytest

from app.services.tumor_burden_validation import evaluate_tumor_size_mapping


def _ranking():
    return pd.DataFrame(
        {
            "source_image": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "median_response_area_fraction": [0.40, 0.30, 0.20, 0.10],
            "median_blind_rank": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_monotonic_independent_sizes_produce_positive_response_association():
    labels = pd.DataFrame(
        {
            "source_image": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "tumor_size": [40.0, 30.0, 20.0, 10.0],
        }
    )
    summary, joined = evaluate_tumor_size_mapping(_ranking(), labels)
    assert summary["clinical_claim"] == "NONE"
    assert summary["spearman_response_area_vs_tumor_size"] == pytest.approx(1.0)
    assert summary["spearman_descending_rank_agreement"] == pytest.approx(1.0)
    assert summary["descriptive_linear_fit"]["r_squared"] == pytest.approx(1.0)
    assert len(joined) == 4


def test_mapping_requires_exact_frozen_cohort_membership():
    labels = pd.DataFrame(
        {
            "source_image": ["a.jpg", "b.jpg", "c.jpg"],
            "tumor_size": [40.0, 30.0, 20.0],
        }
    )
    with pytest.raises(ValueError, match="match the frozen cohort exactly"):
        evaluate_tumor_size_mapping(_ranking(), labels)


def test_mapping_rejects_invalid_size_values():
    labels = pd.DataFrame(
        {
            "source_image": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "tumor_size": [40.0, 30.0, -2.0, 10.0],
        }
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        evaluate_tumor_size_mapping(_ranking(), labels)
