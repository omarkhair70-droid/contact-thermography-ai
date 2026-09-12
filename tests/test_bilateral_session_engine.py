from __future__ import annotations

import numpy as np

from app.services.bilateral_session_engine import (
    SideFieldResult,
    analyze_bilateral_asymmetry,
    build_three_channel_session_evidence,
    compose_side_signal,
)


def _side(signal: np.ndarray) -> SideFieldResult:
    signal = np.asarray(signal, dtype=np.float32)
    mask = np.isfinite(signal)
    return SideFieldResult(
        signal_map=signal,
        observable_mask=mask,
        coverage_count=mask.astype(np.uint16),
        frame_offsets_xy=((0.0, 0.0),),
        provenance={"fixture": True},
    )


def test_compose_side_signal_deduplicates_overlap_by_averaging() -> None:
    first = np.ones((4, 5), dtype=np.float32)
    second = np.full((4, 5), 3.0, dtype=np.float32)
    mask = np.ones((4, 5), dtype=bool)

    result = compose_side_signal(
        [first, second],
        [mask, mask],
        [(0.0, 0.0), (3.0, 0.0)],
    )

    assert result.signal_map.shape == (4, 8)
    assert np.allclose(result.signal_map[:, :3], 1.0)
    assert np.allclose(result.signal_map[:, 3:5], 2.0)
    assert np.allclose(result.signal_map[:, 5:], 3.0)
    assert np.all(result.coverage_count[:, 3:5] == 2)
    assert result.provenance["deduplicates_overlap"] is True


def test_bilateral_asymmetry_is_low_for_mirrored_matching_fields() -> None:
    yy, xx = np.mgrid[:64, :64]
    left = 0.2 + 0.002 * yy + 0.003 * xx
    left[20:30, 12:22] += 0.3
    right = np.fliplr(left).copy()

    result = analyze_bilateral_asymmetry(_side(left), _side(right), output_size=64)

    assert result.features["status"] == "OK"
    assert result.features["joint_fraction"] > 0.95
    assert result.score < 0.05
    assert np.nanpercentile(result.asymmetry_map, 95) < 0.05


def test_bilateral_asymmetry_detects_one_sided_distribution_change() -> None:
    yy, xx = np.mgrid[:64, :64]
    left = 0.2 + 0.002 * yy + 0.003 * xx
    right = np.fliplr(left).copy()
    right[22:34, 38:50] += 1.0

    result = analyze_bilateral_asymmetry(_side(left), _side(right), output_size=64)

    assert result.features["status"] == "OK"
    assert result.score > 0.2
    assert result.features["asymmetry_p99"] > result.features["asymmetry_p90"]


def test_three_channel_engine_is_label_free_and_human_first() -> None:
    yy, xx = np.mgrid[:96, :96]
    left = 0.25 + 0.001 * yy
    right = np.fliplr(left).copy()
    left[35:48, 28:41] += 0.7

    result = build_three_channel_session_evidence(
        _side(left),
        _side(right),
        anomaly_radii=(2, 4, 8),
    )

    assert result.provenance["human_first"] is True
    assert result.provenance["label_free"] is True
    assert result.provenance["species_specific"] is False
    assert set(result.provenance["channels"]) == {
        "core_hyperthermia",
        "bilateral_asymmetry",
        "abnormal_skin_thermal_behavior",
    }
    assert result.scores["core_hyperthermia_score"] > 0.0
    assert result.scores["bilateral_asymmetry_score"] > 0.0
    assert result.scores["abnormal_skin_behavior_score"] > 0.0
    assert result.scores["clinical_claim"] == "NONE"
