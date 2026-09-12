import numpy as np

from app.services.human_research_transfer_v2 import (
    BILATERAL_SPATIAL_FEATURES,
    extract_dimensionless_bilateral_spatial_features,
)


def _base_fields(size=80):
    y, x = np.mgrid[0:size, 0:size]
    left = 0.15 * np.sin(x / 8.0) + 0.10 * np.cos(y / 9.0) + 0.003 * x
    right = np.fliplr(left).copy()
    return left.astype(np.float64), right.astype(np.float64)


def test_v2_features_are_invariant_to_joint_positive_affine_transform():
    left, right = _base_fields()
    original = extract_dimensionless_bilateral_spatial_features(left, right)
    shifted_scaled = extract_dimensionless_bilateral_spatial_features(
        left * 7.25 + 31.7,
        right * 7.25 + 31.7,
    )

    assert tuple(original) == BILATERAL_SPATIAL_FEATURES
    for name in BILATERAL_SPATIAL_FEATURES:
        assert np.isclose(original[name], shifted_scaled[name], atol=2e-5), name


def test_v2_spatial_features_respond_to_localized_hotspot_without_distribution_only_shortcut():
    left, right = _base_fields()
    baseline = extract_dimensionless_bilateral_spatial_features(left, right)

    yy, xx = np.mgrid[0:left.shape[0], 0:left.shape[1]]
    hotspot = np.exp(-((xx - 25.0) ** 2 + (yy - 38.0) ** 2) / (2.0 * 5.0 ** 2))
    anomalous_left = left + 1.2 * hotspot
    changed = extract_dimensionless_bilateral_spatial_features(anomalous_left, right)

    assert changed["max_side_local_contrast_p95"] > baseline["max_side_local_contrast_p95"]
    assert changed["bilateral_pattern_mae"] > baseline["bilateral_pattern_mae"]
    assert changed["bilateral_hotspot_peak_abs_diff"] > baseline["bilateral_hotspot_peak_abs_diff"]


def test_v2_bilateral_pattern_is_low_for_mirrored_symmetric_pair():
    left, right = _base_fields()
    features = extract_dimensionless_bilateral_spatial_features(left, right)
    assert features["bilateral_pattern_mae"] < 1e-5
    assert features["bilateral_pattern_corr_distance"] < 1e-5


def test_v2_support_masks_and_nan_regions_are_respected():
    left, right = _base_fields()
    support = np.ones(left.shape, dtype=bool)
    support[:10, :] = False
    left = left.copy()
    right = right.copy()
    left[~support] = np.nan
    right[~support] = np.nan

    features = extract_dimensionless_bilateral_spatial_features(
        left,
        right,
        left_support=support,
        right_support=support,
    )
    assert set(features) == set(BILATERAL_SPATIAL_FEATURES)
    assert np.isfinite(list(features.values())).all()
