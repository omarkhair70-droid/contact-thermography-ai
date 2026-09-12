import numpy as np
import pytest

from app.services.tlc_signal_processing import (
    HueTemperaturePoint,
    TLCCalibrationError,
    build_calibrated_temperature_map,
    build_relative_thermal_map,
    summarize_signal_map,
    summarize_spatial_signal,
    summarize_transient_sequence,
)


def _rgb_strip_as_bgr() -> np.ndarray:
    return np.asarray([[[0, 0, 255], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8)


def test_relative_hue_index_tracks_red_green_blue_colour_play() -> None:
    image = _rgb_strip_as_bgr()
    mask = np.ones((1, 3), dtype=np.uint8) * 255

    result = build_relative_thermal_map(image, mask)

    assert result.mode == "relative_hue_index"
    np.testing.assert_allclose(result.signal_map[0], [0.0, 0.5, 1.0], atol=0.02)
    assert result.provenance["absolute_temperature"] is False


def test_empirical_calibration_upgrades_same_signal_path_to_celsius() -> None:
    image = _rgb_strip_as_bgr()
    mask = np.ones((1, 3), dtype=np.uint8)
    points = [
        HueTemperaturePoint(0.0, 30.0),
        HueTemperaturePoint(120.0, 32.0),
        HueTemperaturePoint(240.0, 34.0),
    ]

    result = build_calibrated_temperature_map(image, mask, points)

    assert result.mode == "absolute_temperature_c"
    np.testing.assert_allclose(result.signal_map[0], [30.0, 32.0, 34.0], atol=0.05)
    assert result.provenance["absolute_temperature"] is True


def test_calibration_rejects_duplicate_hue_points() -> None:
    image = _rgb_strip_as_bgr()
    mask = np.ones((1, 3), dtype=np.uint8)

    with pytest.raises(TLCCalibrationError):
        build_calibrated_temperature_map(
            image,
            mask,
            [(0.0, 30.0), (0.0, 31.0)],
        )


def test_static_summaries_do_not_invent_values_when_response_is_unobservable() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    result = build_relative_thermal_map(image, mask)

    summary = summarize_signal_map(result)
    spatial = summarize_spatial_signal(result)

    assert summary["count"] == 0
    assert summary["mean"] is None
    assert spatial["left_minus_right"] is None


def test_transient_baseline_extracts_delta_rate_and_peak() -> None:
    mask = np.ones((2, 2), dtype=bool)
    maps = [
        np.full((2, 2), 0.2, dtype=np.float32),
        np.full((2, 2), 0.5, dtype=np.float32),
        np.full((2, 2), 0.8, dtype=np.float32),
    ]

    summary = summarize_transient_sequence(maps, [mask, mask, mask], [0.0, 1.0, 2.0])

    assert summary["delta_mean"] == pytest.approx(0.6)
    assert summary["linear_rate_per_s"] == pytest.approx(0.3)
    assert summary["time_to_peak_s"] == pytest.approx(2.0)
