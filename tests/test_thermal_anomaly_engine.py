import numpy as np

from app.services.thermal_anomaly_engine import (
    build_multiscale_thermal_anomaly,
    fuse_evidence_maps,
)


def _synthetic_field(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    signal = np.full((size, size), 0.35, dtype=np.float32)
    hotspot = (xx - 32) ** 2 + (yy - 32) ** 2 <= 6 ** 2
    signal[hotspot] = 0.85
    mask = np.ones((size, size), dtype=bool)
    return signal, mask


def test_multiscale_engine_localizes_synthetic_hotspot():
    signal, mask = _synthetic_field()
    result = build_multiscale_thermal_anomaly(
        signal,
        mask,
        radii=(2, 4, 8),
        min_ring_pixels=8,
    )

    center = float(result.evidence_map[32, 32])
    quiet = float(result.evidence_map[8, 8])
    assert np.isfinite(center)
    assert center > quiet
    assert result.features["global_anomaly_score"] > 0.0
    assert result.provenance["label_free"] is True
    assert result.provenance["species_specific"] is False


def test_flat_field_has_low_anomaly_evidence():
    signal = np.full((48, 48), 0.5, dtype=np.float32)
    mask = np.ones_like(signal, dtype=bool)
    result = build_multiscale_thermal_anomaly(
        signal,
        mask,
        radii=(2, 5),
        min_ring_pixels=8,
    )
    values = result.evidence_map[result.observable_mask]
    assert values.size > 0
    assert float(np.nanmax(values)) < 0.05


def test_masked_regions_are_not_used_as_reference_output():
    signal, mask = _synthetic_field()
    mask[:12, :] = False
    result = build_multiscale_thermal_anomaly(
        signal,
        mask,
        radii=(2, 4),
        min_ring_pixels=8,
    )
    assert not result.observable_mask[:12, :].any()
    assert np.isnan(result.evidence_map[:12, :]).all()


def test_fusion_adapter_accepts_optional_visual_evidence():
    thermal = np.full((8, 8), 0.8, dtype=np.float32)
    visual = np.full((8, 8), 0.2, dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    fused = fuse_evidence_maps(
        thermal,
        mask,
        visual_evidence=visual,
        thermal_weight=0.75,
    )
    assert np.allclose(fused[mask], 0.65)


def test_fusion_without_visual_returns_thermal_channel():
    thermal = np.full((8, 8), 0.42, dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    fused = fuse_evidence_maps(thermal, mask)
    assert np.allclose(fused[mask], 0.42)
