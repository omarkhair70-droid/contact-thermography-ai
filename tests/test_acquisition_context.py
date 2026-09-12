import pytest

from app.services.acquisition_context import (
    AcquisitionContextError,
    acquisition_context_completeness,
    normalize_acquisition_context,
)


def test_normalizes_human_capture_context_without_inventing_missing_values():
    context = normalize_acquisition_context({
        "capture_role": "spatial_tile",
        "timestamp": "2026-09-12T17:00:00+03:00",
        "room_temperature_c": "25.4",
        "room_humidity_percent": 58,
        "acclimatization_minutes": 15,
        "lighting_profile_id": "mumguard-light-v1",
        "camera_profile_id": "camera-a",
        "tlc_batch_id": "batch-17",
        "protocol_revision": "MG-PROT-01",
        "preparation_flags": ["NICOTINE", "topical_product_or_cosmetic"],
    })
    assert context["capture_role"] == "SPATIAL_TILE"
    assert context["room_temperature_c"] == 25.4
    assert context["room_humidity_percent"] == 58.0
    assert context["acclimatization_minutes"] == 15.0
    assert context["preparation_flags"] == ["NICOTINE", "TOPICAL_PRODUCT_OR_COSMETIC"]


def test_missing_context_stays_explicit_and_does_not_block_measurement_contract():
    context = normalize_acquisition_context({})
    assert context["capture_role"] == "UNKNOWN"
    assert context["room_temperature_c"] is None
    assert context["tlc_batch_id"] is None
    completeness = acquisition_context_completeness(context)
    assert completeness["fraction"] == 0.0
    assert completeness["semantics"] == "PROVENANCE_COMPLETENESS_NOT_CLINICAL_QUALITY"


def test_rejects_invalid_environment_values_and_unknown_capture_roles():
    with pytest.raises(AcquisitionContextError):
        normalize_acquisition_context({"room_humidity_percent": 140})
    with pytest.raises(AcquisitionContextError):
        normalize_acquisition_context({"capture_role": "SPATIAL_AND_TEMPORAL"})
    with pytest.raises(AcquisitionContextError):
        normalize_acquisition_context({"preparation_flags": ["MADE_UP_FLAG"]})


def test_temporal_frame_role_is_explicit_not_inferred_from_sequence_index():
    spatial = normalize_acquisition_context({"capture_role": "SPATIAL_TILE"})
    temporal = normalize_acquisition_context({"capture_role": "TEMPORAL_FRAME"})
    assert spatial["capture_role"] != temporal["capture_role"]
