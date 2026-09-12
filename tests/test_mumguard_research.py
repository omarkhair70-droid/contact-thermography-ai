import io
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from app.services.contact_field import normalize_field, original_box
from app.services.mumguard_research_runtime import analyze_research, candidate_status, persist_evidence
from app.services.tlc_observability import observe
from scripts.run_mumguard_ablation import folds
from research.mumguard.fit_subject_heads import fit


def source_image(rgb):
    out = io.BytesIO(); Image.fromarray(rgb).save(out, format='PNG')
    return out.getvalue()


@pytest.mark.parametrize('metadata,reason', [
    ({}, 'MISSING_OR_UNSUPPORTED_SPECIES'),
    ({'species': 'human'}, 'HUMAN_DECISION_HEAD_UNAVAILABLE'),
    ({'species': 'mouse', 'tlc_profile_id': 'another'}, 'TLC_PROFILE_MISMATCH'),
    ({'calibration_status': 'CALIBRATED'}, 'CALIBRATION_NOT_VERIFIED'),
    ({'acquisition_type': 'radiometric-IR'}, 'ACQUISITION_TYPE_UNSUPPORTED'),
])
def test_domain_gates(metadata, reason):
    result = candidate_status(metadata)
    assert reason in result['quality_reasons']
    assert result['available'] is False
    assert result['research_binary_class'] is None


def test_missing_contact_is_not_frame_or_negative(tmp_path):
    rgb = np.full((192, 256, 3), [30, 180, 60], np.uint8)
    evidence, arrays = analyze_research(source_image(rgb), {'species': 'mouse'})
    assert not arrays['contact'].any() and arrays['frame'].any()
    assert evidence['local_contrast_score'] is None
    assert evidence['decision_status'] == 'ABSTAIN_UNUSABLE_MEASUREMENT'
    assert evidence['feature_channels']['physical_q90'] is None
    assert 'MISSING_INDEPENDENT_CONTACT_MASK' in evidence['quality_reasons']
    assert not evidence['disease_head_executed']
    persist_evidence(evidence, arrays, tmp_path)
    saved = json.loads((tmp_path/'evidence.json').read_text())
    assert saved['research_binary_class'] is None
    assert np.load(tmp_path/'maps.npz')['contact'].sum() == 0


def test_geometry_and_misaligned_mask():
    source = source_image(np.full((100, 200, 3), 120, np.uint8))
    _, _, _, _, geometry = normalize_field(source)
    assert geometry['resize_dimensions'] == [224, 112]
    assert original_box({'x': 112, 'y': 112, 'size': 112}, geometry) == [50., 0., 150., 100.]
    with pytest.raises(ValueError, match='EXIF'):
        normalize_field(source, np.ones((200, 100)))


def test_unobservable_contact_and_glare_have_separate_states():
    rgb = np.zeros((224, 224, 3), np.uint8)
    rgb[:100] = 255
    mask = np.ones((224, 224), bool)
    states, observation = observe(rgb, mask, np.zeros_like(mask))
    assert (states[:100] == 3).all() and (states[100:] == 1).all()
    assert observation['absolute_temperature'] is None
    evidence, _ = analyze_research(source_image(rgb), {'species': 'mouse'}, mask)
    assert evidence['local_contrast_score'] is None
    assert 'LOW_CHROMATIC_SUPPORT' in evidence['quality_reasons']


def test_mismatched_token_hash_rejected():
    rgb = np.zeros((224, 224, 3), np.uint8)
    with pytest.raises(ValueError, match='image/revision'):
        analyze_research(source_image(rgb), {}, tokens=np.ones((16, 16, 384)),
                         token_provenance={'input_sha256': 'wrong'})


def test_folds_fit_scaler_only_on_training_subjects():
    X = np.array([[1000., 1000.], [1., 1.], [2., 2.], [3., 3.]])
    rows = folds(X, np.array([1, 1, 1, 0]), ['a', 'b', 'c', 'd'])
    assert rows[0]['model']['mean'] == [2., 2.]
    assert 'a' not in rows[0]['train_subjects']
    with pytest.raises(ValueError, match='finite'):
        fit(np.array([[np.nan], [1.]]), np.array([1, 0]))


def test_presence_endpoint_rejects_generic_benign():
    from scripts.train_lct_native_binary import _binary_label
    with pytest.raises(ValueError):
        _binary_label('BENIGN')


def test_inactive_payload_persists_to_history_and_report(client, sample_png, monkeypatch):
    from app.services import analysis_engine
    monkeypatch.setattr(analysis_engine, 'detect_circular_plates', lambda _: (_ for _ in ()).throw(AssertionError('client must route before circles')))
    response = client.post('/api/exams/analyze', files=[('files', ('mouse.png', sample_png, 'image/png'))], data={
        'exam_id': 'mumguard-contract', 'metadata_json': json.dumps([{'filename': 'mouse.png',
        'tlc_profile_id': 'client-device-tlc-pending', 'species': 'mouse', 'acquisition_type': 'contact-LCT'}])})
    assert response.status_code == 200, response.text
    p = response.json()['sources'][0]['plates'][0]['local_research_evidence']
    assert p['status'] == 'INACTIVE_RESEARCH_CANDIDATE'
    assert p['provenance']['species'] == 'mouse'
    assert p['research_binary_class'] is None and p['disease_head_executed'] is False
    assert p['normalized_image_url'].endswith('/normalized.png')
    assert p['maps_url'].endswith('/maps.npz')
    saved = client.get('/api/exams/mumguard-contract').json()['result']
    assert saved['sources'][0]['plates'][0]['local_research_evidence'] == p
    assert 'MumGuard local research' in client.get('/reports/mumguard-contract').text


def test_client_qc_is_supplied_before_encoding(monkeypatch):
    from app.services import analysis_engine
    from app.services.tlc_profiles import resolve_tlc_profile
    events = []
    def encoder(*args):
        events.append('encoded')
        return {}, np.ones(384)
    monkeypatch.setattr(analysis_engine.dinov2_service, 'analyze_plate_rgb', encoder)
    monkeypatch.setattr(analysis_engine, 'assess_plate_quality', lambda *args: (_ for _ in ()).throw(AssertionError('wrong QC branch')))
    rgb = np.full((256, 256, 3), 100, np.uint8)
    from app.services.client_device_domain import summarize_response
    f = summarize_response(rgb, np.zeros((256, 256), np.uint8))
    result = analysis_engine.analyze_plate(rgb, 'x', resolve_tlc_profile('client-device-tlc-pending'),
        feature_override=f, mask_override=np.zeros((256, 256)), qc_override={'status': 'REVIEW_REQUIRED'})
    assert events == ['encoded'] and result['qc']['status'] == 'REVIEW_REQUIRED'
