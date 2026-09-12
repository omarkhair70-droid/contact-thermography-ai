"""Inactive candidate: callable evidence extraction, never disease scoring/activation."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from app.services.contact_field import normalize_field, original_box
from app.services.local_contrast_features import extract, CONTRACT
from app.services.tlc_observability import observe


def candidate_status(metadata=None):
    m = metadata or {}
    reasons = []
    if m.get('species') not in {'mouse', 'human'}:
        reasons.append('MISSING_OR_UNSUPPORTED_SPECIES')
    if m.get('species') == 'human':
        reasons.append('HUMAN_DECISION_HEAD_UNAVAILABLE')
    if m.get('tlc_profile_id') != 'client-device-tlc-pending':
        reasons.append('TLC_PROFILE_MISMATCH')
    if not m.get('device_profile_id') or m.get('device_profile_id') == 'client-device-unknown':
        reasons.append('UNRESOLVED_DEVICE_PROFILE')
    if m.get('acquisition_type') != 'contact-LCT':
        reasons.append('ACQUISITION_TYPE_UNSUPPORTED')
    calibration = str(m.get('calibration_status') or 'UNCALIBRATED')
    if calibration in {'UNCALIBRATED', 'CLIENT_IMAGES_RECEIVED_UNCALIBRATED'}:
        reasons.append('UNCALIBRATED_TLC')
    else:
        reasons.append('CALIBRATION_NOT_VERIFIED')
    return {'name': 'mumguard-local-evidence', 'status': 'INACTIVE_RESEARCH_CANDIDATE',
        'available': False, 'evidence_engine_available': True, 'decision_head_available': False,
        'target_species': 'human', 'development_challenge_species': 'mouse',
        'feature_contract_version': CONTRACT,
        'research_binary_class': None, 'model_score': None, 'clinical_claim': 'NONE',
        'clinical_use': False, 'decision_status': 'ABSTAIN', 'quality_reasons': reasons,
        'semantics': 'uncalibrated local evidence; not tumor probability, diagnosis, or size',
        'provenance': dict(m)}


def analyze_research(source, metadata, contact=None, exclusion=None, tokens=None, token_provenance=None):
    status = candidate_status(metadata)
    rgb, cm, em, frame, transform = normalize_field(source, contact, exclusion)
    states, observation = observe(rgb, cm, em)
    reasons = list(status['quality_reasons'])
    if contact is None:
        reasons.append('MISSING_INDEPENDENT_CONTACT_MASK')
    if metadata.get('contact_annotation_status') != 'INDEPENDENT_REVIEWED':
        reasons.append('UNCONFIRMED_CONTACT_ANNOTATION')
    if tokens is not None:
        from app.services.dinov2_patch_encoder import REVISION
        if not token_provenance or token_provenance.get('input_sha256') != transform['normalized_rgb_sha256'] or token_provenance.get('revision') != REVISION:
            raise ValueError('patch tokens do not match image/revision')
    # Missing/unsupported domains still get explicit geometry/QC, never an inferred disease label.
    evidence, heat, support = extract(rgb, cm, em, tokens)
    reasons.extend(evidence['quality_reasons'])
    if tokens is None:
        reasons.append('DINO_CHANNEL_UNAVAILABLE')
    for region in evidence['candidates']:
        region['original_box_xyxy'] = original_box(region, transform)
    evidence.update(status)
    measurement_eligible = (
        metadata.get('contact_annotation_status') == 'INDEPENDENT_REVIEWED'
        and bool(evidence['candidates'])
        and not {'MISSING_OR_UNSUPPORTED_SPECIES', 'TLC_PROFILE_MISMATCH',
                 'UNRESOLVED_DEVICE_PROFILE', 'ACQUISITION_TYPE_UNSUPPORTED'} & set(reasons)
    )
    evidence.update({'quality_reasons': sorted(set(reasons)), 'observability': observation,
        'provenance': {**metadata, **transform, 'tokens': token_provenance},
        'local_evidence_available': bool(evidence['candidates']),
        'measurement_eligible': measurement_eligible,
        'decision_status': 'ABSTAIN_INACTIVE_RESEARCH' if measurement_eligible else 'ABSTAIN_UNUSABLE_MEASUREMENT',
        'disease_head_executed': False})
    return evidence, {'rgb': rgb, 'contact': cm, 'exclusion': em, 'frame': frame,
                      'observability': states, 'local_contrast': heat, 'support': support}


def persist_evidence(evidence, arrays, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'maps.npz', **{k: v for k, v in arrays.items() if k != 'rgb'})
    Image.fromarray(arrays['rgb']).save(directory/'normalized.png')
    heat = arrays['local_contrast']
    overlay = arrays['rgb'].copy()
    measured = arrays['support'] > 0
    overlay[measured, 0] = np.maximum(overlay[measured, 0], np.clip(heat[measured]*255, 0, 255).astype('uint8'))
    Image.fromarray(overlay).save(directory/'overlay.png')
    (directory/'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding='utf-8')
