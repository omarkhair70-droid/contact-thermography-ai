import csv
import json
import hashlib
import pytest
from app.services.mumguard_intake import read_intake
from scripts.train_mumguard_human_head import train
from scripts.adapt_mumguard_sensor import adapt
from scripts.calibrate_mumguard_decision import calibrate


def make_intake(tmp_path):
    signature = {'species': 'human', 'anatomy': 'breast', 'tlc_profile_id': 'test-tlc',
        'device_profile_id': 'test-device', 'protocol_revision': 'test-v1', 'feature_contract_version': 'mumguard-local-v1'}
    rows = []
    for i, split in enumerate(['train', 'train', 'calibration', 'test', 'adaptation', 'adaptation']):
        sid = 's4' if split == 'adaptation' else f's{i}'
        rows.append({**signature, 'subject_id': sid, 'split': split,
            'source_sha256': hashlib.sha256(str(i).encode()).hexdigest(),
            'lesion_presence': 'unknown' if split == 'adaptation' else 'present' if i % 2 else 'absent',
            'malignancy': 'unknown', 'reference_verification': 'VERIFIED',
            'data_use': 'AUTHORIZED_RESEARCH', 'contact_review': 'INDEPENDENT_REVIEWED',
            'repeat_pair_id': 'pair' if split == 'adaptation' else '', 'features_json': json.dumps([float(i), .2])})
    p, lock = tmp_path/'intake.csv', tmp_path/'split.json'
    lock.write_text(json.dumps({'status': 'LOCKED', 'signature': signature,
        'feature_names': ['physical_q90', 'physical_median'], 'assignments': {r['subject_id']: r['split'] for r in rows}}))
    save(p, rows)
    return p, lock, rows


def save(path, rows):
    with path.open('w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


@pytest.mark.parametrize('change', [
    {'species': 'mouse'}, {'lesion_presence': 'BENIGN'},
    {'lesion_presence': 'absent', 'malignancy': 'benign'},
    {'device_profile_id': 'different'}, {'reference_verification': 'UNKNOWN'},
    {'contact_review': 'AUTOMATIC'}, {'split': 'calibration'},
    {'features_json': '[null, 1]'},
])
def test_rejects_invalid_intake(tmp_path, change):
    p, lock, rows = make_intake(tmp_path)
    rows[0].update(change); save(p, rows)
    with pytest.raises(ValueError):
        read_intake(p, lock)


def test_target_train_adaptation_and_calibration_are_disjoint(tmp_path):
    p, lock, _ = make_intake(tmp_path)
    head = train(p, lock)
    assert head['training_subjects'] == ['s0', 's1']
    assert head['model']['mean'] == [.5, .2]
    assert head['threshold'] is None
    a = adapt(p, lock)
    assert a['subjects'] == ['s4'] and not a['normal_bank_created']
    hp = tmp_path/'head.json'; hp.write_text(json.dumps(head))
    result = calibrate(p, lock, hp, .3, .7)
    assert [r['subject_id'] for r in result['rows']] == ['s2']
    assert not result['test_subjects_used'] and not result['deployment_ready']
    head['training_subjects'].append('s2'); hp.write_text(json.dumps(head))
    with pytest.raises(ValueError, match='provenance|overlap'):
        calibrate(p, lock, hp, .3, .7)
