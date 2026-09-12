"""Evaluate a prespecified two-threshold candidate on locked calibration subjects."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.mumguard_intake import read_intake, aggregate, lock_hash
from research.mumguard.fit_subject_heads import score


def calibrate(intake, splits, head_path, low, high):
    if not 0 <= low < high <= 1:
        raise ValueError('require 0 <= low < high <= 1')
    rows, lock = read_intake(intake, splits)
    head = json.loads(Path(head_path).read_text())
    if head.get('species') != 'human' or head.get('endpoint') != 'lesion_presence' or head.get('split_sha256') != lock_hash(splits):
        raise ValueError('matching human head and locked split required')
    if head.get('signature') != lock['signature'] or head.get('feature_names') != lock['feature_names']:
        raise ValueError('head signature/feature mismatch')
    ids, X, y = aggregate(rows, 'calibration')
    train_ids = sorted(k for k, v in lock['assignments'].items() if v == 'train')
    if sorted(head.get('training_subjects', [])) != train_ids or set(ids) & set(head['training_subjects']):
        raise ValueError('training provenance mismatch or calibration overlap')
    values = score(head['model'], X)
    return {'status': 'INACTIVE_CALIBRATION_CANDIDATE', 'clinical_claim': 'NONE',
        'split_sha256': lock_hash(splits), 'tau_low': low, 'tau_high': high,
        'selection': 'caller prespecified; not optimized on test', 'test_subjects_used': [],
        'rows': [{'subject_id': s, 'lesion_presence': int(label), 'score': float(v),
                  'research_binary_class': 'TUMOR_LIKE' if v >= high else 'NO_TUMOR_LIKE' if v < low else None}
                 for s, label, v in zip(ids, y, values)], 'deployment_ready': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('intake', 'splits', 'head', 'out'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--tau-low', type=float, required=True)
    p.add_argument('--tau-high', type=float, required=True)
    a = p.parse_args()
    result = calibrate(a.intake, a.splits, a.head, a.tau_low, a.tau_high)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
