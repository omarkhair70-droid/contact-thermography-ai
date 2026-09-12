"""Estimate paired repeatability on adaptation subjects; no disease or normal bank."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from app.services.mumguard_intake import read_intake, lock_hash


def adapt(intake, splits):
    rows, lock = read_intake(intake, splits)
    pairs = {}
    for r in rows:
        if r['split'] == 'adaptation':
            if not r.get('repeat_pair_id'):
                raise ValueError('adaptation requires independently identified same-region repeat pairs')
            pairs.setdefault((r['subject_id'], r['repeat_pair_id']), []).append(r)
    if not pairs or any(len(v) != 2 for v in pairs.values()):
        raise ValueError('exactly two real acquisitions per repeat pair required')
    by_subject = {}
    for (sid, _), pair in pairs.items():
        by_subject.setdefault(sid, []).append(np.abs(pair[0]['_features']-pair[1]['_features']))
    spread = np.median([np.median(v, axis=0) for v in by_subject.values()], axis=0)
    return {'status': 'INACTIVE_SENSOR_REPEATABILITY_ESTIMATE', 'clinical_claim': 'NONE',
        'adapter': 'identity', 'reason': 'No nuisance transformation learned without directional matched-condition metadata',
        'median_absolute_repeat_difference': spread.tolist(), 'subjects': sorted(by_subject),
        'signature': lock['signature'], 'split_sha256': lock_hash(splits),
        'normal_bank_created': False, 'disease_labels_used': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intake', required=True, type=Path)
    p.add_argument('--splits', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    result = adapt(a.intake, a.splits)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
