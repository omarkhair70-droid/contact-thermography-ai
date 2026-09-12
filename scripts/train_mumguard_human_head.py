"""Train a small human-only development head on locked train subjects."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.mumguard_intake import read_intake, aggregate, lock_hash
from research.mumguard.fit_subject_heads import fit


def train(intake, splits):
    rows, lock = read_intake(intake, splits)
    ids, X, y = aggregate(rows, 'train')
    return {'status': 'INACTIVE_RESEARCH_CANDIDATE', 'species': 'human', 'clinical_claim': 'NONE',
        'endpoint': 'lesion_presence', 'threshold': None, 'training_subjects': ids,
        'split_sha256': lock_hash(splits), 'signature': lock['signature'],
        'feature_names': lock['feature_names'], 'model': fit(X, y),
        'mouse_supervision_used': False, 'human_validation': 'NOT_ESTABLISHED'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intake', required=True, type=Path)
    p.add_argument('--splits', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    result = train(a.intake, a.splits)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
