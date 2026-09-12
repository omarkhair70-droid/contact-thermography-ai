"""Endpoint-specific human intake and locked subject splits for future research."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SIGNATURE_FIELDS = ('species', 'anatomy', 'tlc_profile_id', 'device_profile_id', 'protocol_revision', 'feature_contract_version')
SPLITS = {'adaptation', 'train', 'calibration', 'test'}


def read_intake(path, split_path):
    lock = json.loads(Path(split_path).read_text())
    if lock.get('status') != 'LOCKED' or not lock.get('assignments'):
        raise ValueError('nonempty LOCKED subject assignments are required')
    names = lock.get('feature_names', [])
    if not 1 <= len(names) <= 6 or len(set(names)) != len(names):
        raise ValueError('one to six unique prespecified feature names required')
    rows = list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
    if not rows:
        raise ValueError('no human acquisitions supplied')
    subjects, hashes, domains = {}, set(), set()
    for row in rows:
        subject = row['subject_id']
        if row['species'] != 'human':
            raise ValueError('human research scripts reject mouse input')
        if row['split'] not in SPLITS or lock['assignments'].get(subject) != row['split']:
            raise ValueError('subject split differs from locked assignments')
        if subject in subjects and subjects[subject] != (row['split'], row['lesion_presence']):
            raise ValueError('subject overlap or inconsistent endpoint')
        subjects[subject] = (row['split'], row['lesion_presence'])
        if row['lesion_presence'] not in {'present', 'absent', 'unknown'}:
            raise ValueError('explicit lesion_presence mapping required; BENIGN is not absence')
        if row['malignancy'] not in {'benign', 'malignant', 'unknown', 'not_applicable'}:
            raise ValueError('invalid malignancy ontology')
        if row['malignancy'] in {'benign', 'malignant'} and row['lesion_presence'] == 'absent':
            raise ValueError('benign/malignant lesion cannot become no lesion')
        if row['split'] != 'adaptation' and (row['lesion_presence'] == 'unknown' or row['reference_verification'] != 'VERIFIED'):
            raise ValueError('supervised pools need verified endpoint evidence')
        if row['data_use'] != 'AUTHORIZED_RESEARCH' or row['contact_review'] != 'INDEPENDENT_REVIEWED':
            raise ValueError('research rights and reviewed contact required')
        digest = row['source_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or digest in hashes:
            raise ValueError('invalid/duplicate source checksum')
        hashes.add(digest)
        domain = tuple(row.get(k, '') for k in SIGNATURE_FIELDS)
        if any(not v or 'unknown' in v.lower() or 'pending' in v.lower() for v in domain):
            raise ValueError('unresolved sensor signature')
        if dict(zip(SIGNATURE_FIELDS, domain)) != lock.get('signature'):
            raise ValueError('signature mismatch; cross-device experiments require their own lock')
        domains.add(domain)
        vector = np.asarray(json.loads(row['features_json']), float)
        if vector.shape != (len(names),) or not np.isfinite(vector).all():
            raise ValueError('finite ordered feature vector required; no missing-channel imputation')
        row['_features'] = vector
    if len(domains) != 1:
        raise ValueError('mixed device/profile pool requires separate explicit experiment')
    return rows, lock


def aggregate(rows, split):
    selected = [r for r in rows if r['split'] == split]
    ids = sorted({r['subject_id'] for r in selected})
    if not ids:
        raise ValueError(f'empty {split} pool')
    X = np.array([np.median([r['_features'] for r in selected if r['subject_id'] == sid], axis=0) for sid in ids])
    y = np.array([int(next(r for r in selected if r['subject_id'] == sid)['lesion_presence'] == 'present') for sid in ids])
    return ids, X, y


def lock_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
