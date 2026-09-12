"""Validate private mounted sources against the immutable development ledger."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps


def validate(manifest, image_root):
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if sorted(r['subject_id'] for r in rows) != [f'CLIENT-MOUSE-{i:04d}' for i in range(30, 39)]:
        raise ValueError('expected exactly nine original development subjects')
    hashes = set()
    for r in rows:
        if r['species'] != 'mouse' or r['modality'] != 'contact-LCT' or r['split'] != 'development':
            raise ValueError('species/modality/split mismatch')
        expected = 'absent' if r['subject_id'].endswith('0038') else 'present'
        if r['lesion_presence'] != expected or not r['label_provenance']:
            raise ValueError('resolved endpoint map mismatch')
        path = Path(image_root)/r['filename']
        if path.resolve().parent != Path(image_root).resolve():
            raise ValueError('image path escapes mount')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != r['source_sha256'] or digest in hashes:
            raise ValueError('source hash mismatch or duplicate original')
        hashes.add(digest)
        with Image.open(path) as image:
            if list(ImageOps.exif_transpose(image).size) != r['source_dimensions']:
                raise ValueError('oriented dimensions mismatch')
        if r['contact_certainty'] != 'UNKNOWN' and not r.get('contact_annotation_sha256'):
            raise ValueError('confirmed contact requires independent annotation provenance')
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/mumguard_acquisitions_v1.jsonl'))
    p.add_argument('--image-root', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps({'validated_originals': len(validate(a.manifest, a.image_root)), 'clinical_claim': 'NONE'}))
