# MumGuard local evidence research kit

Additive research prototype for `omarkhair70-droid/contact-thermography-ai`.
Inspected integration: `c417da1f15146f6ab15974a0b7fbf399cb97842b`.
Clinical claim: `NONE`. No model registry activation, deployment or human disease decision.

The full program is in `docs/MUMGUARD_EXECUTABLE_RESEARCH_PROGRAM.md` in the kit.
The kit's report and candidate implementations are a research handoff, not evidence that
the proposed local method outperforms the saved mouse classifier.

## Install and verify

Use an isolated environment or the repository's existing environment. The core code
requires NumPy and Pillow. It was tested with Python 3.12.10, NumPy 2.4.6, Pillow 12.3.0.
The optional extractor additionally requires a compatible PyTorch installation and
the DINOv2 dependencies/weights. Reuse the repo's `requirements-vision.txt` setup.

From the repository root:

```powershell
python -m unittest discover -s research/mumguard -p 'test_*.py' -v
python research/mumguard/audit_current_evidence.py --out research/mumguard/current_evidence_audit.json
```

The evidence audit reads three existing committed files, which are not duplicated in
this kit. Apply the kit to the inspected repository, or use `--repo PATH`.

## Extract a local evidence map from an existing acquisition

First mount the actual original JPEGs from existing project storage. Create a contact
mask independently of the color response or disease label. White pixels mean confirmed
contact support; black means outside/unknown. An optional exclusion mask marks artifacts
white. Masks must match the EXIF-oriented original image dimensions. Do not substitute
the current application's letterbox frame mask or use non-response as normal tissue.

Replace the example data paths below with the actual mounted asset paths:

```powershell
python research/mumguard/local_contrast.py --image private/IMG-20260910-WA0030.jpg --contact-mask private/WA0030-contact.png --subject-id CLIENT-MOUSE-0030 --species mouse --tlc-profile-id client-device-tlc-pending --device-profile-id client-device-unknown --acquisition-id WA0030 --out work/mumguard/0030
```

Optional DINO descriptors:

```powershell
python research/mumguard/extract_tokens.py --normalized-image work/mumguard/0030/normalized.png --out work/mumguard/0030/tokens.npz --device cpu
python research/mumguard/local_contrast.py --image private/IMG-20260910-WA0030.jpg --contact-mask private/WA0030-contact.png --subject-id CLIENT-MOUSE-0030 --species mouse --tlc-profile-id client-device-tlc-pending --device-profile-id client-device-unknown --acquisition-id WA0030 --tokens-npz work/mumguard/0030/tokens.npz --out work/mumguard/0030
```

Keep all image/mask/provenance arguments identical on the second extraction call. The
normalized pixel hash and backbone revision are checked. Do not reuse a token cache
after changing source, orientation, resolution or normalization. Patch tokens require
area coverage in the disjoint region rather than center-point membership. Full-image
ViT tokens still contain global context, so crop-context claims require a later ablation
with credible contact ROI/ring pairs.

Outputs:

- `evidence.json`: named local features, coverage, quality reasons, provenance and null binary class.
- `maps.npz`: float local-contrast map and support-count map. Zero without support means unmeasured.
- `normalized.png`, `contact.png`, `local_contrast.png`: aligned display images.

The physical score is a provisional dimensionless photometric contrast, not a learned
tumor score. DINO evidence is kept as a separate block rather than silently fused with
an arbitrary scale. Dark contact is missing/ambiguous evidence. There is no clinical
threshold, absolute-temperature conversion, lesion size estimate or location validation.

## Run subject-level B/F experiments

Extract evidence for all nine original subjects. Create a CSV with this exact header:

```csv
subject_id,label,species,tlc_profile_id,device_profile_id,evidence_path
CLIENT-MOUSE-0030,TUMOR_BEARING,mouse,client-device-tlc-pending,client-device-unknown,0030/evidence.json
```

Add all eight known positives and the single `HEALTHY` subject `CLIENT-MOUSE-0038`.
Evidence paths are relative to the CSV's directory unless absolute. The example row
does not constitute a complete manifest. Repeats must be real separate acquisitions;
the script rejects duplicate original hashes and pools genuine repeats per subject.

```powershell
python research/mumguard/fit_subject_heads.py --manifest work/mumguard/subjects.csv --out work/mumguard/subject_heads.json --acknowledge-development-only
```

This compares physical contrast, detailed photometric features, and DINO/fusion when
every subject has DINO descriptors. It uses eight leave-one-positive-out folds with
the negative in training. It does not estimate independent specificity or select a
human threshold. Missing DINO skips the DINO-dependent experiments; no imputations.
Missing local evidence stops fitting so an unusable acquisition cannot become a negative.
The script is restricted to the known 8/1 mouse cohort, one device/profile, explicit labels.
It is not the full future ablation framework or the canonical native trainer.

## Implementation execution completed after the research handoff

The nine original private JPEGs were mounted outside git, hash-verified against
`data/mumguard_acquisitions_v1.jsonl`, and processed with the pinned CPU DINOv2
checkpoint. Private pixels, maps and token tensors remain outside git. The committed
`artifacts/mumguard-local-v1/` directory contains hashes, per-subject numerical evidence,
folds, perturbations, permutation diagnostics, the frozen protocols and model card.

No independent contact annotations or genuine repeat pairs exist. E1 and strict
contact-versus-background tests are recorded as blocked. Conservative chromatic support
was usable in four subjects; unknown-contact full-field local photometry ran on all nine
as an explicit setup-sensitive diagnostic. Missing local/DINO channels were not imputed.
Whole-image DINO baselines were retained, including exact reproduction of the saved fit.

The integrated candidate is human-first and inactive. Uploads persist source-level local
maps and provenance, but missing contact yields unusable-measurement abstention and no
disease head. Human intake, adaptation, small-head and calibration contracts are present;
their split template is intentionally unlocked until real target acquisitions exist.
