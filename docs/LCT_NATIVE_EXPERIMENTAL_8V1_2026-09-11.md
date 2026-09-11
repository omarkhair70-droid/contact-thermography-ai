# Experimental target-native 8-vs-1 classifier — 2026-09-11

## Scope

This lane activates an **internal research/demo** binary classifier for the current client mouse contact-LCT domain while preserving `clinical_claim=NONE`.

The client class map relayed by the project owner is:

- `CLIENT-MOUSE-0030` through `CLIENT-MOUSE-0037`: `TUMOR_BEARING`
- `CLIENT-MOUSE-0038`: `HEALTHY`

The intended target is **tumor presence/absence only** (`TUMOR_LIKE` vs `NO_TUMOR_LIKE`). Tumor-size estimation is explicitly out of scope. The client clarified that visible TLC response can depend on superficial/deeper presentation, so visual response extent/intensity must not be interpreted as tumor size.

## Why this is experimental

The supplied cohort has 8 tumor-bearing subjects and only 1 no-tumor subject. With one negative subject there is no defensible subject-level binary cross-validation: putting the only negative in a test fold leaves no negative example for training, while keeping it in training leaves no independent negative for specificity estimation.

Therefore:

- the canonical native trainer remains strict and is **not weakened**;
- this lane uses a separate, explicit single-negative demo trainer;
- the output is not an independent validation result;
- specificity, AUROC, sensitivity/specificity trade-offs and clinical performance are not claimed;
- four additional genuine same-domain healthy/control mouse subjects remain the next engineering data priority for the current five-negative gate.

## Model

Artifact: `artifacts/lct_native_exp8v1_dino.json`

Version: `0.1.0-exp8v1-dino`

Model:

- Logistic Regression
- official frozen `dinov2_vits14` embeddings
- 384-dimensional normalized DINO block
- `class_weight=balanced`
- `C=1.0`
- `solver=liblinear`
- seed `20260911`
- threshold `0.5`

The artifact is portable JSON rather than a pickle/joblib object. It stores the linear coefficients and intercept directly and remains locked to:

- feature contract `lct-target-v1`
- TLC profile `client-device-tlc-pending`
- mouse target domain
- `clinical_claim=NONE`

Although the runtime receives the 417-feature target contract, this experimental version consumes only the 384-dimensional DINO block at offset 33. The live bridge is explicitly version-locked to this artifact so a future fused model cannot silently inherit zero-filled LCT/QC feature groups.

## Reproducible training

Use the derived Lane 5 DINO embeddings plus their source-image index and the client ground-truth CSV:

```bash
python scripts/train_lct_single_negative_demo.py \
  --embedding-npy client_device_dinov2_embeddings.npy \
  --embedding-index-csv client_device_dinov2_summary.csv \
  --ground-truth-csv data/client_mouse_ground_truth_template.csv \
  --output-json artifacts/lct_native_exp8v1_dino.json \
  --acknowledge-single-negative
```

The acknowledgement flag is intentionally required so this lane cannot be mistaken for the normal validation trainer.

## Current fit check

On the same nine subjects used to fit the model, the threshold separates the supplied labels. These are **resubstitution scores only** and must not be presented as held-out accuracy:

| Subject | Label | Model score |
|---|---|---:|
| 0030 | TUMOR_BEARING | 0.570133 |
| 0031 | TUMOR_BEARING | 0.568223 |
| 0032 | TUMOR_BEARING | 0.598274 |
| 0033 | TUMOR_BEARING | 0.619351 |
| 0034 | TUMOR_BEARING | 0.567608 |
| 0035 | TUMOR_BEARING | 0.555280 |
| 0036 | TUMOR_BEARING | 0.563967 |
| 0037 | TUMOR_BEARING | 0.596718 |
| 0038 | HEALTHY | 0.400324 |

`model_score` is a research classifier score. It is **not** a cancer probability, diagnosis, tumor-size estimate or independently validated risk score.

## Live runtime

`app/services/native_classifier_runtime.py` fail-closes unless the registry is `ACTIVE_RESEARCH`, non-clinical, profile-compatible and points to a real artifact.

`app/services/dinov2_service.py` attaches `native_binary_research` to client-profile plate results. Its payload includes:

- `research_binary_class`
- `model_score`
- `decision_threshold`
- `validation_status=NOT_VALIDATED_SINGLE_NEGATIVE`
- training subject counts
- non-clinical semantics

Publication-reference TLC profiles do not receive this target-native classification.

## Next upgrade

When additional genuine same-device / same-domain no-tumor mice become available, rebuild the full 417-feature subject table and return to `scripts/train_lct_native_binary.py` for subject-level cross-validation. The experimental 8-vs-1 artifact should then be retired rather than treated as cumulative validation evidence.
