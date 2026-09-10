# LCT Target Dataset v1

## Purpose

This is the first **target-modality contact-LCT dataset manifest** for the project. It exists to keep real contact-LCT evidence, client-device data, rights status, provenance, and train/test roles separate from the auxiliary infrared program.

`clinical_claim = NONE`

## Current contents

The manifest contains 11 target-modality subjects/cases:

- 2 human observed LCT case exemplars from the 2026 Malaysian review/cohort illustration:
  - one biopsy-confirmed invasive carcinoma case shown as a true-positive LCT example;
  - one biopsy-confirmed fibrocystic benign case shown as a true-negative LCT example.
- 9 private client-device mouse images. The client stated that all mice were experimentally tumor-injected. Their tumor-size mapping is still pending.

The two human rows are `REFERENCE_ONLY`. The nine client rows are `FROZEN_TARGET_EVAL` and must not be used as a source of synthetic negative labels.

## Why this is not yet a binary training set

A binary classifier needs enough **eligible positive and eligible negative target-domain examples from a coherent species/device domain**. Dataset v1 deliberately has zero `train_eligible=true` rows because:

1. the two human figures are isolated illustrated exemplars, not a representative dataset;
2. the client mouse cohort is positive-only;
3. mixing human and mouse rows into one binary train/test pool would create a species shortcut instead of a tumor classifier;
4. mixing TLC formulations/devices without profile control would create another shortcut;
5. rights and raw-image availability differ by source.

The validator therefore reports `binary_training_ready=false` until at least five eligible positives and five eligible negatives from one species are available. Five-per-class is only a minimal engineering gate for a provisional research fit, not a claim of adequate scientific sample size.

## Data-use rules

- Preserve `tlc_profile_id` and `device_profile_id` for every contact-LCT row.
- Never relabel the nine client mice as healthy/no-tumor.
- Never put one subject in both train and test.
- Do not use literature-derived hypothetical failure scenarios as ground-truth images.
- Do not train on rights-restricted material unless permission is obtained.
- Keep auxiliary infrared models behind an OOD/abstain gate when probing contact-LCT images.
- Do not convert a model score into a clinical diagnosis or cancer probability.

## Immediate acquisition priority

The highest-value missing data is a **same-domain negative cohort** for the client setup: contact-LCT images from mice that were not tumor-injected, or preferably pre-injection/baseline images from the same animals under the same acquisition protocol. Those images would be materially more useful for a client-native binary model than adding many unrelated infrared images.

In parallel, continue the public-data hunt for contact-LCT case images with case-level pathology labels and clear rights. Each newly discovered source must enter `data/lct_target_source_registry.csv` before its images are admitted to this manifest.

## Next model gate

Once a coherent target-domain positive/negative pool exists:

1. extract deterministic TLC response/morphology/QC features;
2. extract frozen DINOv2 embeddings;
3. run subject-level Logistic/SVM baselines;
4. compare TLC-only, DINO-only, and fused feature blocks;
5. hold out a frozen target test split;
6. only then consider fine-tuning a visual backbone if sample size and validation justify it.

The desired research output remains `TUMOR_LIKE`, `NO_TUMOR_LIKE`, or `ABSTAIN_OOD`, with provenance and evidence attached.
