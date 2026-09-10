# DMR-IR auxiliary baseline results — 2026-09-10

## Scope

This is an **auxiliary infrared experiment**, not validation of the client contact-LCT device. The run used the original UFF DMR-IR archive, 56 subjects total (37 CANCER, 19 HEALTHY), 1,522 usable image/matrix records, official DINOv2 ViT-S/14 embeddings, and eight explicit temperature-summary features. Multiple frames/views were aggregated to one vector per subject before cross-validation.

`clinical_claim = NONE`

## Baseline OOF results

Five-fold subject-level out-of-fold evaluation:

| Model | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC | Brier | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| Logistic, fused DINO + thermal | 0.840 | 0.838 | 0.842 | 0.909 | 0.955 | 0.138 | TN 16 / FP 3 / FN 6 / TP 31 |
| Calibrated linear SVM, fused | 0.736 | 0.946 | 0.526 | 0.903 | 0.955 | 0.148 | TN 10 / FP 9 / FN 2 / TP 35 |

The Logistic baseline is the preferred fused candidate because it has much better specificity and balanced accuracy while preserving useful sensitivity.

## Ablation review

The Kaggle artifact stores 384 DINO features followed by eight explicit thermal features (`mean`, `std`, `p10`, `p50`, `p90`, `p90-p10`, `min`, `max`). Re-running the same subject-level folds over feature blocks showed:

| Logistic feature block | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| DINO only | 0.760 | 0.730 | 0.789 | 0.876 | 0.932 | 0.168 |
| Thermal only | **0.906** | **0.865** | **0.947** | **0.923** | **0.970** | **0.095** |
| DINO + thermal | 0.840 | 0.838 | 0.842 | 0.909 | 0.955 | 0.138 |

Across 20 alternate stratified five-fold seeds, the thermal-only Logistic result remained comparatively stable: mean balanced accuracy 0.906 ± 0.024, AUROC 0.931 ± 0.004, AUPRC 0.973 ± 0.001, Brier 0.092 ± 0.003.

### Important interpretation

The strongest signal in DMR-IR currently comes from absolute/summary temperature statistics, not from the generic DINO representation. Subject-level p90, p10, median, and mean temperatures each individually separate the labels strongly in this dataset. This is useful, but it also raises a **dataset/protocol confounding risk**: the result could reflect acquisition-condition or cohort temperature differences in addition to disease biology. It must therefore be stress-tested before any transfer claim.

The result does **not** mean an 0.92-AUROC cancer model exists for the client device. The client images are contact-LCT mouse images with a large measured domain shift from the publication reference set, whereas DMR-IR is human infrared thermography.

## Next required experiments

1. Reproduce the ablation directly from the stored feature table.
2. Stress-test temperature-only performance with acquisition-aware covariates where recoverable and with normalization variants that remove absolute temperature offset.
3. Compare static vs dynamic record subsets and left/right aggregation.
4. Keep DMR-IR as an auxiliary representation/data source only.
5. Continue the target-domain contact-LCT data hunt for labeled tumor/control images, especially mouse/contact-LCT material.
6. Probe the frozen auxiliary heads on client-device images only with an explicit OOD/abstain flag; do not treat those outputs as validation.
7. Keep the nine client mouse images frozen as target-domain tumor-bearing data and tumor-burden evaluation, not binary train/test evidence.

## Decision

The DMR-IR run is scientifically useful and the training harness is working. The next model step is **not blind DINO fine-tuning**. It is confound-aware thermal baseline validation plus target-domain LCT data acquisition and transfer experiments.
