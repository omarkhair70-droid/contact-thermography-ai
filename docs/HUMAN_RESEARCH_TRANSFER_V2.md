# MumGuard human research transfer v0.2

## Why v0.2 exists

The first DMR-IR transfer run proved that the source pipeline was reproducible on 56 human subjects, but the v0.1 contract was intentionally not activated. Its features removed additive temperature offset only and flattened each thermal field into a distribution, which discarded the spatial morphology that MumGuard is designed to reason about.

v0.2 changes the transfer question from "do the source and target share the same numeric thermal scale?" to "do they share dimensionless bilateral spatial morphology after a common within-exam normalization?"

## Shared contract

For each bilateral exam, LEFT and RIGHT are normalized together using pooled median and pooled P90-P10 spread. Features are then extracted from the normalized fields. Therefore the contract is invariant to a joint positive affine transform `x -> a*x+b` (`a > 0`).

The feature vector includes:

- strongest side hotspot peak after normalization;
- local positive contrast around focal structure;
- spatial gradient strength;
- hotspot connected-component coherence and normalized location;
- LEFT/RIGHT differences in mean, spread, upper-tail response, hotspot strength, local contrast, gradient and coherence;
- mirrored bilateral pattern MAE and correlation distance.

This deliberately excludes absolute Celsius and never assumes that one DMR-IR degree maps to any fixed MumGuard Contact-TLC value.

## Source construction

DMR-IR segmented thermal matrices are preserved as 2-D fields. Repeated source records are resized with support-aware interpolation and median-aggregated per side. One subject-level LEFT/RIGHT pair is then converted to the v0.2 feature vector before cross-validation.

## Training and abstention

The candidate trainer compares balanced Logistic Regression with a calibrated linear SVM using subject-level stratified out-of-fold predictions. Balanced accuracy is the primary selection criterion and Brier score breaks ties. A nearest-source OOD gate is fitted in standardized feature space.

Artifacts remain `RESEARCH_TRANSFER_CANDIDATE` after training. They are not activated automatically.

## Activation boundary

No live human decision semantics change in this lane. Before any research activation:

1. regenerate the v0.2 source artifact from the original DMR-IR archive;
2. review subject count, extraction failures, OOF balanced accuracy, sensitivity, specificity, AUROC/AUPRC and Brier score;
3. verify the OOD gate on MumGuard synthetic/control sessions;
4. confirm symmetric control -> low concern / not suspicious, focal hotspot -> higher concern / suspicious, and low-quality or OOD -> inconclusive;
5. keep `clinical_claim=NONE` and keep clinical cancer probability unavailable until outcome-linked MumGuard calibration exists.

Mouse experiments remain development evidence only and are never used as human supervision in this transfer head.
