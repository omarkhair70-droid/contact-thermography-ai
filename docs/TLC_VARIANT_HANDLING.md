# TLC Formulation / Device Variant Handling

## New client fact

The client stated that images produced by their own device will look different from the publication/reference images because the **TLC material/formulation itself is different**.

This matters and is now a first-class product requirement.

## What changes when the TLC formulation changes

A contact liquid-crystal thermogram is not an infrared-camera frame. The visible colour response depends on the thermochromic liquid-crystal formulation and its operating/calibration range. Therefore a Hue/RGB pattern learned or thresholded on one TLC formulation must not be assumed to have the same absolute thermal meaning on another formulation.

The following are **profile-dependent** and must be calibrated/normalized per device/TLC formulation:

- absolute Hue/RGB/Lab distribution;
- colour-to-temperature mapping, if an absolute temperature map is ever produced;
- active-response colour range;
- saturation / brightness acceptance ranges;
- colour thresholds currently used by the reference signal mask;
- any colour-derived model feature whose scale shifts between formulations.

## What can remain formulation-agnostic or more transferable

The following may still transfer better, subject to validation on client-device images:

- circular plate detection / cropping;
- image-quality checks such as focus and clipping;
- connected-region geometry;
- focal / linear / branched / diffuse morphology after profile-specific signal segmentation;
- normalized spatial location / area ratios;
- LEFT vs RIGHT comparison when both sides are captured with the same TLC/device/profile;
- learned visual representation, provided domain shift is measured and the reference model is not treated as calibrated clinical output.

## Architecture rule

Do **not** hard-code one global colour interpretation for all contact thermography inputs.

Every exam must carry a `tlc_profile_id` (and optionally a `device_profile_id`). The analysis pipeline should resolve that profile before signal segmentation / colour interpretation.

Recommended pipeline:

`Raw image -> device/TLC profile -> QC -> normalization -> active-response segmentation -> morphology -> DINOv2 -> bilateral comparison -> report`

## Current baseline status

The current 26 reference plates remain useful as Dataset Zero for software, morphology, representation and workflow development, but their absolute colour response must be treated as belonging to `reference-publication-unknown`.

The client device should use `client-device-tlc-pending` until real device images and the exact TLC formulation/calibration information are available.

## First client-device calibration pass

When client-device images arrive:

1. ingest them as a separate domain/profile; do not mix silently with Dataset Zero;
2. inspect Hue/Saturation/Value and Lab distributions;
3. update the profile-specific active-response mask;
4. compare DINOv2 embedding distribution against the publication reference domain;
5. test same-exam LEFT/RIGHT stability;
6. if temperature interpretation is required, obtain or derive the formulation-specific colour/temperature calibration curve;
7. version the profile and keep model/report provenance with every examination.

## Clinical boundary

A change in TLC formulation is a **domain shift**. Current reference anomaly scores must not be treated as calibrated cancer-risk scores on client-device images. The product may still provide research image/morphology/asymmetry analysis while formulation-specific calibration and later clinical validation are completed.
