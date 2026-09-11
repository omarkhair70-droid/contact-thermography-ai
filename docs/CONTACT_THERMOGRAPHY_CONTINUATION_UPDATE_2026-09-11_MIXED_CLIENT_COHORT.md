# Contact Thermography AI — mixed client cohort clarification

Date: 2026-09-11

Repository: `omarkhair70-droid/contact-thermography-ai`

This addendum supersedes every earlier statement that the nine client mouse images are all tumor-bearing or that new control images are necessarily required before the first native binary experiment.

## Client clarification

The client clarified that the contact-LCT mouse images already supplied include both:

- tumor-bearing mice; and
- normal / no-tumor mice.

The exact `WA0030` through `WA0038` image-to-class mapping is not yet recorded in the repository.

The earlier all-positive interpretation came from treating a client statement about experimentally injected tumor animals as if it described every supplied image. That interpretation is now withdrawn.

## Immediate data rule

The nine target-device images are now treated as a **mixed-class cohort with class mapping pending**.

Do not:

- label all nine as `TUMOR_BEARING`;
- request replacement controls solely because the old manifest said positive-only;
- infer the missing labels from image appearance or model output.

Do:

- obtain or recover the exact mapping `WA0030` ... `WA0038` -> `TUMOR_BEARING` or `HEALTHY`;
- record it in `data/client_mouse_ground_truth_template.csv`;
- then build the internal-development manifest and train on the client cohort itself.

The original DINOv2 Kaggle run intentionally recorded `labels_seen=false`; therefore it cannot recover the missing class map from its outputs.

## Updated engineering gate

The first target-native binary model no longer has a hard-coded five-per-class requirement. For a tiny exploratory internal baseline, the software gate requires at least two mapped subjects in each class. This is an engineering minimum, not a sample-size adequacy claim.

With very small class counts:

- Logistic Regression remains available;
- calibrated Linear SVM is used only when class counts support its inner calibration folds;
- metrics must remain explicitly `INTERNAL_RESEARCH_CROSS_VALIDATION`;
- `clinical_claim=NONE` remains mandatory.

If the resolved nine-image mapping gives enough examples of both classes, **no additional mouse images are required to run the first native Tumor-like / No-tumor-like research baseline**.

## Current blocker

The current blocker is now only:

**exact per-image class mapping for the already-supplied client mouse cohort.**

Tumor-size/volume mapping is still useful for the separate tumor-burden experiment but is not required to fit the first binary classifier.

Public-data / Deep Research work remains useful for external evidence and future expansion, but it is no longer the primary blocker if the client cohort itself contains both classes.

## Runtime status

Until the class map is resolved and a model is trained, `lct-native-binary` remains fail-closed with no emitted research class. Its blocked status is `BLOCKED_ON_CLIENT_CLASS_MAP`.

The target feature contract remains `lct-target-v1` and remains profile-locked to `client-device-tlc-pending` until the real TLC formulation/calibration is documented.
