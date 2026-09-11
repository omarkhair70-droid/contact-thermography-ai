# Contact Thermography AI — Master Handoff

The previous RC0.5-era handoff is superseded.

## Canonical continuation files

Read these in order:

1. `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md` — full project history and scientific/product context.
2. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_LANES_M_P.md` — Lanes M/N/O/P history.
3. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_POST_PR33.md` — historical state after PR #33; its all-positive client-cohort assumption is superseded.
4. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_MIXED_CLIENT_COHORT.md` — **current source of truth**.

Repository: `omarkhair70-droid/contact-thermography-ai`

Canonical working branch: `integration`

## Current native-mouse rule

The client clarified that the nine already-supplied target-device mouse contact-LCT images include both tumor-bearing and normal/no-tumor animals.

Therefore the previous assumptions that all nine were tumor-bearing and that new controls were necessarily required are withdrawn.

The current blocker is only the exact per-image mapping:

`WA0030` ... `WA0038` -> `TUMOR_BEARING` or `HEALTHY`.

Until that map is recorded, the canonical nine remain non-trainable evidence and `lct-native-binary` remains `BLOCKED_ON_CLIENT_CLASS_MAP`.

Once the mapping is resolved, the cohort-preparation gate can promote the correctly labelled client rows into an internal-development manifest. The first result is `INTERNAL_RESEARCH_CROSS_VALIDATION`, not independent external validation.

The tiny-cohort engineering gate now permits a first exploratory baseline with at least 2 mapped subjects in each class. This is an engineering minimum, not scientific or clinical sample-size adequacy. Logistic Regression remains the minimum baseline; calibrated Linear SVM is used only when class counts support calibration.

Tumor-size/volume mapping remains useful for the separate tumor-burden experiment but is not required for the first binary classifier.

`lct-target-v1` remains profile-locked to `client-device-tlc-pending`. Do not invent a calibrated profile name or silently mix TLC/device/acquisition domains.

Historical/public contact-LCT research remains useful for external evidence and future expansion, but it is not the primary blocker if the client cohort itself supplies both classes.

`clinical_claim=NONE` remains mandatory.

## New-chat instruction

Use:

> Read the master continuation files, ending with `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_MIXED_CLIENT_COHORT.md`, in repo `omarkhair70-droid/contact-thermography-ai`. Inspect current `integration`. Do not redo merged lanes and do not assume the nine client mice are all positive. The client clarified that the supplied cohort includes both tumor-bearing and normal/no-tumor mice. Recover/record the exact WA0030-WA0038 class map, then build the internal-development manifest and run the target-native baseline. Preserve `client-device-tlc-pending` and `clinical_claim=NONE`.

Do not restart from the old parallel-lane plan or merge stale PR #7/#8; both are closed and superseded.
