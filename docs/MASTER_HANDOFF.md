# Contact Thermography AI — Master Handoff

The previous RC0.5-era handoff is superseded.

## Canonical continuation files

Read these in order:

1. `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md` — full project history and scientific/product context.
2. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_LANES_M_P.md` — Lanes M/N/O/P history.
3. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_POST_PR33.md` — **current source of truth** after the client clarified that the existing nine mice are the available tumor-bearing cohort.

Repository: `omarkhair70-droid/contact-thermography-ai`

Canonical working branch: `integration`

## Current native-mouse rule

PR #33 superseded the old requirement for five new tumor-bearing mice.

The original nine client mouse subjects (`CLIENT-MOUSE-0030` through `CLIENT-MOUSE-0038`) are documented tumor-bearing target-device positives. Their canonical source rows remain non-trainable evidence, but the cohort-preparation gate may promote copies into a generated internal-development manifest once genuine same-domain controls pass validation.

The current minimum engineering blocker for a first native mouse binary baseline is now:

- the existing 9 client tumor-bearing development positives; plus
- at least 5 genuine same-domain control/negative mice.

More controls are strongly preferred. The first binary result must be described as `INTERNAL_RESEARCH_CROSS_VALIDATION`, **not independent external validation**.

`lct-target-v1` remains profile-locked to `client-device-tlc-pending`. Do not invent a calibrated profile name or silently mix TLC/device/acquisition domains.

Historical human contact-LCT sources now document genuine normal/control evidence (including Davison 1972 and Ewing 1973), but no verified downloadable same-domain healthy mouse dataset has been found. Human contact-LCT or infrared negatives must not be relabelled as mouse controls.

`clinical_claim=NONE` remains mandatory.

## New-chat instruction

Use:

> Read `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md`, `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_LANES_M_P.md`, and then `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_POST_PR33.md` in repo `omarkhair70-droid/contact-thermography-ai`. Inspect current `integration`. Do not redo merged lanes or revert to the old 5-new-positive rule. The existing nine client tumor-bearing mice may be used only through the internal-development promotion gate; genuine same-domain controls are the current binary-training blocker. Preserve TLC/device provenance and `clinical_claim=NONE`.

Do not restart from the old parallel-lane plan or blindly merge stale PR #7/#8; their accepted work was superseded by later integration PRs.
