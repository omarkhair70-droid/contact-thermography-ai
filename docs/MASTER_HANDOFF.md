# Contact Thermography AI — Master Handoff

The previous RC0.5-era handoff is superseded.

## Canonical continuation files

Read these in order:

1. `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md` — full project history and scientific/product context.
2. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_LANES_M_P.md` — latest state after Lanes M/N/O/P and the current data-acquisition blocker.

Repository: `omarkhair70-droid/contact-thermography-ai`

Canonical working branch: `integration`

The latest continuation update records that the open-license reference pack, external-data intake gate, USM access packet, and same-device mouse native-cohort acquisition/readiness gate are now integrated. The original nine client mouse positives remain `FROZEN_TARGET_EVAL` and must not be reused as training positives. The current minimum engineering gate for a first native mouse baseline requires at least 5 **new** tumor-bearing trainable subjects plus 5 genuine controls in a coherent target domain; larger balanced cohorts are preferred.

`lct-target-v1` remains profile-locked to `client-device-tlc-pending`. Do not invent a calibrated profile name or silently mix TLC/device/acquisition domains.

`clinical_claim=NONE` remains mandatory.

## New-chat instruction

Use:

> Read `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md` and then `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_LANES_M_P.md` in repo `omarkhair70-droid/contact-thermography-ai`. Inspect current `integration`. Do not redo merged lanes, do not synthesize negatives, and do not reuse the frozen nine client positives for training. Continue from target-domain data acquisition/readiness and preserve TLC/device provenance plus `clinical_claim=NONE`.

Do not restart from the old parallel-lane plan or blindly merge stale PR #7/#8; their accepted work was superseded by later integration PRs.
