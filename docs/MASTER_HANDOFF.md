# Contact Thermography AI — Master Handoff

The previous RC0.5-era handoff is superseded.

## Canonical continuation files

Read these in order:

1. `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md` — full project history and scientific/product context.
2. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_LANES_M_P.md` — Lanes M/N/O/P history.
3. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_POST_PR33.md` — historical state after PR #33; its all-positive client-cohort assumption is superseded.
4. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_MIXED_CLIENT_COHORT.md` — historical mixed-cohort state before the exact map was resolved.
5. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_EXP8V1.md` — **current source of truth**: exact 8/1 map, experimental native model and Coolify continuation.

Repository: `omarkhair70-droid/contact-thermography-ai`

Canonical deployment branch after reviewed/green merge: `integration`

## Current native-mouse rule

The exact supplied client mouse map is now resolved:

- `WA0030` through `WA0037` = `TUMOR_BEARING`
- `WA0038` = `HEALTHY`

The current cohort is therefore 8 positive / 1 negative.

The product target is tumor presence/absence only: `TUMOR_LIKE` vs `NO_TUMOR_LIKE`. Tumor-size estimation is out of scope. The client clarified that visible TLC response can depend on superficial/deeper presentation, so apparent response extent/intensity must not be interpreted as tumor size.

The canonical native trainer remains the path for defensible subject-level internal cross-validation when class counts support it. With only one negative, proper binary CV/specificity estimation is not possible.

A separate explicit internal demo lane may fit the 8/1 cohort so end-to-end product inference can be exercised now. That artifact must remain marked `ACTIVE_RESEARCH`, `INTERNAL_RESEARCH_DEMO_ONLY`, `NOT_VALIDATED_SINGLE_NEGATIVE` and `clinical_claim=NONE`.

Additional genuine same-domain no-tumor mice remain the highest-value data addition. Four more independent controls reach the current five-negative engineering target. Multiple images/augmentations of the same negative animal do not count as additional subjects.

`lct-target-v1` remains profile-locked to `client-device-tlc-pending`. Do not invent a calibrated profile name or silently mix TLC/device/acquisition domains.

Historical/public contact-LCT research remains reference/auxiliary evidence unless its domain, subject labels and rights meet the target intake contract.

## New-chat instruction

Use:

> Read the master continuation files, ending with `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_EXP8V1.md`, in repo `omarkhair70-droid/contact-thermography-ai`. Inspect current `integration` and any open experimental 8-vs-1 PR. The exact client map is WA0030-WA0037 tumor-bearing and WA0038 healthy. Preserve `client-device-tlc-pending`, `clinical_claim=NONE`, and the distinction between the internal demo fit and real subject-level validation. Finish green CI/merge/Coolify smoke validation, then prioritize additional genuine same-domain no-tumor controls.

Do not restart from the old parallel-lane plan or merge stale PR #7/#8; both are closed and superseded.
