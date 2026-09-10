# Parallel Agent Launch

All workers use repository `omarkhair70-droid/contact-thermography-ai`.

Rules for every worker:
- read `docs/MASTER_HANDOFF.md` first;
- work only on the assigned branch;
- do not create a new repository or independent ZIP/project copy;
- do not merge directly to `main`;
- open a PR to `integration` when the lane is green;
- preserve current non-clinical semantics;
- treat client-device TLC as a separate profile/domain from publication Dataset Zero.

## Worker A — Live DINOv2 / AI runtime

Branch: `feat/live-dinov2`
Issue: #1

Prompt:

> Work in repo `omarkhair70-droid/contact-thermography-ai` on branch `feat/live-dinov2`. Read `docs/MASTER_HANDOFF.md`, `docs/TLC_VARIANT_HANDLING.md`, `docs/LANE_LIVE_DINOV2.md`, and issue #1 before coding. Implement the lane fully, run regression tests, do not redesign the UI or deployment, and do not turn reference scores into cancer probabilities. Preserve `tlc_profile_id` / `device_profile_id`. When green, commit and open a PR to `integration`; do not merge it yourself.

## Worker B — Client demo UI

Branch: `feat/product-ui`
Issue: #2

Prompt:

> Work in repo `omarkhair70-droid/contact-thermography-ai` on branch `feat/product-ui`. Read `docs/MASTER_HANDOFF.md`, `docs/TLC_VARIANT_HANDLING.md`, `docs/LANE_PRODUCT_UI.md`, and issue #2. Turn the technical dashboard into a polished client-demo examination workflow today. Keep backend/model contracts compatible. Include a simple TLC/device profile field/default so provenance is preserved, but never imply that different TLC formulations share one calibrated colour scale. Run the relevant tests, commit, and open a PR to `integration`; do not merge it yourself.

## Worker C — Oracle staging

Branch: `feat/oracle-deploy`
Issue: #3

Prompt:

> Work in repo `omarkhair70-droid/contact-thermography-ai` on branch `feat/oracle-deploy`. Read `docs/MASTER_HANDOFF.md`, `docs/LANE_ORACLE_DEPLOY.md`, and issue #3. Produce a repeatable Oracle staging deployment with Docker, PostgreSQL-ready persistence, durable generated/uploaded image storage abstraction, environment-only secrets, healthcheck/restart policy, HTTPS/reverse-proxy readiness, and rollback docs. Do not change AI semantics or redesign the UI. Run deployment/smoke validation that is possible from the repo, commit, and open a PR to `integration`; do not merge it yourself.

## Worker D — QA / hardening

Branch: `feat/qa-hardening`
Issue: #4

Prompt:

> Work in repo `omarkhair70-droid/contact-thermography-ai` on branch `feat/qa-hardening`. Read `docs/MASTER_HANDOFF.md`, `docs/TLC_VARIANT_HANDLING.md`, `docs/LANE_QA_HARDENING.md`, and issue #4. Build the regression and hardening lane: health/upload/history/report/DINO endpoints, corrupt and empty inputs, multi-plate extraction, LEFT/RIGHT pairing, NaN/Inf JSON safety, TLC-profile provenance/domain-shift checks, and explicit assertions that no endpoint returns a current cancer probability/diagnostic claim. Add a demo smoke checklist. Commit and open a PR to `integration`; do not merge it yourself.

## Integration owner

The lead/integration chat reviews all PRs into `integration`, resolves cross-lane conflicts, runs the final end-to-end smoke, then promotes a single demo candidate to `main`.
