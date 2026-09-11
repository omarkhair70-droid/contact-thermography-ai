# Omar's Engineering Mandate — Solve the MumGuard Product Problem

Date: 2026-09-11
Applies to: `docs/ASTRA_MUMGUARD_RESEARCH_CHALLENGE_2026-09-11.md`

## What success means

The goal is not to return a conventional medical-ML answer such as "more labelled data is needed" and stop.

Assume that no additional proprietary labelled dataset is available to us right now beyond the current assets in the repository and the known 9-subject mouse cohort. Your job is to use deep research, engineering creativity, external evidence, pretrained/foundation models, modality physics, weak/self-supervision, public or legally usable data, transfer/domain adaptation, representation learning, simulation only where scientifically legitimate, and any other defensible technique to produce the strongest executable path toward the actual MumGuard human product.

The requested output is a **solution-first engineering program**, not a blocker report.

## Non-acceptable stopping point

A response whose practical conclusion is only one of the following is insufficient:

- "collect more data"
- "this cannot be done"
- "wait for human labels"
- "the cohort is too small"
- "build a larger balanced dataset"

Those facts may be constraints, but they are not the requested deliverable.

If labelled human data is ultimately required for clinical validation, that must be treated as a later validation dependency, not as permission to stop solving the engineering problem now.

## Required behavior under the current-data constraint

You must answer:

1. What can we build **now** that materially advances MumGuard toward human tumor-like vs no-tumor-like inference?
2. How can we extract more transferable signal from the existing mouse Contact-LCT data without pretending patches are independent subjects?
3. What external public/open/raw/unlabelled/weakly-labelled data or pretrained knowledge can be legally and scientifically leveraged?
4. Can the problem be reformulated so it needs far fewer disease labels—for example local anomaly modelling, within-subject contrast, self-supervision, prototype/metric learning, one-class modelling, multimodal/physics-informed representations, teacher-student methods, pseudo-labelling with safeguards, or other methods you derive yourself?
5. How do we bridge mouse -> human without simply training a mouse classifier and hoping it transfers?
6. What parts of the current system should become modality foundation components rather than disease-specific components?
7. What experiments can falsify bad ideas quickly and identify a winner?

Do not restrict yourself to the examples above. Discover a better formulation if one exists.

## Mandatory deliverable even if ideal data is absent

Return an executable model/research architecture that can be implemented with the assets available now. It may be a staged system rather than a single binary head, but it must be concrete enough for engineers to build.

For the top recommendation, specify:

- exact inputs
- preprocessing/segmentation
- representation model(s)
- trainable vs frozen components
- losses/objectives
- how mouse labels are used
- how local/internal controls are used
- how external or unlabelled data are used
- how domain/species effects are separated from tumor signal
- inference path
- uncertainty/abstention behavior
- exact scripts/artifacts to add to this repository
- evaluation possible now
- criteria for promotion or rejection

## Data acquisition is allowed only as a secondary output

If, after exhausting engineering alternatives, some new data is still necessary, do **not** stop there.

You must first provide the strongest current-data solution and then state only the **smallest irreducible additional evidence** needed to validate or calibrate it. Avoid arbitrary dataset targets such as "20/20" unless derived from an explicit learning curve or uncertainty calculation.

Separate these two questions:

- **Can we engineer a useful research system now?** — solve this now.
- **What evidence is later needed to prove clinical/generalization performance?** — specify separately.

## Product-level North Star

The desired engineering destination is not a mouse-demo classifier. It is a reusable MumGuard stack where an unseen human acquisition can ultimately be processed into a robust tumor-like vs no-tumor-like research decision, with the amount of required labelled human data minimized as aggressively as science allows.

The current mouse model and live application are assets, not sacred architecture. Preserve what transfers; replace what does not.

## Final instruction

Use your strongest research capability to find a path that a normal implementation team would not immediately see. Search deeply, compare formulations, run thought-through ablations, and return a ranked executable solution.

Do not give Omar a reason why the project cannot proceed. Give Omar the best defensible way to proceed **despite** the data constraint, and separately state what later evidence would be required to prove that it works on the intended human population.
