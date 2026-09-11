# Astra Research Challenge — MumGuard Mouse → Human Contact-Thermography AI

Date: 2026-09-11
Repository: `omarkhair70-droid/contact-thermography-ai`
Canonical product branch: `integration`
Research branch: `research/astra-mumguard-transfer-20260911`

## Your role

Act as the lead ML research scientist and research engineer for this problem. Do not behave like a coding assistant that merely implements a suggested architecture. Your job is to discover the strongest scientifically defensible and practically executable strategy, test competing hypotheses, reject weak ideas, and define the shortest credible path from the current preclinical evidence to the real MumGuard human product.

We can implement the experiments you design. Be ambitious, but do not manufacture evidence, labels, independent subjects, or validation.

## Real product target

MumGuard is a human breast-health product. Public MumGuard material describes a handheld thermographic device that measures breast temperature distribution, pairs with the MumGuard Care app, and sends collected data to an AI system intended to identify abnormal/lump-like breast findings. The intended end user is a human woman, not a mouse.

Important distinction: the current private research images supplied to this repository are **Contact Liquid Crystal Thermography (Contact-LCT/TLC) mouse images**. Public MumGuard material uses broader thermographic language and a “thermographic matrix”; do not assume every public hardware detail is identical to the private TLC acquisition stack unless evidence supports that mapping.

Final research objective:

`unseen human MumGuard thermography/contact-thermography acquisition -> robust tumor-like vs no-tumor-like research inference`

Longer-term clinical validation is outside the present engineering proof and must remain separate from internal research performance.

## Current proprietary preclinical cohort

Exactly 9 independent mouse subjects are currently labelled:

- `CLIENT-MOUSE-0030` through `CLIENT-MOUSE-0037` = `TUMOR_BEARING` (8 subjects)
- `CLIENT-MOUSE-0038` = `HEALTHY / NO_TUMOR` (1 subject)

The scientific target is **presence/absence only**. Tumor-size estimation is explicitly out of scope. The client clarified that TLC visibility can depend on superficial/deeper presentation; apparent response size or intensity must not be interpreted as tumor size.

Do not count patches, crops, augmentations, repeated frames, or multiple regions from one mouse as additional independent subjects.

## Current live system

The deployed research application already performs:

`Upload -> profile-aware normalization -> TLC response segmentation -> QC -> morphology/features -> DINOv2 representation -> reference analysis -> experimental native classifier -> report/history/storage`

Key current profile:

- `tlc_profile_id = client-device-tlc-pending`
- `clinical_claim = NONE`

Current experimental classifier version:

- `0.1.0-exp8v1-dino`
- research/demo-only 8-vs-1 fit
- status: `ACTIVE_RESEARCH`
- validation status: `NOT_VALIDATED_SINGLE_NEGATIVE`

The current binary fit is useful as an end-to-end baseline only. It must not be treated as validated accuracy, specificity, or human readiness.

## Live smoke observations

Two known training-cohort images have now exercised the deployed path end-to-end:

- known tumor subject `0033` -> `TUMOR_LIKE`, research score about `0.617`
- known healthy subject `0038` -> `NO_TUMOR_LIKE`, research score about `0.401`

These are **training-set smoke tests**, not independent validation.

## Existing technical assets worth preserving unless experiments disprove their value

- Working FastAPI/web product and deployed Oracle/Coolify runtime.
- Durable upload/generated-image storage and report/history pipeline.
- Contact-TLC segmentation and morphology feature extraction.
- Acquisition/QC flags.
- DINOv2 ViT-S/14 representation path.
- TLC/profile provenance and profile locking.
- Publication/reference Contact-LCT assets for reference/domain analysis.
- Reproducible experimental training/runtime model registry.

Do not throw these away by default. Treat them as reusable components and baselines.

## The actual research problem

The challenge is **not** simply “train a better binary classifier on 8 positive and 1 negative mouse.”

The challenge is:

> Given an extremely small, imbalanced, preclinical mouse Contact-LCT cohort, scarce public raw same-domain labelled data, an existing DINOv2 + TLC feature stack, and a final human breast product, discover a strategy that extracts the maximum transferable information from the mouse experiment and reduces the amount of labelled human data required for a reliable human model.

The research must explicitly solve or quantify two separate problems:

1. **small-data learning inside the mouse Contact-LCT domain**
2. **cross-species / acquisition-domain transfer from mouse preclinical evidence to human MumGuard data**

Do not conflate success on mouse training subjects with success on human scans.

## Required research behaviour

### 1. Reframe the task if needed

Do not assume whole-image supervised binary classification is the optimal formulation. Investigate whether the problem is better represented as one or a combination of:

- localized anomaly detection
- multiple-instance learning (MIL)
- local tissue-vs-surrounding-tissue contrast
- prototype / metric learning
- self-supervised representation learning
- contrastive learning
- one-class or normality modelling
- positive-unlabelled learning
- weak supervision
- domain adaptation
- cross-species transfer learning
- feature disentanglement
- test-time adaptation
- physics-informed TLC modelling
- morphology + pretrained-vision hybrid models
- uncertainty-aware ensembles
- hierarchical models separating acquisition/domain effects from biological signal

These are starting points, not a required list. If a better formulation exists, use it.

### 2. Exploit within-subject structure without fake sample inflation

A tumor-bearing mouse image can contain an abnormal region and surrounding tissue that may act as an **internal spatial control**. Investigate whether this structure can produce useful representation learning or local contrast supervision.

Possible hypothesis:

`image -> valid TLC field -> local patches / response components -> compare candidate ROI against nearby/background tissue -> aggregate local evidence -> subject-level tumor-like decision`

This may yield many training instances for representation learning, but evaluation must remain subject-level. Never report patch count as effective subject count.

### 3. Separate modality learning from disease supervision

Investigate whether the current and future unlabeled thermography/TLC images can teach:

- acquisition invariances
- response morphology
- local spatial structure
- device/profile characteristics
- glare/artifact robustness
- tissue/background separation

without requiring tumor labels.

Then use the scarce labels only for the biological decision layer.

### 4. Solve Mouse → Human deliberately

The final product is human. Design an explicit transfer strategy rather than assuming a mouse classifier generalizes.

Investigate questions such as:

- Which parts of the representation are likely modality/physics-specific and reusable across species?
- Which parts are anatomy/species-specific and should be relearned?
- Can mouse supervision initialize a local anomaly representation while a small human cohort calibrates the final head?
- Would domain-adversarial training, CORAL/MMD-style alignment, prototypes, adapters, LoRA-like vision adaptation, test-time normalization, or feature disentanglement help?
- Is it better to freeze DINOv2, adapt only a small head, or learn a TLC-specific representation from unlabeled images?
- Can human unlabeled MumGuard scans provide most of the domain adaptation before labels arrive?
- What is the smallest human labelled study that would actually discriminate promising strategies?

Do not claim a numeric “minimum human dataset” without deriving it from experimental uncertainty, expected effect size, or learning curves.

### 5. Use TLC physics and acquisition metadata

Do not treat these images as generic RGB photographs.

Investigate whether performance improves by modelling:

- TLC hue / saturation / value response
- response area and topology
- connected components
- morphology / elongation / solidity / branching
- local colour transitions
- spatial gradients
- surrounding-tissue contrast
- acquisition glare and frame-touch artifacts
- device/TLC profile identity
- temperature-response range if/when calibration metadata becomes available

If the exact TLC formulation/range is unknown, preserve that uncertainty rather than inventing calibration.

### 6. Do not use synthetic data as fake biological evidence

Synthetic or augmented images may be used for:

- invariance training
- robustness
- colour/exposure perturbation
- blur/compression/crop/rotation robustness
- acquisition-artifact simulation

But do not create synthetic “healthy mice” or “tumor mice” and count them as independent ground truth or validation.

### 7. Search for external evidence aggressively but critically

Perform a fresh literature/data search for:

- Contact-LCT / TLC breast thermography
- preclinical tumor thermography
- human contact thermography datasets
- raw image availability
- breast thermography domain adaptation
- small-data medical imaging
- cross-species imaging transfer
- MIL / anomaly methods relevant to thermal/contact patterns
- weakly supervised tumor localization
- device-domain adaptation

For every external dataset or source, record:

- modality
- species
- anatomy
- acquisition device/profile
- raw vs figure-derived images
- labels/ground truth
- subject independence
- rights/license
- whether it is valid for training, validation, reference-only use, or not usable

Do not silently train on publication figures or infrared datasets as if they are native Contact-LCT.

## Experimental program required

Build a ranked experimental matrix. At minimum compare:

### Baseline A — current whole-image DINO binary head

Keep current `0.1.0-exp8v1-dino` as the reproducible baseline.

### Candidate B — local anomaly / internal-control model

- derive valid field and response regions
- construct ROI vs local surrounding-tissue comparisons
- use pretrained embeddings plus TLC-specific features
- aggregate local evidence to subject level

### Candidate C — MIL

- bag = one independent subject
- instances = subject patches/regions
- labels only at bag/subject level
- prevent leakage between train/test subjects

### Candidate D — prototype / metric approach

- tumor and healthy/local-normal prototypes where scientifically defensible
- quantify distance and uncertainty
- examine whether the single healthy subject can be used as an anchor without being memorized

### Candidate E — one-class / anomaly approach

Evaluate whether a normality or local-consistency model is more defensible than a conventional binary head under extreme class imbalance.

### Candidate F — hybrid DINOv2 + TLC physics/morphology

Compare pretrained visual representation against handcrafted TLC features and a fused model.

### Candidate G — self-supervised / contrastive pretraining

Use all available unlabeled valid regions to learn Contact-TLC-specific invariances, followed by a very small supervised head.

### Candidate H — Mouse → Human adaptation design

Even if human raw data is not yet available, produce the exact adaptation experiment and data contract that should be run the moment human scans arrive.

## Evaluation discipline

Because the cohort has only one negative subject, conventional stratified binary cross-validation cannot establish specificity.

You must design evaluation that is maximally informative without pretending the limitation disappears.

Required:

- strict subject-level separation
- no patch leakage across subject folds
- no augmented version of a subject in both train and test
- report training-fit results separately from held-out evidence
- uncertainty intervals where meaningful
- sensitivity analysis to preprocessing, threshold, and augmentation
- leave-one-tumor-out or other bounded analyses only if their limitations are stated
- explicit statement that one negative cannot estimate population specificity

Where the data cannot answer a question, identify the smallest next experiment that would answer it.

## Human transition data contract

Design the exact human data we should ask MumGuard to collect next. Include:

- minimum metadata fields
- device/profile/acquisition provenance
- subject-level identity without leaking personal identifiers
- ground-truth source (e.g. pathology/imaging/clinical adjudication as applicable)
- breast side / location if useful
- repeated scan handling
- pre/post-acquisition QC
- control definition
- train/validation/test isolation
- prospective vs retrospective split
- raw image/data retention requirements

Prefer a **learning-curve based acquisition plan** over an arbitrary “20/20” rule.

## Success criteria

A strong answer is not a generic proposal. It should produce:

1. a ranked set of hypotheses with expected upside and failure modes
2. a concrete experimental design for each top hypothesis
3. leakage-safe subject-level evaluation rules
4. exact code/data changes required in this repository
5. an ablation table template
6. a decision rule for selecting the next model
7. a Mouse → Human transfer architecture
8. a minimal human-data acquisition plan derived from learning curves/uncertainty
9. a list of external datasets/sources with usage classification and rights
10. a recommendation on which current components to preserve, replace, or retire
11. a versioned target such as `few-shot-local-anomaly-v1` only if evidence supports that direction
12. a final “what we still cannot know” section

## Research output format

Return the work in this order:

### A. Reality check
What is actually learnable from the current 9 subjects, and what is not?

### B. Product-aware problem formulation
Reformulate the ML problem around the real human MumGuard goal.

### C. Literature / dataset findings
Separate usable training sources, reference-only sources, and invalid-domain sources.

### D. Candidate architectures
Rank them and explain why.

### E. Experimental matrix
Specify exact train/test unit, features, losses, augmentations, leakage controls and metrics.

### F. Mouse → Human transfer strategy
Define the transition architecture and the role of unlabeled vs labelled human data.

### G. Human acquisition plan
Specify what MumGuard should collect next and in what order.

### H. Execution plan
Give repository-level tasks/branches/scripts/artifacts needed to implement the top strategy.

### I. Kill criteria
State what experimental result would cause each major hypothesis to be abandoned.

### J. Final recommendation
Choose the highest-value next experiment, not the most fashionable architecture.

## Non-negotiable scientific rules

- `clinical_claim = NONE` for the present research system.
- Do not call model score “cancer probability.”
- Do not infer tumor size from visible TLC area/intensity.
- Do not count augmented patches as independent animals or patients.
- Do not mix infrared and Contact-LCT domains without an explicit domain-transfer experiment.
- Do not mix different TLC/device profiles silently.
- Do not use publication/reference figures as native labelled training data unless the intake contract, rights and ground truth support it.
- Do not assume mouse classification automatically transfers to humans.
- Do not throw away the current pipeline merely because the final target is human; test which components transfer.

## Existing repository context to read first

Read these before proposing experiments:

1. `docs/MASTER_HANDOFF.md`
2. `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md`
3. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_LANES_M_P.md`
4. `docs/CONTACT_THERMOGRAPHY_CONTINUATION_UPDATE_2026-09-11_EXP8V1.md`
5. current `integration` implementation, especially:
   - `app/services/dinov2_service.py`
   - `app/services/client_device_domain.py`
   - `app/services/client_device_qc.py`
   - `app/services/native_classifier_runtime.py`
   - `app/services/analysis_engine.py`
   - model artifacts / trainer scripts / manifests

## Final instruction

Do not optimize for a reassuring answer. Optimize for discovering a strategy that has the best chance of producing a genuinely useful MumGuard human model from unusually scarce preclinical evidence.

If the correct conclusion after experiments is that a component does not transfer, say so and replace that component. If a surprising formulation can extract more signal without violating subject-level independence, pursue it. The goal is not to defend the current model; the goal is to solve the real product problem as far as the evidence allows.
