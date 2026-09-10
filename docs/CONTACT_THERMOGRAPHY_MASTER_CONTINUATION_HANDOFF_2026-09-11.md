# Contact Thermography AI — Full Master Continuation Handoff

**Date:** 2026-09-11  
**Repository:** `omarkhair70-droid/contact-thermography-ai`  
**Canonical branch:** `integration`  
**Integration SHA at handoff creation:** `ba02868ebfbcbf445969d5b2b81f75f0d4c41be8`  
**Current product status:** integrated research/demo application with real client-device LCT handling, live DINOv2, Oracle-ready deployment, QA hardening, target-domain dataset governance, gated native classifier training path, and a 417-feature client-device contract.

---

## 1. Why this file exists

This is the canonical continuation state for a new ChatGPT/Codex conversation. The new chat should read this file first and continue from the **current state**, not re-audit or rebuild work that is already merged.

Do **not** create independent RC ZIP projects as new sources of truth. The GitHub repository is canonical. Work should land on isolated feature branches, PR into `integration`, be tested, then eventually promote one validated demo candidate to `main`.

Recommended new-chat instruction:

> Read `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md` in repo `omarkhair70-droid/contact-thermography-ai`, inspect the current `integration` branch and open PRs, then continue from the exact next actions in the handoff. Do not redo completed lanes, do not invent labels, preserve TLC/device provenance and `clinical_claim=NONE` until the documented native-validation gate is truly passed.

---

## 2. Product goal

The end product is a **browser-based Contact / Liquid Crystal Thermography (LCT) AI analysis application**, not an infrared-camera thermography product.

Target user flow:

`Login/demo entry -> Dashboard -> New Exam -> upload contact-LCT images -> assign LEFT/RIGHT/POSITION -> QC -> plate/response extraction -> LCT signal + morphology -> DINOv2 -> bilateral/within-cohort analysis -> research classification when valid -> saved exam/history -> structured report`

The long-term research target is:

`client-device contact-LCT image -> QC + TLC profile-specific signal + DINOv2 + target-native features -> OOD gate -> TUMOR_LIKE / NO_TUMOR_LIKE research classification`

The current project must **not** present unvalidated outputs as cancer probability or diagnosis.

---

## 3. Hard scientific/domain rules

### Modality

This project is **contact thermochromic liquid crystal thermography**. TLC reflected visible colour changes across its colour-play interval and depends on formulation, illumination, view angle, pressure/placement, white balance, ambient conditions, and acquisition setup.

Do not silently treat LCT as ordinary infrared thermography.

### TLC formulation/device domain shift

The client explicitly stated that their own TLC material/formulation produces images that look different from the publication/reference examples. This is a real domain shift.

Every exam and dataset row must preserve:

- `tlc_profile_id`
- `device_profile_id` when known
- source/domain provenance

Current profiles:

- publication-derived reference plates: `reference-publication-unknown`
- real client-device images: `client-device-tlc-pending`

Colour thresholds, response segmentation, and any future colour-to-temperature calibration must be profile-specific. Do **not** output absolute °C from client RGB LCT images unless a formulation-specific calibration exists.

### Clinical boundary

Until a sufficiently large, coherent, patient/animal-level labelled contact-LCT dataset with proper ground truth and validation exists:

- `clinical_claim=NONE`
- no current cancer probability
- no diagnosis
- no claim that nearest-reference/anomaly score equals malignancy
- no synthetic `NO_TUMOR` labels
- no training on rows marked reference-only or frozen-evaluation

The product can show research signals, morphology, QC, OOD/domain status, similarity, and later a **research classification** only when the target-native training gate is satisfied.

---

## 4. Client facts that materially changed the project

The client supplied **9 real contact-LCT mouse images**:

- `IMG-20260910-WA0030.jpg`
- `IMG-20260910-WA0031.jpg`
- `IMG-20260910-WA0032.jpg`
- `IMG-20260910-WA0033.jpg`
- `IMG-20260910-WA0034.jpg`
- `IMG-20260910-WA0035.jpg`
- `IMG-20260910-WA0036.jpg`
- `IMG-20260910-WA0037.jpg`
- `IMG-20260910-WA0038.jpg`

Client clarification:

- these are **mice**, not humans;
- the mice were experimentally **tumor-bearing / tumor-injected**;
- there was no biopsy workflow because the experimenters already knew tumors had been induced;
- conditions/situations were intended to be the same;
- mouse weights were approximately similar;
- the important variable between these cases is **tumor size**;
- the currently supplied 9-image set therefore contains positives only and is **not** a valid binary training set.

These nine images are preserved as a **frozen target-domain positive/evaluation cohort**. Do not reuse them as both training and testing evidence.

The team/client originally wanted us to try to solve the data gap ourselves because they had difficulty obtaining a ready labelled contact-LCT dataset. That remains a central project objective: proactive data acquisition/research plus model development, not merely waiting for the client to supply a perfect dataset.

---

## 5. Client commercial status

The work evolved from an informal proof-of-concept into substantial **AI R&D + data acquisition + model engineering + product engineering + deployment/QA**.

A WhatsApp message has already been sent to the client explaining:

- no ready adequate dataset existed for their exact contact-LCT modality;
- we built the analysis/product pipeline and ran DINOv2 on real device images;
- we identified TLC/device domain shift;
- we acquired and trained on external thermal data as auxiliary research;
- we are building a target LCT dataset and native classifier path;
- the 9 client mice are tumor-bearing and useful for tumor-burden association once sizes are mapped;
- next phase is target-native `TUMOR_LIKE / NO_TUMOR_LIKE`, blind evaluation, product integration.

The message proposed **30,000 EGP** for the phase covering work completed to date plus continuing to a testable target-model research prototype, with larger validation/final deployment treated as a separate milestone if needed.

At handoff time, wait for the client's response before renegotiating. Do not send a second fee explanation unless a response requires it.

---

## 6. Core application work already completed

The integrated product includes:

- FastAPI browser application
- Dashboard / New Exam / History / Reference workflow
- drag-and-drop multi-image upload
- LEFT / RIGHT / POSITION metadata
- alternating pair helper
- profile controls for TLC/device provenance
- automatic circular plate extraction for publication-style data
- client-device whole-image response path for real device images
- engineering QC
- thermochromic response segmentation
- visible-response morphology features
- original/response mask/QC/result cards
- bilateral alignment/difference analysis
- exam persistence/history/detail
- HTML report
- strict NaN/Inf JSON safety
- explicit non-diagnostic wording
- SQLite local support and PostgreSQL-ready persistence
- durable upload/generated-media storage abstraction
- Docker/Oracle staging assets
- healthchecks / restart policy / rollback docs
- shared-host Oracle mode that avoids hijacking ports 80/443

### Oracle deployment state

Repo-side Oracle deployment preparation is built, including a shared-host mode defaulting to loopback high port (`127.0.0.1:8110`) so it can sit safely behind an existing reverse proxy.

A final public Oracle deployment/smoke is **not** yet the scientific priority and must not be falsely claimed as completed. Final client-facing staging should happen after the model/data state selected for demo is integrated and green.

---

## 7. DINOv2 / Dataset Zero reference intelligence

The original publication/reference dataset is **Dataset Zero**:

- 26 extracted circular thermograms from 7 supplied publication figures
- reference-only / compressed / annotated material
- not a clinically representative labelled training cohort

Official Meta **DINOv2 ViT-S/14** was run on Kaggle GPU.

Reference embedding artifact:

- shape: `(26, 384)`
- dtype: `float32`
- canonical valid NPY size: `40064 bytes`
- SHA256: `cad838124969a4a9d06ad122f44d5cdfa9d200c780e437a00ce6c3abf8163ee9`

A previous repo copy had been corrupted by binary transfer; this was fixed with the original valid Kaggle artifact. That was a repository-upload issue, not a DINO/Kaggle failure.

Integrated reference intelligence includes:

- live DINO embedding for uploaded analysis
- 18 explicit LCT features + 384 DINO = 402-D fused representation
- reference unusualness/retrieval heads
- bilateral learned representation: `L + R + |L-R| + L*R` = 1536-D before reduction/modeling

All these historical/reference scores remain **research/reference signals**, not cancer probabilities.

---

## 8. Lane E — real client-device domain adaptation

The real 9-image client set was run through a dedicated client-device path rather than publication circular-plate assumptions.

Completed Lane E work includes:

- client-device response segmentation tuned to reject dark/saturated setup background
- real client selector wired into normal upload routing for `client-device-tlc-pending`
- 45-configuration threshold sweep
- blind/frozen response-area ranking before tumor-size ground truth is revealed
- colour-domain-shift report
- client-device acquisition QC including glare/clipping/edge response
- official DINOv2 embeddings for all 9 images
- client-client cosine distance matrix
- nearest publication-reference comparisons
- empirical DINO embedding domain-shift summary
- frozen-cohort hashing / anti-post-hoc-label contract
- tumor-size association evaluator that refuses changed ranking/cohort

### Frozen blind response-area ordering

The median response-area order from the sweep was frozen as:

`WA0037, WA0036, WA0034, WA0033, WA0032, WA0035, WA0038, WA0031, WA0030`

This is **not** a tumor-size prediction. It is a pre-label thermochromic-response ranking to compare later against the true tumor-size mapping.

### Client DINO domain result

The 9 client images showed strong embedding-domain shift from the 26 publication references.

Key run summary:

- reference nearest-neighbour median ≈ `0.0968`
- client nearest-reference median ≈ `0.5795`
- median shift / reference IQR ≈ `6.44`
- fraction of client images beyond reference q75 = `1.0`

Interpretation: the client cohort forms its own domain. The publication reference set cannot be used as direct tumour/no-tumour ground truth for these images.

Within the client cohort:

- closest pair observed: `WA0035` / `WA0036` (cosine distance ≈ `0.1358`)
- `WA0035` was relatively central in the client cohort
- `WA0030` was relatively more internally unusual

Again: no diagnosis was inferred from these distances.

---

## 9. Auxiliary DMR-IR experiment

Because no large public modern contact-LCT binary dataset was found, we acquired the original **UFF DMR-IR** infrared dataset as an **auxiliary source-domain experiment**, not as final validation for client LCT.

Kaggle extraction succeeded after replacing broken p7zip extraction with RAR-capable extraction logic.

Recovered original structure:

- `1522` image records
- `56` subjects
- `37 CANCER`
- `19 HEALTHY`

The classifier evaluation was done at **subject level**, aggregating multiple frames/views before cross-validation to avoid image-level leakage.

First fused baseline (DINO + explicit thermal features) produced approximately:

- Logistic Regression sensitivity ≈ `83.8%`
- Logistic Regression specificity ≈ `84.2%`
- Logistic Regression AUROC ≈ `0.909`
- calibrated Linear SVM sensitivity ≈ `94.6%`
- calibrated Linear SVM specificity ≈ `52.6%`

Ablation was more important than the headline metric:

- DINO-only AUROC ≈ `0.876`
- fused AUROC ≈ `0.909`
- explicit thermal features-only AUROC ≈ `0.923`
- when absolute temperature level was removed and only relative/distributional shape retained, AUROC dropped to roughly `0.626`

Interpretation: DMR-IR contains a strong absolute-temperature signal that may mix biology with acquisition/protocol confounding. Therefore the DMR classifier is **auxiliary only** and must not be transferred blindly to RGB contact-LCT.

An OOD/abstention gate was subsequently added so a shifted client image cannot silently receive a source-domain tumour classification.

---

## 10. Data hunt findings / source strategy

The project explicitly distinguishes:

- downloadable/reusable open data
- reference-only article figures
- rights-restricted historical material
- private research cohorts that require direct data access
- private client data

### Strong target-domain leads already identified

1. **USM / Universiti Sains Malaysia contact-LCT program**
   - current/recent human contact-LCT cohort
   - 108 planned; 42 completed at review submission
   - review reports preliminary sensitivity/specificity but the underlying image dataset is not public
   - currently the most actionable direct-access target

2. **USM 2025 dissertation / related case-series work**
   - explicitly evaluates liquid-crystal thermography for breast cancer
   - likely overlaps with the current USM program, so subjects must be deduplicated if access is granted

3. **2026 CC-BY LCT review observed cases**
   - includes pathology-linked contact-LCT exemplars
   - article figures can be treated as attribution-preserved reference material, not a large training cohort

4. **Braster / DeepBraster historical-modern contact-LCT work**
   - important architectural/validation precedent
   - raw labelled thermograms are not publicly available in a reusable training dataset found so far
   - do not claim Braster performance as ours

5. **Davison 1972**
   - 105 abnormal women
   - 17 histologically proven carcinoma cases
   - 197 apparently healthy women referenced in the work
   - image reuse rights are restricted/unclear; use as evidence/methodology and seek archival/data permission

6. **Stevens & Rogers 1971 — mouse contact-LCT**
   - 30 tumour-bearing mice
   - 20 C3H mammary carcinoma + 10 A/Jax hepatoma
   - closest historical modality/species precedent to the client mouse experiment
   - publisher/raw image rights restricted; do not scrape previews into training

7. **Other historical large LCT cohorts**
   - Bothmann 1986: 19,461 examined; 2,002 biopsy/histology subset reported
   - Bothmann 1976: 132 carcinoma cases
   - Tricoire 1970: 300 patients
   - Kucera 1976: 319 unselected patients
   - Sforza 1991: >12,000 clinical experience reported
   - these are archival/data-access leads, not current reusable training datasets

### Important biological/modeling lesson

Mouse tumour thermography literature does **not** support a simplistic rule `hot = tumour`.

Some tumour models can produce hotter or cooler surface regions depending on perfusion, vascularisation, stage, and experimental setup. The native classifier must therefore combine morphology, relative contrast/response structure, QC, DINO representation, and profile provenance rather than one absolute heat/colour direction.

---

## 11. Current target-domain dataset governance

Target Dataset v1 is merged and intentionally conservative.

It includes:

- pathology-linked human contact-LCT exemplars as reference-only rows
- 9 client mouse images as `TUMOR_BEARING`, private-client, frozen target evaluation

It **does not** claim that native binary training is ready.

The native training readiness gate requires a coherent trainable pool with:

- one species per model experiment
- one TLC profile
- one device/acquisition profile or an explicitly validated harmonisation strategy
- train-eligible rows only
- real positive and real negative/control subjects
- subject-level grouping
- minimum currently enforced baseline: at least 5 positive and 5 negative subjects before fitting

Do not weaken this gate simply to produce a classifier number.

### Client mouse ground-truth template

`data/client_mouse_ground_truth_template.csv`

currently contains the nine client subjects with `TUMOR_BEARING` and empty tumour-size/volume fields.

When the real mapping becomes available, populate size/volume + units + measurement method/timepoint, then run the frozen association evaluator. Do not alter the previously frozen response ranking after seeing size labels.

---

## 12. Native contact-LCT classifier path already built

The gated native trainer is merged.

It provides:

- strict readiness validation
- subject-level OOF evaluation
- class-balanced Logistic Regression
- calibrated Linear SVM
- generic `feature_*` input contract
- rejection of positive-only cohorts
- rejection of mixed species
- rejection of mixed TLC/device domains where invalid
- no synthetic negatives
- `clinical_claim=NONE`

This means model-training infrastructure is no longer the main blocker. **Target-domain negative/control data is now the main blocker.**

---

## 13. Target-native 417-feature contract — merged

PR #24 (`Lane L — target-native 417-feature builder`) is **merged** into `integration`.

The first reusable client-device feature vector contains exactly **417 finite features per image**:

- 18 visible-response signal/morphology measurements
- 8 engineering QC measurements
- 7 morphology one-hot values
- 384 normalized DINOv2 ViT-S/14 values

`18 + 8 + 7 + 384 = 417`

Contract version:

`lct-target-v1`

Important implementation details:

- currently profile-locked to `client-device-tlc-pending`
- supports injected/reused DINO embeddings so Lane 5 GPU outputs can be reused without rerunning GPU inference
- preserves label, label provenance, species, TLC profile, device profile, use role, train eligibility, QC and morphology metadata
- feature metadata field was renamed to `contract_version` so `feature_*` selectors count only numeric model features
- regression CI was fixed and passed after this correction

The nine client images remain `FROZEN_TARGET_EVAL` and are **not** made trainable by this builder.

---

## 14. PR / lane history that matters

### Integrated/superseding PRs

- **#6** Lane C — Oracle staging deployment — merged
- **#9** Lane A — live DINOv2 — merged
- **#10** A+C integration repair — merged
- **#11** integrated QA hardening with A+C — merged
- **#12** integrated polished UI with live DINO runtime — merged
- **#13** client-device LCT first pass — merged
- **#14** client-device blind freeze + DINO domain artifacts — merged
- **#16** tumour/no-tumour research data foundation — merged
- **#17** DMR-IR acquisition + subject-level auxiliary baseline — merged
- **#18** DMR-IR ablation/confound analysis — merged
- **#19** target-domain LCT data hunt + OOD abstention gate — merged
- **#20** target-domain LCT Dataset v1/readiness gate — merged
- **#22** gated contact-LCT native binary baseline — merged
- **#24** target-native 417-feature builder — merged

### Old PRs that must not be merged blindly

- **#7** original Lane B UI PR is still open/stale but its accepted work was superseded by **#12** integration. Do not merge #7 into current `integration`.
- **#8** original Lane D QA PR is still open/stale but its accepted work was superseded by **#11** integration. Do not merge #8 into current `integration`.

### Current open work at handoff time

- **PR #26 — `Lane M — open-license LCT references and DeepBraster precedent`**
  - base: `integration`
  - head: `feat/lct-open-reference-pack`
  - latest known CI: **success / green**
  - purpose: case-level contact-LCT reference registry; explicit open-license pathology-linked reference cases; Braster public cases kept rights-restricted; validator forbids reference cases from silently becoming train-eligible; documents DeepBraster/THERMACRAC/THERMARAK/INNOMED precedent without claiming proprietary datasets or their reported performance as ours
  - no restricted images copied
  - native mouse binary training remains gated

Before doing new work, inspect whether #26 is still open and mergeable. If still green and its diff matches this description, review and merge it into `integration`, then branch the next work from the new integration SHA.

---

## 15. Data-access outreach prepared, not yet sent automatically

The repo contains a prioritized data-access plan for the most valuable target-domain sources.

Primary outreach targets include:

- USM investigators behind the 2025-2026 contact-LCT program
- Braster prospective investigators / Jagiellonian group
- historical archive/rightsholder routes for Davison, Bichara, Stevens/Rogers

Minimum requested data contract should include:

- de-identified subject/case ID
- image-to-subject mapping
- confirmed contact-LCT modality
- species
- positive/negative/control label
- label provenance / pathology or experimental ground truth
- TLC formulation/foil profile if known
- device/acquisition profile if known
- explicit data-use/research/commercial terms

Do not request unnecessary personal identifiers.

No outreach email should be sent without user approval if it is an external action. Drafting is fine; sending requires explicit instruction.

---

## 16. What is actually finished vs what is not

### Finished / materially working

- product UI and exam workflow
- client and publication TLC profile separation
- real client-device response analysis
- live DINOv2 integration
- reference unusualness/fused intelligence
- bilateral analysis
- storage/persistence/reporting
- QA/clinical-claim guards
- Oracle deployment preparation
- 9-image real-client frozen analysis cohort
- client DINO domain-shift analysis
- DMR-IR auxiliary dataset acquisition and classifier baseline
- DMR ablations/confound discovery
- OOD abstention concept/tooling
- target dataset governance/readiness validator
- native classifier trainer gated for real target labels
- 417-feature client-device target contract
- target data-source registry and data-access plan

### Not finished / must not be overstated

- no validated client-device tumour/no-tumour classifier yet
- no genuine same-device negative/control mouse cohort yet
- no tumour-size mapping for the 9 client mice yet
- no formulation-specific TLC calibration to absolute temperature
- no prospective clinical validation
- no claim of medical diagnosis
- no final public Oracle client demo smoke of the eventual target-model version yet
- no large reusable public case-labelled contact-LCT dataset found yet

---

## 17. Exact next plan

### Immediate next action

1. Inspect PR #26 status and diff.
2. If still green/mergeable and consistent with the handoff description, merge #26 into `integration`.
3. Update the working integration SHA in any new branch/handoff notes.

### Then continue in two parallel tracks

#### Track A — target data acquisition

Continue deep, source-specific search for **actual reusable labelled contact-LCT images**, especially:

- mouse tumour + matched controls using liquid-crystal contact thermography
- human contact-LCT with pathology-linked benign/malignant/control labels
- thesis supplements / institutional repositories / research data deposits
- direct investigator datasets available by request

Do not waste time repeatedly searching generic IR datasets unless a specific auxiliary experiment is justified.

Prepare/draft outreach to USM first because it is currently the strongest active target-domain lead. If the user explicitly says to send outreach, use the relevant email connector/tool and send only after verifying the public contact route.

#### Track B — model/readiness execution

The model code is ready to move quickly once target-domain negatives arrive:

1. ingest rows with rights + ground-truth metadata
2. run dataset validator
3. build `lct-target-v1` 417-feature table
4. group by subject/animal
5. run native Logistic/SVM baseline
6. compare feature blocks:
   - DINO only
   - LCT signal/morphology only
   - QC-aware fused
7. perform OOD/domain checks
8. freeze evaluation split before interpreting results
9. only then consider MLP/tree models or DINO fine-tuning if data volume supports it

Do **not** fine-tune DINO on tiny/noisy labels just to say a deep model was trained.

---

## 18. Tumour-burden experiment for the 9 client mice

This is separate from binary tumour/no-tumour training because all 9 current mice are tumour-bearing.

When the client provides each mouse's true tumour size/volume:

1. map each image to the correct mouse/case
2. preserve the frozen pre-label ranking
3. run `scripts/evaluate_client_tumor_burden.py` / existing mapping evaluator
4. test association of tumour burden against:
   - response area fraction
   - component/morphology features
   - DINO embedding-derived within-cohort structure
   - potentially multivariable target feature blocks
5. report correlation/association without converting it into diagnostic probability

If there is a real monotonic or otherwise stable association, that becomes one of the first target-device biological validity signals.

---

## 19. Demo/product integration after a native research head exists

Once a valid target-native research classifier passes the gate and evaluation:

Add its output to the existing web app **without removing existing evidence panels**.

Recommended result card semantics:

- `Research classification: TUMOR_LIKE / NO_TUMOR_LIKE`
- model/version
- OOD status / abstention status
- QC status
- TLC/device profile
- evidence summary (response morphology + learned signal)
- no clinical diagnosis wording

If OOD or QC fails, the product should abstain rather than invent a confident class.

Then run:

- full regression suite
- known-image smoke
- client-image smoke
- report smoke
- container smoke on a Docker-capable machine
- Oracle staging deployment on a safe high port/shared-host mode
- public HTTPS smoke

Only then select a demo candidate for `main`.

---

## 20. Git/branch discipline

- `main` = stable/promotion branch
- `integration` = canonical consolidation branch
- new work branches from the latest `integration`
- PRs target `integration`
- workers should not self-merge unless explicitly acting as integration owner after review/green CI
- never silently revert another integrated lane
- inspect overlapping changes in `app/main.py`, analysis services, storage, profile provenance and report semantics carefully

Before merging an older PR, check whether a newer integration PR already superseded it.

---

## 21. Important repo files to read in a fresh chat

At minimum inspect:

- `docs/CONTACT_THERMOGRAPHY_MASTER_CONTINUATION_HANDOFF_2026-09-11.md` — this file
- `docs/MASTER_HANDOFF.md`
- `docs/TLC_VARIANT_HANDLING.md`
- `config/tlc_profiles.example.json`
- `docs/LCT_TARGET_FEATURE_CONTRACT.md`
- `docs/LCT_DATA_ACCESS_OUTREACH_TARGETS.md`
- `data/lct_reusable_image_hunt_v2.csv`
- `data/client_mouse_ground_truth_template.csv`
- `scripts/build_lct_target_features.py`
- `scripts/train_lct_native_binary.py`
- `scripts/evaluate_client_tumor_burden.py`
- `scripts/ood_abstain_gate.py`

Also inspect current open PRs/issues before creating duplicate work.

---

## 22. Practical interpretation for the next assistant

The project is **not stuck because there is no model**. We already have:

- a real DINOv2 backbone
- an auxiliary binary classifier experiment
- a target-native feature representation
- a gated native binary trainer
- an OOD concept/tool
- a real client-domain cohort

The project is currently **data-limited, specifically negative/control and ground-truth target-domain data-limited**.

The smartest next work is therefore:

1. finish/merge the open evidence/reference lane if still valid;
2. aggressively but legally obtain target-domain contact-LCT data;
3. preserve clean labels and subject grouping;
4. train the native research head immediately when the gate opens;
5. integrate it into the already-built product;
6. deploy and demonstrate only after end-to-end validation.

Do not restart the project from “what model should we use?” and do not waste another cycle re-deriving the same domain-shift finding.

---

## 23. Current definition of success

A credible next milestone is **not** “100% accuracy” and not “cancer diagnosis”.

It is:

- a coherent, rights-audited contact-LCT target dataset containing real positive and real negative/control subjects;
- a subject/animal-level split with no leakage;
- a native 417-feature baseline and OOD-aware evaluation;
- reproducible metrics on frozen holdout data;
- a research `TUMOR_LIKE / NO_TUMOR_LIKE` result integrated into the browser app;
- a public staging demo that can ingest the client's real device images and abstain when evidence/QC/domain is insufficient.

That is the direction to continue from this handoff.
