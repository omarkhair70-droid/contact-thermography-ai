# MumGuard executable research program

**Recommendation: build a profile-aware, multiscale local-evidence engine with frozen DINOv2 and TLC photometric features; use mouse supervision only in a small, replaceable decision layer.** Develop the reusable sensing, localization, comparison, and uncertainty components now. Transfer those components to humans, then learn the human decision boundary from human evidence. Do not transfer the current mouse classifier's threshold or its presumed direction of temperature response.

The first implementation target is **`mumguard-local-contrast-v0.1-prototype`**, an inactive research candidate. This report includes runnable code for local evidence extraction, an optional pinned DINO token extractor, bounded mouse head experiments, and an audit of committed evidence. The integrated model recommendation remains a hypothesis to test; no new biological performance is claimed.

Repository inspected: `omarkhair70-droid/contact-thermography-ai`. Canonical `integration` snapshot: `c417da1f15146f6ab15974a0b7fbf399cb97842b`. Research instructions snapshot: `05e37ef36b07638bb913e0585ace8990ddfd16af`. At inspection, their only differences were the two Astra instruction documents. Source review date: 11 September 2026. Existing live-service observations are supplied project evidence, not a deployment retest in this program.

## A. Reality check

### What the available evidence already tells us

There are nine independent mice: eight tumor-bearing subjects, `0030`–`0037`, and one healthy subject, `0038`. The latest resolved class map supersedes older all-positive handoffs and source-registry descriptions. All nine have already participated in development and the experimental fit. They cannot become a pristine external test set by renaming their manifest role. Future subject-excluded experiments are useful exploratory evidence, with their preprocessing-development history disclosed.[1]

The following audit was executed against committed CSV/JSON files, without reconstructing images or inventing observations:

| Check | Result | Engineering implication |
|---|---|---|
| Stored hypotheses described as blind before class-map reveal | Six of eight positives flagged; `0030` and `0031` missed; `0038` unflagged | A small or weak visible response must not imply absence. This is a retrospective audit of stored hypotheses; historical blinding was not independently re-established. |
| Stored DINO training scores at threshold 0.5 | All nine labels separated | Confirms the saved fit, not generalization. |
| Smallest positive training-score margin above 0.5 | 0.05528 | A useful perturbation target; neither uncertainty nor a clinical margin. |
| Thresholds separating these saved training scores | `0.400324 < threshold <= 0.555280` | Many thresholds reproduce the same training fit; threshold calibration is unresolved. |
| Possible exhaustive 8/1 subject-label assignments | Nine | The smallest unrandomized exhaustive one-sided permutation p-value is 1/9, assuming exchangeability. More random shuffles do not create finer evidence. |

These computations are reproduced by `research/mumguard/audit_current_evidence.py`. The audit receipt includes source hashes and eight leave-one-positive-out fold definitions. No held-out classifier experiment is represented as executed.

The checked-out repository contains the saved mouse classifier and label map, but not the nine original JPEGs or their nine-row embedding matrix. Those are existing private assets referenced by the project, not a request for new subjects. Mount them from the team's existing storage to execute the image-level experiment. The provided CPU prototype and evidence audit execute without waiting for any new disease labels.

### What can be learned now

We can test whether a proposed representation is robust to image handling, whether it detects localized photometric departures, whether comparator construction is usable, whether mouse scores depend on setup artifacts, and whether held-out positives retain evidence when excluded from fitting. We can also test whether the single negative is simply being memorized. Patches provide repeated observations for those engineering objectives; they do not increase the independent animal count.

We cannot identify a human disease boundary, population specificity, or clinically useful negative predictive value from these nine animals. Even a genuinely independent, fixed-rule test that correctly classified one negative would have an exact two-sided 95% specificity interval of 2.5%–100%. That illustrative interval does **not** apply to `0038`'s training prediction. Eight independent correct positive tests would have a corresponding lower limit of about 63.1%; overlapping leave-one-out fits do not automatically satisfy that simple binomial model.

### Implementation findings that change the architecture

1. **The current valid mask is a frame mask.** `letterbox_square()` marks original-image pixels, not confirmed tissue-to-TLC contact. Its complement of response cannot safely become a healthy-tissue bank.
2. **Response extraction is not temperature measurement.** The selector uses brightness/saturation/component rules. A dark area may mean an out-of-range material response, poor contact, occlusion, or setup background.
3. **The current backbone path discards spatial information.** `encode_rgb()` uses resize/center crop and returns one 384-dimensional image vector. Keep that path for baseline A, but add full-field patch-token extraction for local models.
4. **The runtime's species boundary is incomplete.** The artifact says `mouse`, but `predict_native_binary()` accepts only a feature vector and registry path. TLC profile matching alone does not enforce species or acquisition compatibility.
5. **QC is reported after classification has already been computed.** In the inspected analysis path, client QC replaces the generic QC result after `analyze_plate()` calls DINO/native inference. Add an explicit eligibility gate before a new disease head.
6. **The label ontology needs correction before human intake.** Existing trainers include `BENIGN` among negative labels. A benign tumor is not equivalent to no tumor for a presence/absence endpoint. Preserve separate lesion-presence, malignancy, and reference-standard fields; never infer presence from a generic benign label.

These are targeted research-contract changes. Preserve the current application, storage, reports, and reproducible baseline.[1]

## B. Product-aware problem formulation

### Learn a measurement and comparison system before learning a disease boundary

The most useful scarce-label formulation is:

```text
acquisition + profile + contact evidence
    -> observable-field and artifact assessment
    -> multiscale candidate regions, including low-response regions
    -> region versus disjoint surrounding contact tissue
    -> frozen visual features + two-sided TLC contrasts
    -> per-acquisition evidence map and uncertainty
    -> subject/exam aggregation
    -> small species-specific research head, when supported
```

The reusable output is an **evidence vector and local map**, not a universal tumor probability. It answers: where does the measured field differ from comparable tissue, in which feature channels, and is that difference stable and measurable? A downstream human head eventually answers whether that evidence supports the selected human endpoint.

Represent a measured image conceptually as `image = renderer_profile(surface field, contact, illumination, camera, time)`. The surface field itself depends on anatomy, physiology, disease, and environmental conditions. This is a decomposition for engineering experiments, not an identifiable causal model fitted from nine mice.

Local comparison can cancel some shared illumination and background level. It cannot automatically remove pressure gradients, regional anatomy, vascular differences, inflammation, or species effects. Therefore the model must retain a parallel global/calibrated channel rather than normalizing away every absolute signal. The existing DMR-IR ablation, where removing absolute thermal level reduced reported performance substantially, is a warning against assuming that local normalization always helps.[1]

### The key additional ingredient: observability

Every pixel/region should distinguish:

- confirmed tissue contact with usable chromatic response;
- confirmed contact with ambiguous or out-of-range response;
- glare, clipping, occlusion, or otherwise unreliable measurement;
- outside contact or unknown contact.

**No visible TLC response is not a negative biological label.** A tumor-bearing subject need not contain an observable abnormal instance. This breaks the simplest MIL assumption that every positive bag contains a detectable positive patch. Keep an explicit `visibility_unknown` state and allow a positive bag to remain unexplained.

The initial product can already produce local evidence and reacquisition guidance. Its usefulness is improved measurement and hypothesis testing; human disease discrimination remains the subsequent, separately evaluated layer. This directly reduces how much disease-labelled data must train a high-dimensional model.

## C. Literature and dataset findings

### Findings that influence the design

**Frozen local descriptors are a credible starting point.** AnomalyDINO uses DINOv2 patch-neighbour comparisons with few normal examples and no task-specific backbone training. Its evidence is principally industrial, so its numerical results do not transfer to MumGuard. Its method motivates candidate B/E, not a medical performance claim.[2] A MICCAI 2025 study also models normative DINOv2 embeddings for medical anomaly detection; that supports testing normalized local descriptors in medical images while still requiring suitable normative examples.[3]

**Bag supervision is useful but does not supply localization truth.** Attention-based MIL learns from one label per bag. We can use its aggregation idea with a subject as the bag, but attention weights must remain attribution, not tumor segmentation ground truth.[4]

**Unlabelled invariance learning is feasible, but only with correct invariances.** VICReg provides paired-view consistency with anti-collapse terms. In this program it is an optional small representation adapter, not justification for training a new foundation model on nine animals.[5]

**Making mouse and human features indistinguishable is not sufficient.** Domain-adaptation theory shows that marginal alignment can worsen prediction under differing label distributions. Eight-of-nine induced mouse positives and a human acquisition stream should not have their entire distributions forcibly aligned. Use same-tissue repeated acquisitions and matched nuisance conditions when available, then test target-domain outcome performance.[6]

**Thermal polarity is not portable.** Song and colleagues observed cooler xenograft regions in mice using infrared thermography. This is evidence against a universal hot-equals-tumor rule; it is neither Contact-LCT training data nor a tumor-size model for this product.[7]

**TLC is a measurement system with a transfer function.** Experimental TLC error work identifies effects including viewing/illumination geometry, film thickness, hysteresis, aging, and noise. Learn or measure the acquisition response separately; do not assign degrees Celsius to an unknown RGB color map.[8]

### Data and pretrained-knowledge ledger

Access, rights, and scientific eligibility are separate. “Downloadable” does not establish same-domain validity; an article's license does not automatically license underlying patient data. No public raw, subject-linked, rights-cleared **MumGuard-profile Contact-LCT** cohort was located in this fresh search. That bounded finding determines the engineering strategy, not whether work proceeds.

| Source | Modality / species / anatomy and device | Rawness, labels, independent unit | Rights/access established | Permitted program role |
|---|---|---|---|---|
| Existing nine client acquisitions | Contact-LCT; mouse; exact imaged anatomy needs metadata; client formulation pending | Original private JPEGs referenced externally; eight induced tumor-bearing, one healthy; nine mice | Existing private project assets; preserve authorized use/provenance | Native development, local structure, subject-excluded experiments; no human validation |
| Existing Dataset Zero | Human breast Contact-LCT publication plates; profile unknown | 26 extracted plates from seven figures; subject independence not established | Existing reference-only governance | Reference/domain illustration only; exclude from learning banks and evaluation |
| USM 2026 LCT article and ongoing program | Human breast; Braster Contact-LCT | Review includes illustrative observations and describes 42 completed of 108 planned; underlying cohort not supplied as open raw data | Article CC BY 4.0, subject to figure credits; raw dataset rights separate | Open figures remain reference-only; direct raw-data request is a secondary opportunity |
| Braster / INNOMED / DeepBraster | Human breast; proprietary Contact-LCT system | Manufacturer report describes automatic interpretation studies; no reusable raw archive verified | Published reports, not an open model-training grant | Architecture/validation precedent only |
| Stevens & Rogers 1971 | Mouse transplantable tumors; historical Contact-LCT setup | Article precedent, not a verified downloadable control dataset | Publisher lists restricted access | Methodology only; no image training |
| UFF DMR-IR original source | Human breast noncontact IR; FLIR SC620 protocol described by UFF | Static/dynamic image and temperature data; preserve patient IDs; repo's recovered subset is 56 subjects | Original source public access/citation route; original reuse terms must be retained | Existing auxiliary experiment; audit absolute-level confounding and local representations within IR |
| Figshare `21225389` DMR-IR mirror | Derivative of the same IR source, not another independent cohort | Archive listed; exact overlap and file integrity must be checked | API explicitly says CC BY 4.0; uploader is a third party | Conditional auxiliary research after provenance/rights audit; never count as additional subjects or native LCT |
| Rodriguez-Guerrero et al., Breast Thermography v3 | Human breast IR; FLIR A300 | Radiometric JPEG, three views per subject; 119 women, 84 benign and 35 malignant; no healthy-subject cohort in paper | Dataset page CC BY-NC 3.0; companion article CC BY-NC 4.0 | Conditional noncommercial IR morphology/benign-vs-malignant study; no commercial incorporation without permission; benign is not no-tumor |
| UTFPR Thermodataset | Human breast dynamic thermography, with other imaging; exact camera/data contract needs inspection | Institutional record reports 21 participants and 280 images across modalities; raw thermography subset not verified | Thesis CC BY-NC-SA 4.0; raw-data reuse terms unverified | Secondary data-access lead; no training ingestion yet |
| Song 2007 | Mouse xenografts; noncontact IR | Original experimental paper; reusable raw corpus not located | Publisher research article, not dataset license | Polarity/biology evidence only |
| Official DINOv2 ViT-S/14 | Natural-image pretrained visual representation | Pretrained weights; no thermography disease labels implied | Official code and standard DINOv2 weights Apache 2.0 | Use now as frozen backbone; pin revision and hashes |
| AnomalyDINO | Algorithm/code, not patient data | Few-shot local descriptor implementation | Official repository Apache 2.0 | Method inspiration or licensed implementation reuse; do not import industrial benchmarks as medical evidence |
| DINOv3 | Newer generic visual foundation model | Pretrained dense features, different representation | Official DINOv3-specific license, not Apache 2.0 | Optional frozen-backbone challenger after license acceptance and same-protocol ablation; not the initial dependency |

Primary sources for the ledger: USM [9], Braster [10], historical mouse article [11], UFF and Figshare [12–13], raw Colombian dataset and paper [14–15], UTFPR [16], official backbone/code repositories [17–18].

**Concrete external-data experiment:** use already authorized DMR-IR data to compare local evidence versus absolute-level features under patient-level and session-aware splits. If its raw temperature matrices are available, render deterministic alternative color maps only to test invariance to display choices. Do not present those renderings as Contact-LCT: real contact changes heat transfer and the device's response may saturate or censor the field. An IR teacher may supervise coarse relative-structure consistency in a separately labelled auxiliary experiment; it must not supply MumGuard pseudo-disease labels.

**Rejected shortcuts:** training a CLIP prompt classifier on “healthy breast” versus “cancer”; importing a mammography/pathology model's disease head; creating healthy mice with image synthesis; absorbing article figures into a normal bank; treating all unlabelled scans as healthy; or pooling IR and TLC RGBs as one domain. None resolves the missing measurement/label mapping.

## D. Candidate architectures

The ranking is an engineering priority, not a claim that candidate B has already won a biological comparison.

| Rank | Candidate | Why test it | Main failure mode / fallback |
|---:|---|---|---|
| 1 | **B + F: local contrast with frozen DINOv2 and TLC features** | Uses shared conditions within one acquisition; most of the representation requires no disease labels; produces inspectable local evidence | Abnormality can be diffuse, low-visibility, or resemble ordinary tissue; retain global features and abstain when comparison is unavailable |
| 2 | **D: robust local prototypes with subject-balanced reference contribution** | Adds reusable memory and asks whether `0038` is an anchor or an outlier | One healthy animal cannot define healthy variation; separate confirmed normal anchor from unlabelled local comparators |
| 3 | **C: constrained MIL on frozen region features** | Uses subject labels without labelling every patch positive | Learns glare/contact/bag-size shortcuts; begin with fixed pooling and at most a tiny head |
| 4 | **E: self-reference / local-consistency anomaly model** | Can run without a healthy-subject training set | A subject can be internally consistent but diseased; useful evidence engine, weak standalone negative decision |
| 5 | **G: small self-supervised adapter** | May improve device invariance on unlabelled valid acquisitions | Nine subjects can teach identity and artifacts or erase biological color; keep frozen features as control |
| 6 | **A: current whole-image DINO head** | Essential reproducibility, runtime, and shortcut baseline | Memorizes single negative, global setup or anatomy; retain as baseline only |
| Transition | **H: human sensor adapter + human local reference + human head** | Transfers reusable sensing, not mouse disease semantics | No validated human transfer evidence yet; compare against a model with no mouse supervision |

A high-capacity end-to-end CNN, a transformer MIL aggregator, adversarial cross-species alignment, or full DINO fine-tuning has lower information value now. A larger network cannot create independent negatives or identify the human decision rule.

**Prototype experiment D, concretely:** start with the same fixed subject-summary vector used by B/F, standardized within each training fold. Compute one medoid per class, or the median distance to positive training subjects against the distance to `0038`. Use `distance_to_negative - median_distance_to_positives` as a research contrast, without a fitted clinical threshold. Compare this with a labelled-normal patch bank built only from valid `0038` contact. Positive-image patch banks remain unlabelled unless independent lesion locations exist; subject labels do not make every patch a tumor prototype. Cap reference contributions per subject and record each prototype's original subject and acquisition.

**One-class experiment E, concretely:** compare training-free within-image ring surprise with nearest-neighbour distance to that single healthy anchor. A version pooling internal comparators from positive animals is an unlabelled local-consistency model, never an established healthy-density model. It must pass contamination and anchor-deletion tests. A positive-unlabelled disease classifier is deferred: the available positive bags do not provide a representative unlabelled population or an identifiable class prior, and normal-looking patches cannot repair that assumption.

### Exact top-candidate specification

**Inputs.** Original RGB acquisition, EXIF orientation, subject ID, acquisition/session ID, species, anatomy/side/location, TLC profile, device/profile identifiers, camera settings if known, raw-file hash, and independent contact and artifact masks. Preserve a separate optional response mask from the existing selector. Optional inputs are real repeats, an anatomically matched contralateral view, an adjudicated prior acquisition, calibration references, and an independently recorded mouse injection/lesion location. Missing fields remain unknown, not default-normal.

**Contact/response preprocessing.** For the nine existing images, create coarse contact polygons and artifact exclusions while the annotator is blind to class. This is acquisition annotation, not tumor annotation. Record an alternative plausible boundary for sensitivity analysis. If contact cannot be identified from a photograph, use a conservative support region and mark the missing field; do not silently use the entire frame. For future device scans, use sensing-head geometry/fiducials plus an independently checked contact signal. Preserve the full field with aspect ratio and padding; do not center crop the new local path.

Keep the existing 256-pixel baseline unchanged. Use a full-field 224-pixel local prototype for immediate low-cost checks; evaluate a 448-pixel token grid only from source images with meaningful resolution. Upsampling a 256-pixel derivative is not new detail. All masks and token coordinates share an explicit transform back to the original image.

**Candidates.** Construct a fixed grid over contact support, at window sides of 1/14, 1/7 and 2/7 of the normalized field (16/32/64 pixels at 224). Also add existing response components and non-response islands inside confirmed contact in the full implementation. Grid candidates prevent the response selector from defining all possible disease evidence. An outer comparison window is three times the candidate width; exclude a two-times-width central square to create a disjoint surrounding ring. Reject candidates/rings with insufficient valid support.

Do not select comparator patches because the disease head called them normal. The initial surrounding ring is simply **unlabelled internal reference tissue**. Stress-test multiple ring widths and contamination, and use matched remote regions only when anatomy/contact metadata justify them. A large abnormal field can contaminate all local references; the global channel and missing-comparator state are essential.

**Frozen representation.** Keep the pinned official `dinov2_vits14` from the current app. Add `forward_features()['x_norm_patchtokens']`, normalize each 384-dimensional token, and retain the full grid. For an ROI, compute local cosine distance to nearby ring tokens, as well as pooled ROI-minus-ring descriptor summaries. A full-image ViT token has global context; compare against independently encoded ROI/ring crops in an ablation before claiming strictly local visual evidence.

**TLC features.** Start with circular hue distributions, saturation/value contrasts, color-gradient contrast, response occupancy, connected-component topology, elongation, solidity, and branching. Compare region with ring using histogram distances and signed as well as absolute differences. Hue descriptors are cyclic photometric features; their sign is not temperature order until calibrated. Normalize morphology by field/contact scale while retaining scale provenance. Area/intensity never becomes tumor-size output.

The included prototype implements circular interpolated/smoothed hue histograms, Jensen–Shannon divergence, saturation/value differences, disjoint ring selection, glare exclusion, optional DINO local distances, and per-scale aggregation. Component topology, registration, model ensembles, and device-observability estimation are specified extensions; they are not falsely reported as implemented in the prototype.

**Trainable components now.** None for the first local evidence maps. Then a strongly regularized logistic head on one to six prespecified subject-level summaries; the head is mouse-only. Do not fit a 384-dimensional covariance from one negative animal. Avoid a learned attention pool until fixed-pooling results justify it. Report regularization and fold-local scaling. Each subject contributes equal total weight regardless of patch/image count.

**Objectives.** For a local vector `u_j`, use fixed pooling initially: per-scale 90th percentile and median, then aggregate across scales. Compare this with top-three nonoverlapping components and max pooling under a fixed candidate budget. Fit a mouse head with subject-balanced binary cross-entropy plus ridge regularization. The provided small-head runner minimizes normalized class-balanced loss with ridge 1 and a weak intercept penalty; it does not recreate baseline A's scikit-learn objective.

For future constrained MIL, use `bag_logit = mean(top_k(w·u_j)) + b`, with `k=3` distinct components, and the same subject-level loss. Compare to fixed aggregate logistic before training attention. Do not supervise any individual patch as tumor from a bag label. If an independent lesion-location annotation exists, add a region-ranking term against a disjoint comparator, with its uncertainty/visibility flagged. Do not generate that annotation from the model's own heatmap.

For optional self-supervision, fit a residual adapter `z' = normalize(z + U V z)` with rank four, zero-initialized residual output, and frozen DINO weights. Use paired-patch consistency, variance/covariance anti-collapse terms and an identity penalty. An initial explicit objective is `25 L_consistency + 25 L_variance + L_covariance + 10 L_identity`, on normalized features with variance targets scaled to descriptor dimension. These are starting engineering constants, not tuned winners. Sample subjects uniformly, limit correlated patches per subject, fit inside each outer fold, and stop by held-out-subject acquisition consistency rather than disease labels. A lower loss alone does not justify promotion.

**Augmentations.** Geometric transforms must preserve image-mask alignment and candidate identity. Use modest rotations, translations, resampling, blur and JPEG compression for robustness sweeps; compare identity-only. Exposure and channel-gain changes begin as stress tests. Do not force invariance to arbitrary hue/saturation changes, aggressive crops that remove abnormalities, or temperature-changing transforms. Learn realistic photometric ranges from repeated acquisition/bench measurements when available. Synthetic localized patterns test code behavior only.

**Mouse labels.** Use the eight positives and one negative only for subject-level loss/ranking and explicit development probes. `0038` may populate a labelled mouse anchor in training folds, but its identity must remain visible in memory-bank provenance. Do not let its patches masquerade as independent normal animals. An all-subject deployment fit is kept separate from every fold model.

**Uncertainty and inference.** Return the evidence map, original-coordinate regions, feature channels, observable coverage, missing comparators, profile/domain status, perturbation spread, and model provenance. Perturbation min/max is an engineering stability envelope, not a confidence interval. Disagreement, insufficient coverage, unsupported profile/species, acquisition failure, or uncalibrated human disease head yields abstention. No response yields missing evidence, not `NO_TUMOR_LIKE`.

The human-facing binary contract is a later, separately calibrated two-threshold rule: score at/above `tau_high` -> `TUMOR_LIKE`; score below `tau_low` -> `NO_TUMOR_LIKE`; otherwise abstain. Both decisions require measurement eligibility and a matching human model. Until that calibration exists, the human path returns local research evidence with `research_binary_class=null`. The product's endpoint remains presence/absence; the abstention state represents inadequate support for that decision.

## E. Experimental matrix

### Evaluation units and isolation

Freeze the original nine as a **development cohort**, not a new independent test set. All views, masks, augmentations and derived embeddings of a subject belong together. Freeze extraction settings before inspecting candidate outcomes. Any choice changed after inspecting them is a new exploratory version. Preserve source/profile hashes, model revision, split IDs, exclusion reasons, feature definitions, and subject counts in every receipt.

For supervised A/B/C/D/F/G, use eight leave-one-positive-out folds: seven positives plus `0038` train; the excluded positive tests. Fit every learned scaler, projection, memory-bank selection, adapter, head and threshold inside the training portion. Never train the adapter on an excluded mouse “without its label” and call the result inductive holdout. Thresholds based on the training negative remain development thresholds.

Holding out `0038` leaves no negative training subject. Skip conventional binary fitting in that fold and report why; do not substitute a synthetic negative. A predefined label-free self-comparison model can score `0038` without fitting to it. That is one negative observation, not a validated specificity estimate.

### Ranked experiment schedule

| ID | Comparison / objective | Features and training | Measurements now | What it resolves |
|---|---|---|---|---|
| E0 | Reproduce saved A and historical rule | Existing artifacts; no new fit | Executed audit: 9/9 stored fit; historical rule 6/8 positives flagged | Establishes baseline and rejects low-response-implies-absence |
| E1 | Valid frame vs independently annotated contact | Fixed extraction, no disease training | Usable comparator fraction, annotation disagreement, map changes | Whether local controls actually represent tissue |
| E2 | B photometric vs B DINO vs F fusion | Frozen descriptors; zero-parameter maps, then <=6-feature ridge head | Per-subject scores, eight held-out-positive scores, negative training score separately | Whether local evidence adds signal beyond whole-image fitting |
| E3 | A full frame vs contact-only vs background-only | Same frozen backbone/head settings and folds | Held-out-positive margins, perturbation sensitivity, background dependence | Detects setup shortcuts; background control is not a healthy label |
| E4 | C mean vs max vs fixed top-three vs tiny attention | One bag per subject; class-balanced bag BCE and ridge | Sensitivity to bag size, region removal and held-out subjects | Whether MIL buys anything over simpler pooling |
| E5 | D `0038` anchor vs internal-only vs both | Cosine prototypes; no full covariance; equal per-subject contribution | Anchor deletion, comparator contamination sweep, scale sensitivity | Whether normal-anchor memorization dominates |
| E6 | E local consistency vs one-class normal bank | Label-free self-comparison; separate `0038` bank variant | Score on excluded negative without normal-bank fitting; maps and stability | Whether useful evidence can survive without the negative anchor |
| E7 | F morphology-only vs DINO-only vs fusion; retain global channel | Same subject aggregation and ridge; no size endpoint | LOPO positive counts and margins; individual failures, especially `0030/0031` | Whether biology is lost by local normalization |
| E8 | G frozen vs rank-four adapter | Paired transformations, consistency/anti-collapse/identity losses; no target holdout exposure | Same-task outputs, nuisance robustness, descriptor effective rank | Whether adaptation helps more than it distorts |
| E9 | H no mouse supervision vs mouse-derived representation | Human adapter frozen at evaluation; same human train/calibration/test IDs | Planned target learning curves, paired errors, coverage | Measures actual benefit or harm of mouse knowledge |

E1–E8 require the existing raw assets and contact annotations to produce biological-image results. The included prototype does not claim those experiments have run. Its eight passing tests exercise uniform fields, localized contrast, missing contact, dark unobservable fields, artifact exclusion, token handling and hue wrap-around using artificial pixels strictly as software tests.

### Leakage and falsification controls

- **Subject exclusion:** assert original hashes and subject IDs are disjoint across train/test, including derivative lineage. Perceptual-duplicate checks supplement hashes; a differently compressed copy is still the same acquisition.
- **Label permutation:** enumerate all nine choices of the sole negative, refitting the whole learned pipeline each time. Treat as a shortcut diagnostic; population significance is unavailable at this resolution, and session confounding can invalidate exchangeability.
- **Negative-anchor attack:** remove `0038` from the reference bank and compare a label-free score. If all structure disappears, the representation has not demonstrated useful internal comparison.
- **Matched artifact removal:** mask/reacquire glare or frame edges. Compare score change with that from removing the proposed evidence region. Occlusion can itself shift a DINO embedding; use matched control occlusions and do not claim causal localization.
- **Comparator contamination:** vary allowed remote/ring candidates, remove the most influential comparator, and deliberately include a known non-tissue region in a software stress test. Do not relabel tissue healthy during this test.
- **Scale and visibility:** inspect both missed positive subjects, but do not retune solely until they score positive. Record broad/diffuse, saturated and low-response failures explicitly.
- **Batch invariance:** duplicating or reordering frames must not increase independent evidence. Use fixed per-subject and per-view aggregation and candidate budgets.
- **Device/profile separation:** leave an entire profile/session out when such a dataset exists. Artificial color perturbations are not proof of genuine device generalization.

### Metrics and selection

Report one row per subject, number of acquisitions, valid/comparator coverage, individual fold scores, perturbation spread, and any independently annotated region overlap. Without location truth, report heatmaps but no lesion-localization accuracy. Runtime/memory should be measured on the current CPU deployment environment; do not extrapolate GPU paper timings.

Do not headline AUROC from eight positive comparisons with the same single negative. Do not bootstrap patches or resample the one negative to create a specificity interval. AUROC may be retained only as a clearly labelled descriptive development rank, alongside the nine individual values. LOPO training sets overlap, so report counts and variation as exploratory, not a simple independent binomial trial.

Use a lexicographic engineering selection rule: first pass provenance and missing-evidence behavior; next reject shortcut dependence; then compare stability and usable coverage; finally compare held-out-positive evidence. Prefer the simpler model unless an added component improves at least six of eight paired held-out-positive margins, has positive median improvement, and does not increase artifact dependence or unusable coverage. This is a prespecified **resource-allocation heuristic**, not statistical significance or a human-readiness gate. If no candidate passes, keep the sensing/local evidence engine and revise the disease formulation rather than promote a spurious winner.

### Ablation table template

| Run/version | Feature block | Train/test subject IDs | Positives tested | Negative tested independently | Held-out-positive results | Training-negative score | Usable coverage | Perturbation spread | Artifact dependence | Runtime | Decision |
|---|---|---|---:|---:|---|---|---|---|---|---|---|
| A-current | Global DINO | All nine fit; no held-out data | 0 | 0 | Not measured | 0.400324 stored | Not audited | Not measured | Not measured | Not measured | Baseline only |
| B-local | Photometric contrasts | Eight LOPO folds | Pending | 0 | Pending | Report separately | Pending | Pending | Pending | Pending | Pending |
| F-local | Photometric + DINO | Same folds | Pending | 0 | Pending | Report separately | Pending | Pending | Pending | Pending | Pending |
| H-human | Selected representation + human head | Frozen target split | Pending | Pending | Pending | Not a validation metric | Pending | Pending | Pending | Pending | Separate target gate |

## F. Mouse-to-human transfer strategy

### Transfer three things, relearn two

Transfer the code and tested priors for contact/observability assessment, multiscale local comparison, and frozen visual plus photometric representation. Relearn the human acquisition mapping and the human disease decision. Mouse-derived local representations are candidates; whether they help is an experimental question.

Use separate registries for `sensor_profile`, `representation`, `reference_bank`, and `decision_head`. A model signature includes species, anatomy, hardware, TLC formulation/batch, acquisition protocol, resolution/geometry, feature version, checkpoint hash, endpoint and calibration status. An unknown value is an explicit unsupported/uncertain condition, not a wildcard.

```text
Shared: geometry + observable-field logic + local features + uncertainty contract
   |
   +-- Mouse profile adapter -> mouse reference bank -> mouse development head
   |
   +-- Human profile adapter -> human local/paired/prior evidence -> human head
```

There is no claim that nine mouse subjects can disentangle biological species effects. Instead, nuisance removal is constrained by observations that preserve biology: repeated scans of the same region, bench measurements of the same surface through different sensing materials, or genuinely paired acquisitions. Domain prediction becoming harder is a diagnostic; it is not proof that tumor information survived.[6]

### What unlabelled human acquisitions do

On a dedicated adaptation pool, estimate contact/coverage patterns, artifact rates, field geometry, and repeatability. Train acquisition adapters with same-region paired consistency. Do not fit a one-class human normal bank from all unlabelled subjects. Retain separate pools for unknown biological status and reference-standard negative status. A spatial ring and a contralateral breast are comparators, not automatic disease-negative labels.

When both breasts are scanned, compare matching locations with registration confidence and preserve side-specific outputs. Bilateral disease and normal asymmetry require the single-breast local/global channels to remain active. For longitudinal comparison, use a frozen prior that was adjudicated suitable, flag treatment/cycle/protocol changes, and never automatically absorb a new suspicious scan into the baseline. At evaluation, a patient's permitted pre-index history may be inference input under a prespecified longitudinal protocol; their future scans cannot train the model or alter the past decision.

### Exact target adaptation ablation

Hold human development/calibration/test assignments fixed across four arms:

1. Frozen DINO + local physics, **no mouse labels used**.
2. The same representation plus mouse-trained small feature mapping; human head fit from scratch.
3. Arm 1 plus paired unlabelled human acquisition adapter.
4. Arm 3 plus the mouse mapping, with a shrinkage coefficient allowed to be zero.

Use identical human-label budgets, matching nuisance processing and head complexity. Human outcomes, not mouse performance, determine whether the mouse mapping survives. If arm 3 matches or beats arm 4 consistently, retire mouse disease supervision while retaining the successful sensor engineering.

Add CORAL/MMD only as a separate matched-condition experiment after target data exists; do not align the full diseased-mouse and unknown-human populations. Exclude online entropy-minimizing test-time learning initially: it can become confident in the wrong class. Deterministic per-acquisition normalization is different from learning on a test cohort.

### Physics as an engineering multiplier

Write the forward sensor contract now: calibrated surface field and geometry -> device RGB/response with uncertainty. Until calibration exists, leave the RGB-to-temperature inverse absent. A single image cannot generally identify both the surface thermal field and an unknown contact/color transfer function.

The most efficient later bench evidence is a controlled, non-biological surface scanned through the actual TLC stack across its operating range, including contact/angle/illumination changes and heating/cooling history. This can establish response range, repeatability and censoring without disease labels. Any heat-equation/optical simulation should test nuisance sensitivity and inverse-measurement behavior only. It must not provide synthetic tumor/healthy patient labels or claimed biological detection limits.

If the deployed human hardware outputs radiometric temperatures rather than TLC RGB, implement that sensor branch directly and share downstream local-comparison abstractions. Do not force it through an RGB TLC adapter merely because the mouse prototype used one. Public MumGuard material establishes a handheld thermographic product and AI workflow, not an exact private-to-public hardware equivalence.[19]

## G. Human acquisition plan

This is secondary to the build above. Its purpose is to obtain the missing evidence with as few disease labels as practicable.

### Collection order

1. **Recover current assets and acquisition annotations.** The original nine files, available acquisition logs and contact polygons unlock the next experiments without any additional biological subjects.
2. **Resolve the human sensor contract.** Obtain raw output examples and device/formulation/protocol metadata, plus repeat scans if already available. Build the adapter and observability checks on those unlabelled acquisitions.
3. **Acquire unlabelled target-domain diversity.** Sample across actual devices, operators, sessions and relevant anatomy; include repeated placement for measurement consistency. This trains sensing invariance, not disease calibration.
4. **Add outcome-linked human cases to a fixed development stream.** Include no-lesion controls, benign lesions, malignant lesions, and clinically relevant confounders. Label lesion presence separately from malignancy. Use routine reference standards rather than model scores.
5. **Calibrate and evaluate a locked candidate prospectively.** Freeze all learned components and thresholds before a distinct evaluation stream. Consecutive recruitment measures real operational coverage; enriched development sampling must not be used to estimate population predictive values.

### Minimum human data contract

| Group | Required fields / rule |
|---|---|
| Identity | Site-scoped pseudonymous `subject_id`; stable linkage across both breasts, views and visits; `exam_id`, `acquisition_id`, parent/derivative IDs; no names, phone numbers or unredacted embedded identifiers |
| Intended endpoint | `lesion_presence={present,absent,unknown}`, `malignancy={malignant,benign,unknown,not_applicable}`, lesion type and location if known; preserve contradictory/uncertain evidence |
| Reference standard | Source modality/report or pathology, date or relative interval, adjudicator role, verification status, follow-up basis for controls, lesion-to-side/location linkage; scans before diagnosis/treatment when possible |
| Acquisition | Species, anatomy, side, view/position, orientation, placement order, device/model/firmware, camera/exposure/WB, TLC formulation/batch/profile, protocol revision, operator/session/site IDs |
| Measurement | Ambient conditions, acclimatization/contact duration, contact pressure/quality if measurable, acquisition timestamp/elapsed time, raw units, bit depth, calibration ID and uncertainty; unknown values explicit |
| Relevant context | Age band, menstrual/menopausal/lactation status where appropriate, recent treatment/surgery, inflammation/fever or other recorded confounders; collect only justified fields with permission |
| QC | Pre-acquisition completion, post-acquisition blur/glare/contact/coverage flags, exclusion reasons, reacquisition relation, raw and reviewed masks, reviewer blinded-to-outcome flag |
| Governance | Consent/data-use/ethics provenance, permitted training/commercial/redistribution scopes, retention policy, access log, raw and derived checksums |
| Split | Subject-level development/adaptation/calibration/test allocation, prospective/retrospective flag, site/session/time boundary; all derivatives inherit allocation |

A “control” is a subject/breast meeting a prespecified reference-standard absence-of-target definition within a relevant interval. Self-report alone, a contralateral breast, or a nonmalignant lesion is not automatically that control. Resolve benign lesions into endpoint-specific classes rather than forcing an answer from a broad diagnosis label.

### Label efficiency and evidence size

Use the unlabelled pool to select development cases spanning acquisition diversity and model disagreement. Maintain a random/consecutive evaluation stream separately so active selection does not bias final performance. Ask for localized lesion annotations only where routine imaging supplies location; mask annotation and repeated scans often provide more immediate representation value than more bag labels.

For development learning curves, keep all existing validation subject IDs fixed and train each arm on nested subject subsets with approximately doubling label counts as cases arrive. Plot performance and coverage against **independent labelled subjects**, with paired errors and class-specific counts. Do not fit a precise asymptotic curve to two or three noisy points. Once the curve stops improving materially at the prespecified operational objective, redirect acquisition toward the dominant failure mode or measurement redesign.

For later fixed-rule validation, choose the acceptable error bound first. If there are zero false positives among `n` independent negative subjects, the exact one-sided 95% upper false-positive-rate bound is `1 - 0.05^(1/n)`. Thus 29 error-free negatives bound that rate below 10%, and 59 below 5%. The analogous calculation on positives bounds false negatives. These are conditional validation examples, not a sufficient training dataset or a clinical sample-size promise; any errors, subgroups, multiple devices, abstention selection and repeated interim looks require a different calculation.

To discriminate two candidates, use paired outcome differences on the same subjects. A planning approximation is `n ≈ (1.96+0.84)^2 (q-delta^2)/delta^2`, where `q` is their discordant-error rate and `delta` the improvement worth detecting. For an illustrative `q=0.20`, `delta=0.10`, this is about 149 independent subjects. Neither quantity is measured yet; estimate them from development, then plan the comparison. This explicitly avoids inventing a universal “20/20” minimum.

Do not select thresholds on the final test negatives and then use the zero-error formula on those same subjects. Report abstentions and failed acquisitions against all enrolled exams, plus conditional classification performance at the stated coverage. A system that abstains on nearly everyone has not solved the product objective.

## H. Execution plan

### Included executable package

The accompanying `MUMGUARD_RESEARCH_KIT.zip` contains additive files under `research/mumguard/` and this report. It does not replace the app or activate a registry entry.

| Included file | What executes now |
|---|---|
| `local_contrast.py` | Full-field normalization, explicit contact mask, glare exclusion, multiscale disjoint local comparison, circular hue/saturation/value features, optional aligned DINO descriptors, hashes and maps; always uncalibrated |
| `extract_tokens.py` | Optional full-grid DINOv2 extraction using the application's pinned official revision; requires PyTorch/backbone availability; not run on private assets here |
| `fit_subject_heads.py` | Small mouse-only B/F experiments from extracted evidence; subject grouping, fold-local scaling, eight LOPO folds, missing-DINO candidate skipping, no threshold promotion |
| `audit_current_evidence.py` | Executed audit of existing CSV/JSON evidence and sample-size arithmetic |
| `test_local_contrast.py` | Eight executed engineering tests on artificial pixel fields |
| `test_subject_heads.py` | Six executed head/grouping tests, including rejection of duplicate originals, benign-label conflation and human input in the mouse lane |
| `current_evidence_audit.json` | Executed counts, saved-fit margins, source hashes and fold design |
| `README.md` | Setup, exact commands, limitations and input contracts |

The current source checkout and runtime should be retained. Copy the additive kit into an isolated feature branch from the inspected `integration` state or rebase it onto the current branch after checking for changes. Proposed branch: `research/mumguard-local-contrast-v1`; PR target: `integration`. No push, merge or deployment is part of this delivered research program.

### Implementation sequence and acceptance

**Stage 1 — recover and make evidence inspectable.** Add `data/mumguard_acquisitions_v1.jsonl` and `scripts/validate_mumguard_acquisitions.py`, enforcing the contract above. Mount the nine existing source images; verify their subject mapping and hashes. Annotate contact/artifacts with class hidden. Produce local evidence/maps for every acquisition or an explicit failure reason. Use the supplied prototype to start E1/E2 immediately.

**Stage 2 — add reusable sensing components.** Add `app/services/contact_field.py`, `tlc_observability.py`, `local_contrast_features.py`, and `dinov2_patch_encoder.py`. Implement multi-mask handling, registered token extraction, profile calibration metadata and the optional component/topology branch. Keep the old 417-feature contract untouched; create a new `mumguard-local-v1` contract with named feature blocks and units. Never zero-fill a missing DINO/temperature channel into a model that requires it.

**Stage 3 — run the bounded mouse experiment.** Extend the supplied head runner into `scripts/run_mumguard_ablation.py` and `scripts/stress_mumguard_evidence.py`. Produce `artifacts/mumguard-local-v1/folds.json`, `feature_schema.json`, `subject_scores.csv`, `perturbation_report.json`, `model_card.json`, source/weight hashes, and selected candidate provenance. Compare A/B/F before spending on C/G. Acceptance is inspectable evidence with shortcut/coverage tests, not a mouse specificity claim.

**Stage 4 — integrate an inactive, domain-aware research result.** Add `app/services/mumguard_research_runtime.py` and tests for missing species, profile mismatch, incomplete field, unavailable embedding and invalid calibration. Route profile and acquisition type before circle detection. Run QC before disease scoring. Preserve original images, map transforms, feature channels, masks and versioned results in storage/history/report output. Add a new `local_research_evidence` object; do not overload the legacy `native_binary_research` payload.

**Stage 5 — prepare target adaptation.** Add `scripts/adapt_mumguard_sensor.py`, `scripts/train_mumguard_human_head.py`, `scripts/calibrate_mumguard_decision.py`, `data/mumguard_human_intake_template.csv`, and a locked split specification. Human training scripts reject missing endpoint mapping, false negative assignments from `BENIGN`, mixed device profiles without an explicit experiment, and adaptation/calibration/test overlap. These stages accept data when available; the current sensing build does not depend on finishing them.

### Components to preserve, extend, or retire

| Component | Decision |
|---|---|
| FastAPI/web UI, durable media, history/reports, deployment | Preserve; add evidence payload and overlays |
| TLC profile provenance and existing 417-feature baseline | Preserve/version; extend with actual sensor/species/endpoint signatures |
| Existing response segmentation and morphology | Preserve as feature/candidate channels; stop treating them as tissue-contact truth |
| DINOv2 | Preserve frozen backbone; add full-field tokens and local-crop ablation |
| Reference retrieval/publication assets | Preserve clearly labelled reference analysis; exclude from native training by default |
| Current 8/1 JSON head | Preserve as baseline and explicit mouse demo; do not promote to human inference |
| Implicit `BENIGN -> no tumor` mapping | Replace before human data use |
| Automatic transfer of mouse threshold, low-response negative rule | Retire from the new research design |

## I. Kill criteria

| Hypothesis | Rejection evidence | Next action |
|---|---|---|
| Local rings are informative | Most acquisitions lack credible contact comparators, or scores follow ring placement/artifacts more than internal structure | Keep global evidence; improve acquisition geometry/contact estimation; add valid repeat or bilateral comparisons |
| B/F beats global A | No consistent held-out-positive benefit under the prespecified heuristic, or gains vanish with contact-only masking | Retain the simpler/global branch; do not manufacture a local winner |
| `0038` defines useful normal reference | Removing it destroys ranking, or arbitrary unlabelled anchor choices produce comparable “performance” | Remove disease-normal-bank interpretation; use self-reference plus later human normal anchors |
| MIL finds biological regions | Attention concentrates on borders/glare, is unstable across folds, or increases with duplicated instances | Revert to fixed budget/pooling and revise candidate masks |
| SSL improves reusable representation | Better augmentation loss but worse evidence stability/held-out-positive behavior, collapsed dimensions or subject-identity shortcuts | Discard adapter, retain frozen DINO |
| Cross-species alignment is beneficial | Domain distinction falls but human error/coverage worsens; no gain versus the no-mouse arm | Remove alignment/mouse mapping and retain shared sensing code |
| IR teacher provides useful structure | Gains rely on raw temperature/session differences or disappear when evaluated in real target TLC | Retire teacher; retain IR experiment as a confound diagnostic |
| Current sensing captures the intended finding | Adequate, repeatable acquisitions show no usable relationship to the endpoint, including persistent known-positive misses | Reassess measurement modality/protocol; no architecture can recover a systematically unobserved signal |

Software rejection thresholds for coverage/stability should be set from engineering measurements and frozen before model comparison. Do not choose arbitrary sensitivity/specificity targets and claim they have been met on nine development subjects.

## J. Final recommendation

**Run contact-mask-aware B/F against A first, with a label-free local map and only a small mouse head.** This is the highest-value experiment because it addresses the observed low-response failures, exposes setup shortcuts, extracts reusable modality structure, and can be implemented with the existing images and pretrained weights. It remains useful if the mouse disease head ultimately proves nontransferable.

The human product should be built around an observable, profile-aware evidence representation with local, bilateral, longitudinal and global channels as available. Keep disease labels concentrated in a small human decision/calibration layer. The unexpected source of label efficiency is not generating more diseased-looking images: it is learning the sensor's nuisance response and using within-acquisition/repeated comparisons to avoid relearning that response from disease labels.

**Deliverable now:** a working local-evidence research path, strict data/endpoint contracts, bounded mouse ablations, a reusable inference payload, and an explicit target-transfer experiment. The provided executable prototype starts that path; 14 software tests and the committed-evidence audit passed. Biological image ablations, the optional DINO extraction run and target human experiments remain explicitly unexecuted in this delivery.

### What we still cannot know — irreducible later validation dependencies

The remaining irreducible evidence is **real acquisitions from the intended human device/protocol linked to an appropriate human reference standard**, held independently of model fitting and threshold selection. Unlabelled scans and bench calibration can reduce disease-label demand, but cannot establish whether an anomaly means a tumor-like finding in the intended human population.

That evidence must resolve three separate questions: whether the human sensing setup measures the relevant signal; whether the chosen features discriminate the prespecified endpoint from normal/benign/confounding patterns; and whether the locked decision rule works at useful coverage across intended users, devices and acquisition conditions. Use the uncertainty/learning-curve approach in section G to determine the required amount, rather than imposing an arbitrary cohort target.

There is no later dependency on obtaining a large new mouse cohort before starting this engineering program. Additional controls may inform the mouse baseline, but the human program can advance through reusable sensing, unlabelled target adaptation and a small human head without preserving the mouse classifier as its central architecture.

## Sources

1. Private project repository, inspected commits above: [engineering mandate](https://github.com/omarkhair70-droid/contact-thermography-ai/blob/05e37ef36b07638bb913e0585ace8990ddfd16af/docs/ASTRA_OMAR_ENGINEERING_MANDATE_2026-09-11.md), [research challenge](https://github.com/omarkhair70-droid/contact-thermography-ai/blob/05e37ef36b07638bb913e0585ace8990ddfd16af/docs/ASTRA_MUMGUARD_RESEARCH_CHALLENGE_2026-09-11.md), master/continuation handoffs, `app/services/{analysis_engine,dinov2_service,client_device_domain,client_device_qc,native_classifier_runtime}.py`, trainer scripts, source registries and the audit's hashed data/artifact inputs. Authenticated repository access required. Historical handoffs are superseded where they conflict with the resolved 8/1 map and presence/absence scope.
2. Damm, S. et al. [AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2](https://arxiv.org/abs/2405.14529), WACV 2025; [official Apache-2.0 code](https://github.com/dammsi/AnomalyDINO).
3. Schulthess, N.; Konukoglu, E. [Anomaly Detection by Clustering DINO Embeddings using a Dirichlet Process Mixture](https://papers.miccai.org/miccai-2025/paper/2425_paper.pdf), MICCAI 2025.
4. Ilse, M.; Tomczak, J.; Welling, M. [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html), ICML 2018.
5. Bardes, A.; Ponce, J.; LeCun, Y. [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906), ICLR 2022.
6. Zhao, H. et al. [On Learning Invariant Representations for Domain Adaptation](https://proceedings.mlr.press/v97/zhao19a/zhao19a.pdf), ICML 2019.
7. Song, C. et al. [Thermographic assessment of tumor growth in mouse xenografts](https://pubmed.ncbi.nlm.nih.gov/17487841/), International Journal of Cancer, 2007, DOI 10.1002/ijc.22808.
8. Wiberg, R.; Lior, N. [Errors in thermochromic liquid crystal thermometry](https://www.seas.upenn.edu/~lior/lior%20papers/Errors%20in%20thermochromic%20liquid%20crystal%20thermometry%20%28published%29.pdf), Review of Scientific Instruments, 2004, DOI 10.1063/1.1777406.
9. Mohd Sha'ari, A. B. et al. [Liquid crystal thermography for breast cancer detection: principles, current evidence, limitations, and future directions](https://pmc.ncbi.nlm.nih.gov/articles/PMC13373110/), 15 July 2026, DOI 10.1186/s43046-026-00383-6. Preliminary cohort observations are not validated diagnostic performance.
10. Braster S.A. [INNOMED/DeepBraster company report](https://www.braster.eu/media/wysiwyg/espi2019/BRASTER_RB_86_20191024_ESPI.pdf), 24 October 2019. Manufacturer disclosure, not independent MumGuard evidence.
11. Stevens, J. D.; Rogers, W. [Liquid Crystal Thermography of Transplantable Mouse Tumors, publisher issue listing](https://journals.sagepub.com/toc/vesa/5/4), Vascular Surgery 5(4), 1971, pp. 186–192, DOI 10.1177/153857447100500404. Restricted access; full raw-data evidence not obtained.
12. UFF Visual Lab. [DMR portal](https://visual.ic.uff.br/dmi/) and [original acquisition/project page](https://visual.ic.uff.br/en/proeng/thiagoelias/), accessed 11 September 2026. Modality/provenance source; website diagnostic claims are not adopted.
13. Ahmed, E. [lastdatabase - Copy.rar](https://figshare.com/articles/dataset/lastdatabase_-_Copy_rar/21225389), Figshare, 2022; license/file metadata verified through [Figshare API](https://api.figshare.com/v2/articles/21225389). Third-party DMR-IR mirror.
14. Rodriguez-Guerrero, S. et al. [Breast Thermography, version 3](https://data.mendeley.com/datasets/mhrt4svjxc/3), Mendeley Data, 5 February 2024. Dataset license: CC BY-NC 3.0.
15. Rodriguez-Guerrero, S. et al. [Dataset of breast thermography images for the detection of benign and malignant masses](https://pmc.ncbi.nlm.nih.gov/articles/PMC11131078/), Data in Brief 54, 2024, article 110503. Article license: CC BY-NC 4.0.
16. Monarin, B. P.; Pereira Neto, D. [Thermodataset: institutional record](https://educapes.capes.gov.br/handle/capes/1083394?mode=full) and [original UTFPR thesis](https://riut.utfpr.edu.br/jspui/bitstream/1/35532/1/mama.pdf), 2023. Raw-platform rights and current file availability remain unverified.
17. Meta. [Official DINOv2 repository](https://github.com/facebookresearch/dinov2), standard DINOv2 weights/code Apache 2.0. The newer X-ray-specific model has different terms and is not selected here.
18. Meta. [Official DINOv3 repository](https://github.com/facebookresearch/dinov3) and [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), license dated 19 August 2025.
19. MumGuard. [Product/device description](https://www.mumguard.net/), accessed 11 September 2026. Used for product workflow only; promotional sensitivity/size claims are not adopted.
