# LCT Open Reference Pack v1

Base integration SHA: `86bf7d60fe858b4009fb272f2e71da7767290c86`

Purpose: materialize the explicitly open-license, pathology-linked contact-LCT exemplars already registered by Lane M without copying Braster-restricted images or silently turning reference material into native training data.

## Pack contents

The canonical asset manifest is:

`data/lct_open_reference_assets_v1.csv`

It contains exactly three source figures:

1. Ćwierz et al. Figure 3 — contact thermography from a biopsy-confirmed grade-2 invasive ductal carcinoma case. The publisher article is open access under a Creative Commons Attribution license.
2. Mohd Sha'ari et al. 2026 Figure 3 — source figure for the Malaysian cohort true-positive case with biopsy-confirmed invasive breast carcinoma, no special type. The article is CC BY 4.0.
3. Mohd Sha'ari et al. 2026 Figure 4 — source figure for the Malaysian cohort true-negative case with biopsy-confirmed fibrocystic change. The article is CC BY 4.0.

The Malaysian figures are multimodal composites. They must not be passed wholesale to a contact-LCT image model as if every panel were LCT. The pack records them as `SOURCE_FIGURE`; any later LCT-only derivative crop must preserve attribution, record that it is a derivative, and be manually checked against the source figure.

## Reproducible fetch

Use:

```bash
python scripts/fetch_lct_open_reference_assets.py \
  data/lct_open_reference_assets_v1.csv \
  --out runtime/lct_open_reference_assets
```

A dry run validates the manifest without network access:

```bash
python scripts/fetch_lct_open_reference_assets.py \
  data/lct_open_reference_assets_v1.csv \
  --dry-run
```

The fetcher:

- accepts only `OPEN_LICENSE_REFERENCE` rows;
- requires an explicit `CC-BY*` license status and license-evidence URL;
- refuses `train_eligible=true`;
- downloads only the URLs in the open manifest;
- writes a SHA-256/size/license/attribution `download_manifest.json` beside fetched assets;
- does not fetch any Braster copyright-restricted case-study image.

## Scientific role

These assets are for reference, morphology review, software validation, and provenance-preserving research inspection. They are not a coherent train/validation cohort and they do not satisfy the native binary readiness gate.

Reasons:

- all three are human cases, while the current client target cohort is mouse;
- TLC/device/acquisition profiles are not the same as the client device;
- there are only three open figures and only one benign exemplar;
- the nine client mouse images remain frozen positive/evaluation material;
- no genuine same-device negative/control mouse cohort is present.

Therefore:

- `train_eligible=false` remains mandatory for this pack;
- `clinical_claim=NONE` remains mandatory;
- no `TUMOR_LIKE / NO_TUMOR_LIKE` native mouse head may be trained from this pack.

## Negative/control hunt status — 2026-09-11

Structured status is recorded in:

`data/lct_negative_control_hunt_20260911.csv`

Current result:

- USM remains the strongest active human contact-LCT data-access lead. The 2026 review describes an ongoing 108-participant target with 42 completed at submission and includes a biopsy-confirmed benign true-negative exemplar, but the underlying cohort images are not publicly released.
- ThermaALG/Braster provides strong evidence that real human control cohorts exist in contact-LCT research, but raw case-labelled thermograms have not been found as an openly downloadable dataset.
- Stevens & Rogers (1971) remains the closest mouse/contact-LCT precedent. The available record describes 30 tumour-bearing mice and comparison with normal tissue, not a reusable matched negative/control animal image cohort; publisher image rights are restricted.
- A bounded repository-focused search pass did not surface a suitable openly downloadable case-labelled contact-LCT negative/control dataset. This is not an assertion that none exists anywhere.

## Next acquisition action

Continue source-specific data access rather than weakening the model gate:

1. USM investigators — request de-identified original RGB LCT images plus stable case IDs, pathology/reference-standard provenance, TLC foil/formulation, device/acquisition metadata, and explicit model-development/commercial-use terms.
2. ThermaALG / Jagiellonian / Braster-rightsholder route — request de-identified raw thermograms and control/cancer case mapping with explicit rights.
3. Mouse historical archive route — ask whether original Stevens/Rogers plates/photographs and any control-animal material survive and can be licensed for research/model development.
4. Client route — request genuine same-device control/negative mice and the tumour-size mapping for the frozen nine positives.

Do not send external outreach automatically. Drafting is allowed; sending requires explicit user approval.

## Source evidence

- Ćwierz et al. article and Figure 3: https://sciforschenonline.org/journals/breast-cancer-research-advancements/JBCRA-1-107.php
- Mohd Sha'ari et al. 2026 article, Figures 3 and 4, and CC BY 4.0 rights statement: https://doi.org/10.1186/s43046-026-00383-6
- USM Radiology dissertation listing: https://radiology.kk.usm.my/index.php/penyelidikan/disertasi-thesis
- USM pathology programme listing: https://medic.usm.my/?id=64&view=category
- Hodorowicz-Zaniewska et al. prospective Braster pilot: https://doi.org/10.1177/1534735420915778
- Braster ThermaALG public study summary: https://www.braster.eu/en/system-braster/clinical-trials
- Stevens & Rogers mouse contact-LCT paper: https://doi.org/10.1177/153857447100500404
