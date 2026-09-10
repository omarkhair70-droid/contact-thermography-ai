# Mouse tumor thermography transfer notes

`clinical_claim = NONE`

## Why this matters for the client cohort

The client data are contact-LCT images of tumor-bearing mice, not human infrared breast thermograms. Public evidence directly relevant to mouse tumor thermography therefore matters more for feature design than assuming that every tumor must simply appear hotter.

## Contact-LCT precedent

Stevens and Rogers (1971), *Liquid Crystal Thermography of Transplantable Mouse Tumors*, used cholesteric liquid crystals for direct visual thermal observations over experimentally transplanted tumors and normal tissue. The reported experiment included 20 C3H mice with mammary adenocarcinoma and 10 A/Jax mice with hepatoma; tumors were allowed to reach a palpable size before testing. Raw images are not available as an open dataset, so this source is methods/biology evidence only.

## Mouse IR tumor-burden evidence

Tepper et al. (2013) followed DA3 breast carcinoma tumors in 12 Balb/c mice over several weeks. For larger tumors, image-derived tumor area correlated with manual caliper measurements, and relative tumor temperature changed with tumor area and treatment state. This supports keeping **size/area and relative-pattern features** as a dedicated client-mouse task once the client's tumor-size mapping arrives.

## Sign-direction warning

Song et al. (2007) reported MDA-MB-231 and MCF7 breast cancer xenografts in mice as consistently **cooler** than surrounding skin rather than hotter. This is important for our model design: a mouse tumor detector must not hard-code `tumor = hotter region`. The sign and spatial pattern can depend on tumor model, perfusion, necrosis, growth stage, treatment, and acquisition modality.

## Model consequences

For the client contact-LCT cohort:

1. Use relative local contrast and morphology in addition to absolute color/temperature-like response.
2. Preserve both positive and negative response directions where the TLC profile permits them; do not collapse the representation to a single hyperthermia score.
3. Evaluate tumor-burden mapping separately from binary tumor/no-tumor classification.
4. Prefer same-animal baseline/control or same-device non-tumor mice as the negative domain.
5. Keep IR mouse literature as auxiliary biological evidence only; it does not substitute for contact-LCT labels.

## Current decision

The target binary classifier remains blocked by the absence of a coherent same-domain negative/control client-mouse pool. The next highest-value data are control or pre-injection client-device contact-LCT images acquired under the same setup. Meanwhile, public contact-LCT sources continue to be collected as labelled reference material where rights and case linkage permit.
