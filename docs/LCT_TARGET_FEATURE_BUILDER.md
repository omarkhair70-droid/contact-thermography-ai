# Contact-LCT target-native feature builder

This lane converts client-device contact-LCT images into the feature contract consumed by the native research trainer.

`lct-target-v1` contains 417 numeric features: 18 visible-response measurements, 8 engineering QC measurements, 7 morphology one-hot values, and a 384-dimensional normalized DINOv2 ViT-S/14 embedding.

The current builder is intentionally restricted to `tlc_profile_id=client-device-tlc-pending`. Hue/Lab/color features are profile-specific visible-response measurements and are not interpreted as absolute temperature while calibration is pending.

The nine current client mouse images remain `FROZEN_TARGET_EVAL`; feature extraction does not make them trainable or create a negative class. Native binary training remains gated until real same-species positive and negative/control target data exists with coherent TLC/device provenance.

Use the frozen Lane 5 embedding artifact when available to avoid rerunning DINOv2. The CLI can also invoke the pinned official DINO runtime when embeddings are not supplied.

`clinical_claim=NONE` remains mandatory.