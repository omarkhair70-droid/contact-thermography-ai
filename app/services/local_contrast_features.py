"""Versioned local blocks; preserve the independent legacy 417-feature contract."""
import numpy as np
from research.mumguard.local_contrast import analyze
from research.mumguard.fit_subject_heads import FEATURES, features

CONTRACT = 'mumguard-local-v1'


def schema():
    return {'version': CONTRACT, 'features': [
        {'name': name, 'unit': 'nats' if name == 'hue_js_q90' else 'dimensionless',
         'required_channel': 'dinov2' if name.startswith('dino') else 'photometric'} for name in FEATURES],
        'missing_channel_policy': 'skip dependent candidate; never zero-fill',
        'temperature_channel': None, 'clinical_claim': 'NONE'}


def extract(rgb, contact, exclusion, tokens=None):
    result, heat, support = analyze(rgb, contact, exclusion, tokens=tokens)
    result['feature_contract_version'] = CONTRACT
    vector = features(result) if result['candidates'] else np.full(len(FEATURES), np.nan)
    result['feature_channels'] = {k: float(v) if np.isfinite(v) else None for k, v in zip(FEATURES, vector)}
    return result, heat, support
