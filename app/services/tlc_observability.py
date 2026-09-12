"""Observable color is not temperature or biological absence."""
import numpy as np
from PIL import Image


def observe(rgb, contact, exclusion):
    hsv = np.asarray(Image.fromarray(rgb).convert('HSV'))
    glare = (hsv[..., 2] >= 245) & (hsv[..., 1] <= 30)
    reliable = contact & ~exclusion & ~glare
    chromatic = reliable & (hsv[..., 1] > 55) & (hsv[..., 2] > 85)
    # 0 unknown/outside; 1 ambiguous contact; 2 chromatic contact; 3 artifact.
    states = np.zeros(contact.shape, np.uint8)
    states[reliable] = 1
    states[chromatic] = 2
    states[contact & (exclusion | glare)] = 3
    return states, {'contact_fraction': float(contact.mean()),
        'chromatic_contact_fraction': float(chromatic.sum()/max(1, contact.sum())),
        'artifact_contact_fraction': float((states == 3).sum()/max(1, contact.sum())),
        'temperature_calibration': 'UNAVAILABLE', 'absolute_temperature': None,
        'visibility': 'UNKNOWN', 'state_codes': {'unknown': 0, 'ambiguous': 1, 'chromatic': 2, 'artifact': 3}}
