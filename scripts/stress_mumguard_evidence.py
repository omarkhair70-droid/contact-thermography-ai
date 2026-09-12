"""Deterministic nuisance probes, never synthetic biological labels."""
import io
import cv2
import numpy as np
from PIL import Image, ImageFilter


def conservative_support(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Optical support only: deliberately excludes dark/out-of-range regions.
    # It cannot establish contact or absence and is NOT an independent mask.
    raw = ((hsv[..., 1] > 55) & (hsv[..., 2] > 85) & ~((hsv[..., 2] >= 245) & (hsv[..., 1] <= 30))).astype('uint8')
    return cv2.erode(raw, np.ones((3, 3), np.uint8)) > 0


def variants(rgb):
    image = Image.fromarray(rgb)
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=75)
    return {'identity': rgb, 'jpeg75': np.asarray(Image.open(io.BytesIO(stream.getvalue())).convert('RGB')),
            'blur05': np.asarray(image.filter(ImageFilter.GaussianBlur(.5))),
            'exposure095': np.clip(rgb.astype(float)*.95, 0, 255).astype('uint8')}
