"""Full-field geometry. Contact annotations are independent of color response."""
import hashlib
import io
import numpy as np
from PIL import Image, ImageOps


def normalize_field(source, contact=None, exclusion=None, size=224):
    if size < 196 or size % 14:
        raise ValueError("size must be a multiple of 14 >= 196")
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(source))).convert("RGB")
    w, h = image.size
    dims = (max(1, round(w * size / max(w, h))), max(1, round(h * size / max(w, h))))
    offset = ((size-dims[0])//2, (size-dims[1])//2)
    def canvas(im, mode, method):
        out = Image.new(mode, (size, size))
        out.paste(im.resize(dims, method), offset)
        return np.asarray(out)
    def mask(value):
        if value is None:
            return np.zeros((size, size), bool)
        value = np.asarray(value)
        if value.shape != (h, w):
            raise ValueError("mask must match EXIF-oriented original")
        return canvas(Image.fromarray((value.astype(bool)*255).astype('uint8')), 'L', Image.Resampling.NEAREST) > 127
    rgb = canvas(image, 'RGB', Image.Resampling.BILINEAR)
    frame = mask(np.ones((h, w), bool))
    return rgb, mask(contact), mask(exclusion), frame, {
        'source_sha256': hashlib.sha256(source).hexdigest(),
        'normalized_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'source_dimensions': [w, h], 'resize_dimensions': list(dims),
        'padding_xy': list(offset), 'normalized_dimensions': [size, size],
        'coordinate_rule': 'original_xy=(normalized_xy-padding_xy)*source_dimensions/resize_dimensions',
        'contact_annotation_status': 'MISSING' if contact is None else 'SUPPLIED',
    }


def original_box(candidate, transform):
    x, y, s = candidate['x'], candidate['y'], candidate['size']
    ox, oy = transform['padding_xy']
    rw, rh = transform['resize_dimensions']
    w, h = transform['source_dimensions']
    return [max(0., (x-s/2-ox)*w/rw), max(0., (y-s/2-oy)*h/rh),
            min(float(w), (x+s/2-ox)*w/rw), min(float(h), (y+s/2-oy)*h/rh)]
