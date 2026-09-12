"""Pinned, full-field frozen patch descriptors; no disease head."""
import hashlib
import numpy as np

REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'


def encode_patches(rgb):
    import torch
    from app.services.dinov2_service import runtime
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or any(n % 14 for n in rgb.shape[:2]):
        raise ValueError('patch input must be uint8 RGB with dimensions divisible by 14')
    runtime.load_official()
    t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(runtime.device)
    mean = t.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = t.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    with torch.inference_mode():
        z = runtime.model.forward_features((t-mean)/std)['x_norm_patchtokens']
        z = torch.nn.functional.normalize(z, dim=-1)
    z = z[0].reshape(rgb.shape[0]//14, rgb.shape[1]//14, 384).cpu().numpy()
    if not np.isfinite(z).all():
        raise ValueError('non-finite patches')
    return z, {'backbone': 'dinov2_vits14', 'revision': REVISION,
               'input_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
               'token_sha256': hashlib.sha256(z.tobytes()).hexdigest(),
               'context': 'full-image ViT context; not strictly local crop descriptors'}
