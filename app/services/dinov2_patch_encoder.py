"""Pinned, full-field frozen patch descriptors; no disease head."""
import hashlib
import os

import numpy as np

REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def _validated_rgb(rgb):
    array = np.asarray(rgb)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[2] != 3
        or any(n % 14 for n in array.shape[:2])
    ):
        raise ValueError("patch input must be uint8 RGB with dimensions divisible by 14")
    return array


def encode_patches_batch(images, *, batch_size=None):
    """Encode several normalized MumGuard frames with fewer ViT forward passes.

    Human sessions commonly contain several overlapping tiles per side. Running one
    DINOv2 forward pass per tile made CPU-only Oracle inference take minutes. This
    path keeps the exact frozen patch semantics but batches equal-sized frames.
    """
    import torch

    from app.services.dinov2_service import runtime

    arrays = [_validated_rgb(image) for image in images]
    if not arrays:
        return []
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("batched patch inputs must have identical dimensions")

    runtime.load_official()
    configured = batch_size if batch_size is not None else os.getenv("DINOV2_BATCH_SIZE", "4")
    try:
        size = max(1, int(configured))
    except (TypeError, ValueError):
        size = 4

    results = []
    for start in range(0, len(arrays), size):
        chunk = arrays[start : start + size]
        t = torch.stack(
            [torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255) for array in chunk],
            dim=0,
        ).to(runtime.device)
        mean = t.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = t.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.inference_mode():
            z = runtime.model.forward_features((t - mean) / std)["x_norm_patchtokens"]
            z = torch.nn.functional.normalize(z, dim=-1)
        z = z.float().cpu().numpy()

        for offset, array in enumerate(chunk):
            tokens = z[offset].reshape(array.shape[0] // 14, array.shape[1] // 14, 384)
            if not np.isfinite(tokens).all():
                raise ValueError("non-finite patches")
            results.append(
                (
                    tokens,
                    {
                        "backbone": "dinov2_vits14",
                        "revision": REVISION,
                        "input_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                        "token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                        "context": "full-image ViT context; not strictly local crop descriptors",
                        "batched_inference": True,
                        "batch_size": len(chunk),
                    },
                )
            )
    return results


def encode_patches(rgb):
    """Backward-compatible single-frame wrapper."""
    return encode_patches_batch([rgb], batch_size=1)[0]
