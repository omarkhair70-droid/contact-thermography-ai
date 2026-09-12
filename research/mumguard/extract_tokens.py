"""Encode the full normalized image without center cropping. Optional torch dependency.

Use the normalized.png produced by local_contrast.py, then rerun with --tokens-npz.
The checkpoint and code revision match the existing official DINOv2 application path.
Weights download on first use. This script has not been run on private mouse images.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--normalized-image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    import torch
    rgb = np.asarray(Image.open(args.normalized_image).convert("RGB"))
    h, w = rgb.shape[:2]
    if h % 14 or w % 14:
        raise ValueError("normalized dimensions must be multiples of 14")
    model = torch.hub.load(f"facebookresearch/dinov2:{REVISION}", "dinov2_vits14",
                           pretrained=True, trust_repo=True).eval().to(args.device)
    tensor = torch.from_numpy(rgb.copy()).permute(2,0,1).float().div(255).unsqueeze(0)
    mean = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    std = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    with torch.inference_mode():
        features = model.forward_features(((tensor-mean)/std).to(args.device))
        tokens = torch.nn.functional.normalize(features["x_norm_patchtokens"], dim=-1)
    tokens = tokens[0].reshape(h//14, w//14, 384).float().cpu().numpy()
    if not np.isfinite(tokens).all():
        raise ValueError("non-finite DINO output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, tokens=tokens, input_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        backbone="dinov2_vits14", revision=REVISION)


if __name__ == "__main__":
    main()
