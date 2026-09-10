from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
DINO_DIR = ROOT / "artifacts" / "dinov2_reference"

REFERENCE_INDEX = pd.read_csv(DINO_DIR / "plate_index.csv")
REFERENCE_EMBEDDINGS = np.load(DINO_DIR / "dinov2_embeddings.npy")
FUSED_REFERENCE_HEAD = joblib.load(DINO_DIR / "fused_lct_dinov2_reference_head.joblib")
BILATERAL_REFERENCE_HEAD = joblib.load(DINO_DIR / "bilateral_dinov2_reference_head.joblib")

class DINOv2Runtime:
    """Optional live encoder for newly uploaded images."""
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None

    @property
    def ready(self):
        return self.model is not None

    def load_official(self, model_name="dinov2_vits14", device=None):
        import torch
        from torchvision import transforms
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225),
            ),
        ])
        return self

    def encode_rgb(self, rgb):
        if not self.ready:
            raise RuntimeError("DINOv2 live backbone is not loaded")
        import torch
        from PIL import Image
        x = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            z = self.model(x)
            z = torch.nn.functional.normalize(z, dim=-1)
        return z.squeeze(0).float().cpu().numpy()

runtime = DINOv2Runtime()
