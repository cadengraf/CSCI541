"""Evaluate the authors' clean-VRU MSINet checkpoint on our fixed manifests."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from utils.core.datasets import ManifestItems
from .models import _cpu_latency


def load_model(repo: Path, checkpoint: Path):
    source = repo / "methods" / "MSINet"
    if not (source / "reid" / "models" / "msinet.py").is_file():
        raise FileNotFoundError(f"Missing MSINet authors' code: {source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing clean-VRU MSINet checkpoint: {checkpoint}")
    sys.path.insert(0, str(source))
    try:
        from reid.models.msinet import msinet_x1_0
    finally:
        sys.path.remove(str(source))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "module.classifier.weight" not in saved:
        raise ValueError("MSINet checkpoint is not the expected VRU state dictionary")
    classes = saved["module.classifier.weight"].shape[0]
    model = msinet_x1_0(SimpleNamespace(genotypes="msmt", pretrained=False,
                                       sam_mode="none", source_dataset="vru"), classes)
    weights = {key.removeprefix("module."): value for key, value in saved.items()}
    model.load_state_dict(weights, strict=True)
    return model.eval()


@torch.inference_mode()
def vectors(model, manifest: Path, batch_size: int, workers: int, device):
    data = ManifestItems(manifest)
    # Match the authors' test.py/build_data.py preprocessing: direct 256x256
    # resize and ImageNet normalization, without center cropping or flipping.
    data.transform = T.Compose((T.Resize((256, 256)), T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])))
    loader = DataLoader(data, batch_size=batch_size, num_workers=workers)
    output = []
    for images, _ in loader:
        output.append(F.normalize(model(images.to(device)).float(), dim=1).cpu())
    return torch.cat(output)


def active_parameters(model):
    # MSINet returns raw v/f_v at test time; these training-only heads are idle.
    unused = ("head.", "fc.", "classifier.", "f_head.", "f_fc.", "f_classifier.")
    return sum(p.numel() for name, p in model.named_parameters()
               if not name.startswith(unused))


def latency_ms(model):
    return _cpu_latency(model, image_size=256)
