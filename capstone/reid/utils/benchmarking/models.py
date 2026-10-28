"""Comparable model loading, retrieval metrics, and CPU latency measurement."""

import hashlib
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (efficientnet_b0, get_model, get_model_weights,
                                mobilenet_v3_small, resnet18)

from utils.core.model import JointNet

MAX_RANK = 10


class ImageNetFeatures(nn.Module):
    """Frozen classifier features, evaluated with the same crops and cosine search."""

    def __init__(self, architecture, weights_path):
        super().__init__()
        builders = {"resnet18": resnet18, "mobilenetv3_small": mobilenet_v3_small,
                    "efficientnet_b0": efficientnet_b0}
        hashes = {
            "resnet18": "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8",
            "mobilenetv3_small": "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f",
            "efficientnet_b0": "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934",
        }
        if weights_path is None:
            # TorchVision's official weight enum verifies the URL hash on download.
            self.backbone = get_model(architecture, weights=get_model_weights(architecture).DEFAULT)
        else:
            self.backbone = builders[architecture](weights=None)
            digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            if digest != hashes[architecture]:
                raise ValueError(f"Unexpected hash for cached {architecture} weights: {weights_path}")
            # The official ResNet-18 cache entry uses legacy tar and requires the
            # already hash-verified legacy loader. The newer weights use safe loading.
            saved = torch.load(weights_path, map_location="cpu", weights_only=architecture != "resnet18")
            self.backbone.load_state_dict(saved)
        if architecture in ("resnet18", "regnet_y_400mf", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5"):
            self.backbone.fc = nn.Identity()
        elif architecture == "swin_t":
            self.backbone.head = nn.Identity()
        else:
            self.backbone.classifier[-1] = nn.Identity()

    def forward(self, images):
        features = self.backbone(images)
        return None, features.flatten(1)


def _load_joint(path, global_pool=False):
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark checkpoint: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    trained_mask = saved.get("segmentation_trained", False)
    model = JointNet(pretrained=False, use_mask_pool=trained_mask and not global_pool)
    model.load_state_dict(saved["model"])
    model.compute_mask = trained_mask
    model.segmentation_trained = trained_mask
    return model


@torch.inference_mode()
def _vectors(model, dataset, batch_size, workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    vectors = []
    for images, _ in loader:
        _, features = model(images.to(device))
        vectors.append(F.normalize(features.float(), dim=1).cpu())
    return torch.cat(vectors)


def retrieval_metrics(queries, query_labels, query_cameras, gallery,
                      gallery_labels, gallery_cameras):
    """CMC and AP with same-camera gallery images excluded per query."""
    if any(camera is None for camera in (*query_cameras, *gallery_cameras)):
        raise ValueError("Every benchmark crop needs a camera label")
    eligible_by_camera = {}
    for camera in set(query_cameras):
        eligible_by_camera[camera] = torch.tensor(
            [i for i, other in enumerate(gallery_cameras) if other != camera], dtype=torch.long)
    cmc = torch.zeros(MAX_RANK, dtype=torch.float64)
    ap_total = 0.0
    valid = 0
    for start in range(0, len(queries), 128):
        similarity = queries[start:start + 128] @ gallery.T
        for offset, scores in enumerate(similarity):
            index = start + offset
            eligible = eligible_by_camera[query_cameras[index]]
            if not len(eligible):
                continue
            matches = torch.tensor([gallery_labels[i] == query_labels[index]
                                    for i in eligible.tolist()], dtype=torch.bool)
            if not matches.any():
                continue
            ranked = matches[torch.argsort(scores[eligible], descending=True)]
            positive_ranks = torch.nonzero(ranked).flatten() + 1
            first_rank = int(positive_ranks[0])
            if first_rank <= MAX_RANK:
                cmc[first_rank - 1:] += 1
            precisions = torch.arange(1, len(positive_ranks) + 1, dtype=torch.float64) / positive_ranks
            ap_total += float(precisions.mean())
            valid += 1
    if not valid:
        raise ValueError("No query has a cross-camera positive in the gallery")
    curve = (cmc / valid).tolist()
    return {"queries": valid, "gallery": len(gallery), "rank1": curve[0],
            "rank5": curve[4], "rank10": curve[9], "map": ap_total / valid,
            "cmc": curve}


def _combine_pair_metrics(metrics):
    """Pool cross-drone queries without mixing unrelated drone pairs in a gallery."""
    total = sum(item["queries"] for item in metrics.values())
    if not total:
        raise ValueError("No BuckTales query has a cross-drone positive")
    average = lambda key: sum(item[key] * item["queries"] for item in metrics.values()) / total
    curve = [sum(item["cmc"][rank] * item["queries"] for item in metrics.values()) / total
             for rank in range(MAX_RANK)]
    return {"queries": total, "gallery": sum(item["gallery"] for item in metrics.values()),
            "rank1": curve[0], "rank5": curve[4], "rank10": curve[9],
            "map": average("map"), "cmc": curve}


@torch.inference_mode()
def _cpu_latency(model):
    model.to("cpu").eval()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sample = torch.randn(1, 3, 224, 224)
        for _ in range(5):
            model(sample)
        timings = []
        for _ in range(25):
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1000)
        return statistics.median(timings)
    finally:
        torch.set_num_threads(previous_threads)


def _models(settings):
    models = [(name, "joint", path, False) for name, path in settings["checkpoints"].items()]
    matched = settings.get("matched_checkpoints", {})
    if matched:
        if all(path.is_file() for path in matched.values()):
            saved = [torch.load(path, map_location="cpu", weights_only=True)
                     for path in matched.values()]
            metadata = [item.get("training", {}) for item in saved]
            if (all(item.get("finished") for item in metadata) and
                    len({item.get("epoch") for item in metadata}) == 1 and
                    len({item.get("start_checkpoint") for item in metadata}) == 1):
                models.extend((name, "joint", path, False) for name, path in matched.items())
            else:
                print("Skipping matched arms: both must finish the same epoch from the same start", flush=True)
        else:
            print("Skipping matched arms: run `python train.py compare` to finish both", flush=True)
    for name in settings.get("global_pool_ablation", []):
        if name not in settings["checkpoints"]:
            raise ValueError(f"Unknown pooling ablation checkpoint: {name}")
        models.append((f"{name} (global pool)", "joint", settings["checkpoints"][name], True))
    if settings.get("compare_global_pool") and not settings.get("global_pool_ablation"):
        name, _, path, _ = models[0]
        models.append((f"{name} (global pool)", "joint", path, True))
    if settings["compare_resnet18_imagenet"]:
        cache = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-5c106cde.pth"
        if not cache.is_file():
            print(f"Skipping ImageNet ResNet-18: missing cached weights at {cache}", flush=True)
        else:
            models.append(("ResNet-18 ImageNet features", "resnet18", cache, False))
    directory = Path(__file__).resolve().parent.parent / "checkpoints" / "baselines"
    lightweight = (
        ("compare_mobilenetv3_small_imagenet", "MobileNetV3-Small ImageNet features",
         "mobilenetv3_small", "mobilenet_v3_small-047dcff4.pth"),
        ("compare_efficientnet_b0_imagenet", "EfficientNet-B0 ImageNet features",
         "efficientnet_b0", "efficientnet_b0_rwightman-7f5810bc.pth"),
    )
    for setting, name, kind, filename in lightweight:
        if settings[setting]:
            path = directory / filename
            if path.is_file():
                models.append((name, kind, path, False))
            else:
                print(f"Skipping {name}: missing {path}", flush=True)
    for architecture in settings.get("extra_imagenet_models", []):
        label = {
            "shufflenet_v2_x1_0": "ShuffleNetV2 x1.0",
            "shufflenet_v2_x1_5": "ShuffleNetV2 x1.5",
            "mobilenet_v2": "MobileNetV2",
            "mobilenet_v3_large": "MobileNetV3-Large",
            "efficientnet_b2": "EfficientNet-B2",
            "regnet_y_400mf": "RegNetY-400MF",
            "convnext_tiny": "ConvNeXt-Tiny",
            "swin_t": "Swin-T",
        }.get(architecture)
        if label is None:
            raise ValueError(f"Unsupported benchmark model: {architecture}")
        models.append((f"{label} ImageNet features", architecture, None, False))
    return models
