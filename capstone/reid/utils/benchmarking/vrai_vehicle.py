"""Cross-camera retrieval on the identity-disjoint local VRAI train-data split."""

import csv
from pathlib import Path

import torch

from utils.core.datasets import ManifestItems
from .models import _cpu_latency, _load_joint, _vectors, retrieval_metrics


def run(settings):
    train = ManifestItems(Path(settings["train_manifest"]))
    query_path, gallery_path = Path(settings["query_manifest"]), Path(settings["gallery_manifest"])
    query = ManifestItems(query_path)
    gallery = ManifestItems(gallery_path)
    train_ids, query_ids, gallery_ids = set(train.labels), set(query.labels), set(gallery.labels)
    if train_ids & (query_ids | gallery_ids):
        raise ValueError("VRAI train identities overlap the final query/gallery split")
    if query_ids != gallery_ids:
        raise ValueError("VRAI final query/gallery must have identical identity sets")
    if any(camera is None for camera in (*query.cameras, *gallery.cameras)):
        raise ValueError("VRAI manifests must include camera IDs")
    if {path for path, _, _ in query.samples} & {path for path, _, _ in gallery.samples}:
        raise ValueError("VRAI query/gallery contain shared image files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    variants = [
        ("Vehicle ReID only", Path(settings["reid_only_checkpoint"]), False),
        ("Vehicle joint binary mask", Path(settings["joint_checkpoint"]), False),
        ("Vehicle joint binary mask (global pool)", Path(settings["joint_checkpoint"]), True),
    ]
    warm_paths = (Path(settings["warmstart_reid_only_checkpoint"]),
                  Path(settings["warmstart_joint_checkpoint"]))
    if all(path.is_file() for path in warm_paths):
        warm = [torch.load(path, map_location="cpu", weights_only=True) for path in warm_paths]
        metadata = [item.get("training", {}) for item in warm]
        if (all(item.get("finished") for item in metadata)
                and len({item.get("epoch") for item in metadata}) == 1
                and len({item.get("start_checkpoint") for item in metadata}) == 1):
            variants.extend((
                ("Vehicle warm-start ReID only", warm_paths[0], False),
                ("Vehicle warm-start joint", warm_paths[1], False),
                ("Vehicle warm-start joint (global pool)", warm_paths[1], True),
            ))
        else:
            print("Skipping unfinished VRAI warm-start comparison", flush=True)
    else:
        print("Skipping VRAI warm-start comparison until both checkpoints finish", flush=True)
    for name, path, global_pool in variants:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if not saved.get("training", {}).get("finished"):
            raise ValueError(f"Comparison checkpoint is unfinished: {path}")
        model = _load_joint(path, global_pool=global_pool).to(device).eval()
        preprocessing = "vehicle" if model.preprocessing == "vehicle_full_resize" else "imagenet"
        query_data = ManifestItems(query_path, preprocessing=preprocessing)
        gallery_data = ManifestItems(gallery_path, preprocessing=preprocessing)
        print(f"Extracting VRAI vectors: {name} ({device})", flush=True)
        query_vectors = _vectors(model, query_data, settings["batch_size"], settings["workers"], device)
        gallery_vectors = _vectors(model, gallery_data, settings["batch_size"], settings["workers"], device)
        metrics = retrieval_metrics(query_vectors, query.labels, query.cameras,
                                    gallery_vectors, gallery.labels, gallery.cameras)
        metrics.pop("cmc")
        metrics.update(model=name, parameters=sum(
            parameter.numel() for key, parameter in model.named_parameters()
            if not key.startswith("backbone.fc.") and
            (model.compute_mask or not key.startswith(("mask_head.", "mask_context.")))),
            cpu_ms=_cpu_latency(model))
        rows.append(metrics)
        print(f"  Rank-1 {metrics['rank1']:.2%}, mAP {metrics['map']:.2%}; "
              f"{metrics['cpu_ms']:.1f} ms/crop CPU", flush=True)

    output = Path(settings["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.csv")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "queries", "gallery", "rank1",
                                                  "rank5", "rank10", "map", "parameters", "cpu_ms"))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    print(f"Saved {output} (local split of VRAI train images; not official VRAI test)", flush=True)
