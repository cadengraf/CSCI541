"""VRU full-test vehicle retrieval; shared images and scoring for all models."""

import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.core.datasets import ManifestItems
from .models import _cpu_latency, _load_joint, _vectors
from .msinet_baseline import active_parameters, latency_ms, load_model, vectors


def single_gallery_metrics(query_vectors, query_ids, gallery_vectors, gallery_ids):
    """Rank-k and mAP for one gallery image per identity (AP = reciprocal rank)."""
    if len(query_vectors) != len(query_ids) or len(gallery_vectors) != len(gallery_ids):
        raise ValueError("Feature and manifest lengths differ")
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("VRU protocol requires exactly one gallery image per identity")
    positions = {identity: i for i, identity in enumerate(gallery_ids)}
    if set(query_ids) != set(gallery_ids):
        raise ValueError("Query and gallery identities must match")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = F.normalize(torch.as_tensor(query_vectors, dtype=torch.float32, device=device), dim=1)
    g = F.normalize(torch.as_tensor(gallery_vectors, dtype=torch.float32, device=device), dim=1)
    if q.shape[1] != g.shape[1] or not torch.isfinite(q).all() or not torch.isfinite(g).all():
        raise ValueError("Embeddings need matching dimensions and finite values")
    correct = torch.tensor([positions[identity] for identity in query_ids], device=device)
    gallery_indices = torch.arange(len(g), device=device)
    ranks = []
    for start in range(0, len(q), 128):
        scores = q[start:start + 128] @ g.T
        match = correct[start:start + len(scores)]
        true = scores.gather(1, match[:, None])
        # Tie policy is deterministic and depends only on gallery CSV order.
        rank = 1 + (scores > true).sum(1) + ((scores == true) &
                (gallery_indices[None, :] < match[:, None])).sum(1)
        ranks.append(rank)
    rank = torch.cat(ranks).float()
    return {"queries": len(q), "gallery": len(g),
            "rank1": (rank <= 1).float().mean().item(),
            "rank5": (rank <= 5).float().mean().item(),
            "rank10": (rank <= 10).float().mean().item(),
            "map": rank.reciprocal().mean().item()}


def _paths(dataset):
    return [str(path) for path, _, _ in dataset.samples]


def _external_vectors(path, query_data, gallery_data):
    """Load baseline vectors tied to exact manifest image order."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or set(saved) != {"query_paths", "gallery_paths", "query", "gallery"}:
        raise ValueError(f"{path}: expected query_paths, gallery_paths, query, gallery")
    if saved["query_paths"] != _paths(query_data) or saved["gallery_paths"] != _paths(gallery_data):
        raise ValueError(f"{path}: image order differs from VRU manifests")
    return saved["query"], saved["gallery"]


def _add_matched_group(checkpoints, group, label):
    if not group:
        return
    paths = [Path(path) for path in group.values()]
    if not all(path.is_file() for path in paths):
        print(f"Skipping {label}: both arms are not available yet", flush=True)
        return
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    metadata = [item.get("training", {}) for item in saved]
    if (all(item.get("finished") for item in metadata)
            and len({item.get("epoch") for item in metadata}) == 1
            and len({item.get("start_checkpoint") for item in metadata}) == 1):
        checkpoints.update(group)
    else:
        print(f"Skipping {label}: both arms must finish the same epoch and start checkpoint", flush=True)


def run(settings):
    query_path = Path(settings["query_manifest"])
    gallery_path = Path(settings["gallery_manifest"])
    if not query_path.is_file() or not gallery_path.is_file():
        raise FileNotFoundError("Run `python prepare.py vru` first")
    query_data, gallery_data = ManifestItems(query_path), ManifestItems(gallery_path)
    if set(_paths(query_data)) & set(_paths(gallery_data)):
        raise ValueError("Query/gallery images overlap")
    train_path = Path(settings["train_manifest"])
    with train_path.open(newline="") as file:
        train_ids = {row["id"] for row in csv.DictReader(file)}
    if train_ids & (set(query_data.labels) | set(gallery_data.labels)):
        raise ValueError("VRU training identities leaked into test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    checkpoints = dict(settings["checkpoints"])
    matched = settings.get("matched_checkpoints", {})
    warmstart = settings.get("warmstart_checkpoints", {})
    _add_matched_group(checkpoints, matched, "vehicle comparison")
    _add_matched_group(checkpoints, warmstart, "vehicle warm-start")
    variants = [(name, Path(path), False) for name, path in checkpoints.items()]
    for name in settings.get("global_pool_ablation", []):
        if name not in checkpoints and name in (matched | warmstart):
            continue  # Optional matched arm has not finished yet.
        if name not in checkpoints:
            raise ValueError(f"Unknown VRU global-pool checkpoint: {name}")
        variants.append((f"{name} (global pool)", Path(checkpoints[name]), True))
    for name, checkpoint, global_pool in variants:
        print(f"Extracting VRU vectors: {name} ({device})", flush=True)
        model = _load_joint(checkpoint, global_pool=global_pool).to(device).eval()
        if model.preprocessing == "vehicle_full_resize":
            query_inputs = ManifestItems(query_path, preprocessing="vehicle")
            gallery_inputs = ManifestItems(gallery_path, preprocessing="vehicle")
        else:
            query_inputs, gallery_inputs = query_data, gallery_data
        qv = _vectors(model, query_inputs, settings["batch_size"], settings["workers"], device)
        gv = _vectors(model, gallery_inputs, settings["batch_size"], settings["workers"], device)
        result = single_gallery_metrics(qv, query_data.labels, gv, gallery_data.labels)
        result.update(model=name, parameters=sum(p.numel() for key, p in model.named_parameters()
                                                 if not key.startswith("backbone.fc.") and
                                                 (model.compute_mask or not key.startswith(
                                                     ("mask_head.", "mask_context.")))),
                      cpu_ms=_cpu_latency(model))
        rows.append(result)
        print(f"  Rank-1 {result['rank1']:.2%}, Rank-5 {result['rank5']:.2%}, "
              f"mAP {result['map']:.2%}; {result['cpu_ms']:.1f} ms/crop CPU", flush=True)
        del model, qv, gv
    baseline = settings.get("msinet")
    if baseline and baseline.get("enabled"):
        name = "MSINet VRU clean (authors' checkpoint)"
        print(f"Extracting VRU vectors: {name} ({device})", flush=True)
        model = load_model(Path(baseline["repo"]), Path(baseline["checkpoint"])).to(device)
        qv = vectors(model, query_path, settings["batch_size"], settings["workers"], device)
        gv = vectors(model, gallery_path, settings["batch_size"], settings["workers"], device)
        result = single_gallery_metrics(qv, query_data.labels, gv, gallery_data.labels)
        result.update(model=name, parameters=active_parameters(model), cpu_ms=latency_ms(model))
        rows.append(result)
        print(f"  Rank-1 {result['rank1']:.2%}, Rank-5 {result['rank5']:.2%}, "
              f"mAP {result['map']:.2%}; {result['cpu_ms']:.1f} ms/crop CPU", flush=True)
        del model, qv, gv
    for name, path in settings.get("external_features", {}).items():
        qv, gv = _external_vectors(Path(path), query_data, gallery_data)
        result = single_gallery_metrics(qv, query_data.labels, gv, gallery_data.labels)
        result.update(model=name, parameters="", cpu_ms="")
        rows.append(result)
        print(f"  {name}: Rank-1 {result['rank1']:.2%}, Rank-5 {result['rank5']:.2%}, "
              f"mAP {result['map']:.2%} (external vectors)", flush=True)
    output = Path(settings["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".partial.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "queries", "gallery", "rank1",
                                                  "rank5", "rank10", "map", "parameters", "cpu_ms"))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".partial.csv").replace(output)
    print(f"Saved {output}", flush=True)
