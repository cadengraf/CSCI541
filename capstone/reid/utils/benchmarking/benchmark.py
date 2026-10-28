"""Run the local same-data benchmark and write HTML plus PNG figures."""

from types import SimpleNamespace

import torch

from .models import (ImageNetFeatures, _combine_pair_metrics, _cpu_latency,
                               _load_joint, _models, _vectors, retrieval_metrics)
from .report import _report
from utils.crossdrone.data import read_observations
from utils.crossdrone.retrieval import Observations, sample_rows
from utils.core.datasets import ForegroundVOC, ManifestItems
from utils.core.metrics import voc_mean_iou


def run(settings):
    if not settings["checkpoints"]:
        raise ValueError("benchmark.checkpoints must list at least one checkpoint")
    if settings["batch_size"] < 1 or settings["workers"] < 0:
        raise ValueError("benchmark batch_size must be positive and workers nonnegative")
    aicity_train = ManifestItems(settings["aicity_train"])
    aicity_query = ManifestItems(settings["aicity_query"])
    aicity_gallery = ManifestItems(settings["aicity_gallery"])
    if set(aicity_train.labels) & (set(aicity_query.labels) | set(aicity_gallery.labels)):
        raise ValueError("AICity benchmark identities overlap training identities")
    observations = read_observations(settings["bucktales_manifest"])
    requested_pair = settings.get("bucktales_pair")
    if requested_pair is not None:
        observations = [row for row in observations if row.pair == requested_pair]
    if not observations:
        raise ValueError(f"No BuckTales observations for {requested_pair}")
    sampled, selected = sample_rows(observations, settings["max_gallery_per_identity"])
    bucktales_query, bucktales_gallery = Observations(selected), Observations(sampled)
    bucktales_name = f"BuckTales {len({row.pair for row in observations})} cross-drone pairs"
    datasets = {"AICity cross-camera vehicles": (aicity_query, aicity_gallery),
                bucktales_name: (bucktales_query, bucktales_gallery)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voc = ForegroundVOC(settings["voc_root"], "val", download=False)
    rows = []
    for name, kind, path, global_pool in _models(settings):
        print(f"Benchmarking {name}", flush=True)
        model = (_load_joint(path, global_pool) if kind == "joint"
                 else ImageNetFeatures(kind, path))
        active_params = sum(p.numel() for key, p in model.named_parameters()
                            if not key.startswith("backbone.fc.") and
                            (getattr(model, "compute_mask", True) or
                             not key.startswith(("mask_head.", "mask_context."))))
        model.to(device).eval()
        row = {"name": name, "params": active_params, "datasets": {}, "voc_miou": None}
        for dataset_name, (query_data, gallery_data) in datasets.items():
            qv = _vectors(model, query_data, settings["batch_size"], settings["workers"], device)
            gv = _vectors(model, gallery_data, settings["batch_size"], settings["workers"], device)
            if dataset_name == bucktales_name:
                pair_results = {}
                for pair in sorted({item.pair for item in selected}):
                    qi = [i for i, item in enumerate(selected) if item.pair == pair]
                    gi = [i for i, item in enumerate(sampled) if item.pair == pair]
                    pair_results[pair] = retrieval_metrics(
                        qv[qi], [query_data.labels[i] for i in qi],
                        [query_data.cameras[i] for i in qi], gv[gi],
                        [gallery_data.labels[i] for i in gi],
                        [gallery_data.cameras[i] for i in gi])
                    metrics = pair_results[pair]
                    print(f"    {pair}: Rank-1 {metrics['rank1']:.2%}, "
                          f"mAP {metrics['map']:.2%} ({metrics['queries']} queries)", flush=True)
                row["bucktales_pairs"] = pair_results
                result = _combine_pair_metrics(pair_results)
            else:
                result = retrieval_metrics(qv, query_data.labels, query_data.cameras,
                                           gv, gallery_data.labels, gallery_data.cameras)
            row["datasets"][dataset_name] = result
            print(f"  {dataset_name}: Rank-1 {result['rank1']:.2%}, Rank-5 {result['rank5']:.2%}, "
                  f"mAP {result['map']:.2%} ({result['queries']} queries)", flush=True)
        if kind == "joint" and model.segmentation_trained:
            args = SimpleNamespace(batch_size=settings["batch_size"], workers=settings["workers"])
            row["voc_miou"] = voc_mean_iou(model, voc, args, device)
            print(f"  VOC 21-class mIoU: {row['voc_miou']:.2%}", flush=True)
        row["latency_ms"] = _cpu_latency(model)
        print(f"  Active parameters {active_params:,}; CPU latency {row['latency_ms']:.1f} ms/crop", flush=True)
        rows.append(row)
    output = settings["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.html")
    temporary.write_text(_report(rows, datasets, device), encoding="utf-8")
    temporary.replace(output)
    print(f"Saved {output}", flush=True)
    from .figures import save_figures
    paths = save_figures(rows, datasets, settings["images_dir"])
    for path in paths:
        print(f"Saved {path}", flush=True)
