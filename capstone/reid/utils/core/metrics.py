"""Checkpoint-selection, mask IoU, and image retrieval metrics."""

import math
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def selection_score(metrics, baseline, settings):
    """Return a relative score only when preservation floors are met."""
    weights = {name: weight for name, weight in settings["weights"].items() if name in metrics}
    if not weights:
        raise ValueError("No configured selection metric is available")
    failed = [name for name, ratio in settings["minimum_ratio"].items()
              if name in metrics and metrics[name] < baseline[name] * ratio]
    if failed:
        return None, failed
    score = sum(weight * math.log((metrics[name] + 1e-6) / (baseline[name] + 1e-6))
                for name, weight in weights.items()) / sum(weights.values())
    return score, []


@torch.inference_mode()
def voc_mean_iou(model, dataset, args, device):
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    confusion = torch.zeros((21, 21), dtype=torch.long)
    for images, target in loader:
        logits, _ = model(images.to(device))
        truth = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
        prediction = logits.argmax(dim=1).cpu()
        valid = truth != 255
        pairs = 21 * truth[valid] + prediction[valid]
        confusion += torch.bincount(pairs, minlength=21 * 21).reshape(21, 21)
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    present = union > 0
    return (intersection[present] / union[present]).float().mean().item()


@torch.inference_mode()
def selection_metrics(model, datasets, args, device):
    model.eval()
    results = {}
    for name, queries, gallery in datasets["cross_view"]:
        qv = image_vectors(model, queries, args, device)
        gv = image_vectors(model, gallery, args, device)
        _, _, mean_ap = cross_view_metrics(
            qv, queries.labels, queries.cameras, gv, gallery.labels, gallery.cameras)
        results[f"{name}_map"] = mean_ap
    for name, queries, gallery in datasets["retrieval"]:
        qv = image_vectors(model, queries, args, device)
        if gallery is None:
            correct, total = recall_at_one(qv, queries.labels, qv, queries.labels, True)
        else:
            gv = image_vectors(model, gallery, args, device)
            correct, total = recall_at_one(qv, queries.labels, gv, gallery.labels, False)
        results[f"{name}_recall"] = correct / total
    if datasets["voc"] is not None:
        results["voc_miou"] = voc_mean_iou(model, datasets["voc"], args, device)
    return results


@torch.inference_mode()
def image_vectors(model, dataset, args, device):
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    vectors = []
    for images, _ in loader:
        _, embeddings = model(images.to(device))
        vectors.append(F.normalize(embeddings, dim=1).cpu())
    return torch.cat(vectors)


def recall_at_one(queries, query_labels, gallery, gallery_labels, same_set, chunk=128):
    query_labels = list(query_labels)
    gallery_labels = list(gallery_labels)
    gallery_counts = Counter(gallery_labels)
    correct = eligible = 0
    for start in range(0, len(queries), chunk):
        end = min(start + chunk, len(queries))
        similarity = queries[start:end] @ gallery.T
        if same_set:
            similarity[torch.arange(end - start), torch.arange(start, end)] = -float("inf")
        closest = similarity.argmax(dim=1).tolist()
        for local, index in enumerate(closest):
            label = query_labels[start + local]
            if gallery_counts[label] > int(same_set):
                eligible += 1
                correct += gallery_labels[index] == label
    if not eligible:
        raise ValueError("No query has another image of the same item in the gallery")
    return correct, eligible


def cross_view_metrics(queries, query_labels, query_cameras, gallery, gallery_labels, gallery_cameras):
    """Rank-1 and mAP, searching only cameras different from each query's."""
    if any(camera is None for camera in (*query_cameras, *gallery_cameras)):
        raise ValueError("Cross-view evaluation requires a camera column on every query/gallery row")
    gallery_labels = list(gallery_labels)
    gallery_cameras = list(gallery_cameras)
    correct = valid = 0
    ap_total = 0.0
    for start in range(0, len(queries), 128):
        scores = queries[start:start + 128] @ gallery.T
        for offset, row in enumerate(scores):
            identity = query_labels[start + offset]
            camera = query_cameras[start + offset]
            eligible = [i for i, other_camera in enumerate(gallery_cameras) if other_camera != camera]
            positives = {i for i in eligible if gallery_labels[i] == identity}
            if not positives:
                continue
            ranking = sorted(eligible, key=lambda i: float(row[i]), reverse=True)
            valid += 1
            correct += ranking[0] in positives
            hits = 0
            ap = 0.0
            for rank, index in enumerate(ranking, 1):
                if index in positives:
                    hits += 1
                    ap += hits / rank
            ap_total += ap / len(positives)
    if not valid:
        raise ValueError("No query identity appears in a different gallery camera")
    return correct, valid, ap_total / valid
