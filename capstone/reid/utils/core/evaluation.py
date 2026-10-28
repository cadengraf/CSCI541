"""Held-out ReID retrieval and VOC segmentation evaluation."""

from pathlib import Path

import torch

from .datasets import (AmazonBerkeleyObjects, ForegroundVOC, IdentityPartition,
                       InShop, ManifestItems, index_partition)
from .metrics import cross_view_metrics, image_vectors, recall_at_one, voc_mean_iou
from .model import JointNet
from .identity_data import Products


@torch.inference_mode()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    use_mask_pool = (False if getattr(args, "global_pool", False)
                     else checkpoint.get("use_mask_pool", True))
    model = JointNet(pretrained=False, use_mask_pool=use_mask_pool)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    holdout = getattr(args, "selection", None) is not None
    if args.sop_root:
        sop = Products(args.sop_root, "test", training=False)
        if holdout:
            sop = IdentityPartition(sop, "sop", "holdout", args.seed)
        vectors = image_vectors(model, sop, args, device)
        correct, total = recall_at_one(vectors, sop.labels, vectors, sop.labels, same_set=True)
        print(f"SOP {'holdout ' if holdout else ''}Recall@1: {correct / total:.2%} ({correct}/{total})")
    if args.fashion_root:
        queries = InShop(args.fashion_root, "query")
        gallery = InShop(args.fashion_root, "gallery")
        if holdout:
            queries = IdentityPartition(queries, "fashion", "holdout", args.seed)
            gallery = IdentityPartition(gallery, "fashion", "holdout", args.seed)
        qv = image_vectors(model, queries, args, device)
        gv = image_vectors(model, gallery, args, device)
        correct, total = recall_at_one(qv, queries.labels, gv, gallery.labels, same_set=False)
        print(f"DeepFashion {'holdout ' if holdout else ''}query-to-gallery Recall@1: "
              f"{correct / total:.2%} ({correct}/{total})")
    if args.abo_root:
        abo = AmazonBerkeleyObjects(args.abo_root, "test", args.abo_max_items)
        if holdout:
            abo = IdentityPartition(abo, "abo", "holdout", args.seed)
        av = image_vectors(model, abo, args, device)
        correct, total = recall_at_one(av, abo.labels, av, abo.labels, same_set=True)
        print(f"ABO {'holdout ' if holdout else ''}Recall@1: {correct / total:.2%} ({correct}/{total})")
    for source in args.source:
        name, train_manifest, query_manifest, gallery_manifest = source[:4]
        if query_manifest is None or gallery_manifest is None:
            continue
        cross_view = len(source) > 4 and source[4]
        queries = ManifestItems(Path(query_manifest))
        gallery = ManifestItems(Path(gallery_manifest))
        if holdout and not cross_view:
            queries = IdentityPartition(queries, name, "holdout", args.seed)
            gallery = IdentityPartition(gallery, name, "holdout", args.seed)
        if not cross_view and {(path, box) for path, _, box in queries.samples} & {(path, box) for path, _, box in gallery.samples}:
            raise ValueError(f"{name} query/gallery contain an identical image crop")
        if not set(queries.labels) & set(gallery.labels):
            raise ValueError(f"{name} query/gallery have no shared identities")
        qv = image_vectors(model, queries, args, device)
        gv = image_vectors(model, gallery, args, device)
        if cross_view:
            training_ids = set(ManifestItems(Path(train_manifest)).labels)
            overlap = training_ids & (set(queries.labels) | set(gallery.labels))
            if overlap:
                raise ValueError(f"{name}: {len(overlap)} identities overlap training and evaluation")
            correct, total, mean_ap = cross_view_metrics(
                qv, queries.labels, queries.cameras, gv, gallery.labels, gallery.cameras)
            print(f"{name} cross-camera Rank-1: {correct / total:.2%} ({correct}/{total}); mAP: {mean_ap:.2%}")
        else:
            correct, total = recall_at_one(qv, queries.labels, gv, gallery.labels, same_set=False)
            print(f"{name} {'holdout ' if holdout else ''}query-to-gallery Recall@1: "
                  f"{correct / total:.2%} ({correct}/{total})")
    if args.reid_only:
        return
    masks = ForegroundVOC(args.voc_root, "val", download=False)
    if holdout:
        masks = index_partition(masks, "voc", "holdout", args.seed)
    print(f"VOC {'holdout ' if holdout else ''}mean IoU at 28x28: "
          f"{voc_mean_iou(model, masks, args, device):.2%}")
