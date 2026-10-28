"""Joint identity + VOC training loop and checkpoint selection."""

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .checkpointing import (recovery_path, save_joint_checkpoint, save_recovery,
                            training_signature)
from .datasets import (AmazonBerkeleyObjects, ForegroundVOC, IdentityPartition,
                       InShop, ManifestItems, MixedItems, index_partition)
from .identity_data import PKSampler, Products
from .metrics import selection_metrics, selection_score
from .model import EMBEDDING_SIZE, JointNet, mask_loss


def train(args):
    """Train identity embeddings and VOC masks, then select by calibration data."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_names = []
    sources = ([Products(args.sop_root, "train", training=True, augmentation=args.augmentation)]
               if args.sop_root else [])
    if args.sop_root:
        source_names.append("sop")
    if args.abo_root:
        sources.append(AmazonBerkeleyObjects(args.abo_root, "train", args.abo_max_items, args.augmentation))
        source_names.append("abo")
    if args.fashion_root:
        sources.append(InShop(args.fashion_root, "train", args.augmentation))
        source_names.append("fashion")
    for source in args.source:
        name, train_manifest = source[:2]
        sources.append(ManifestItems(Path(train_manifest), training=True, augmentation=args.augmentation))
        source_names.append(name)
    if not sources:
        raise ValueError("Provide --sop-root, --abo-root, --fashion-root, or --source")
    items = MixedItems(sources)
    calibration = []
    for source in args.source:
        name, train_manifest, test_query, test_gallery = source[:4]
        if len(source) > 4 and source[4] and test_query and test_gallery:
            training_ids = set(ManifestItems(Path(train_manifest)).labels)
            test_queries = ManifestItems(Path(test_query))
            test_gallery = ManifestItems(Path(test_gallery))
            test_ids = set(test_queries.labels) | set(test_gallery.labels)
            if training_ids & test_ids:
                raise ValueError(f"{name}: final test identities overlap training")
            if not set(test_queries.labels) & set(test_gallery.labels):
                raise ValueError(f"{name}: final query/gallery have no shared identities")
            if len(source) >= 7 and source[5] and source[6]:
                calibration_ids = (set(ManifestItems(Path(source[5])).labels)
                                   | set(ManifestItems(Path(source[6])).labels))
                if calibration_ids & test_ids:
                    raise ValueError(f"{name}: final test identities overlap calibration")
        if len(source) < 7 or source[5] is None:
            continue
        name, train_manifest, _, _, _, query_manifest, gallery_manifest = source
        queries = ManifestItems(Path(query_manifest))
        gallery = ManifestItems(Path(gallery_manifest))
        training_ids = set(ManifestItems(Path(train_manifest)).labels)
        validation_ids = set(queries.labels) | set(gallery.labels)
        if training_ids & validation_ids:
            raise ValueError(f"{name}: calibration identities overlap training")
        if not set(queries.labels) & set(gallery.labels):
            raise ValueError(f"{name}: calibration query/gallery have no shared identities")
        calibration.append((name, queries, gallery))
    selection_settings = getattr(args, "selection", None)
    selection_data = {"cross_view": calibration, "retrieval": [], "voc": None}
    if selection_settings:
        if args.sop_root:
            selection_data["retrieval"].append(
                ("sop", IdentityPartition(Products(args.sop_root, "test", training=False),
                                          "sop", "selection", args.seed), None))
        if args.abo_root:
            selection_data["retrieval"].append(
                ("abo", IdentityPartition(AmazonBerkeleyObjects(args.abo_root, "test", args.abo_max_items),
                                          "abo", "selection", args.seed), None))
        if args.fashion_root:
            selection_data["retrieval"].append(
                ("fashion", IdentityPartition(InShop(args.fashion_root, "query"),
                                              "fashion", "selection", args.seed),
                 IdentityPartition(InShop(args.fashion_root, "gallery"),
                                   "fashion", "selection", args.seed)))
        for source in args.source:
            name, _, query_manifest, gallery_manifest = source[:4]
            if query_manifest is None or gallery_manifest is None:
                continue  # Identity-labeled training source without a public test protocol.
            if len(source) > 4 and source[4]:
                continue  # Its disjoint calibration manifest was loaded above.
            selection_data["retrieval"].append(
                (name, IdentityPartition(ManifestItems(Path(query_manifest)), name,
                                         "selection", args.seed),
                 IdentityPartition(ManifestItems(Path(gallery_manifest)), name,
                                   "selection", args.seed)))
        if not args.reid_only:
            selection_data["voc"] = index_partition(
                ForegroundVOC(args.voc_root, "val", download=False),
                "voc", "selection", args.seed)
    instances_per_id = getattr(args, "instances_per_id", 4)
    triplet_margin = getattr(args, "triplet_margin", 0.2)
    triplet_weight = getattr(args, "triplet_weight", 1.0)
    mask_every = getattr(args, "mask_every", 1)
    mask_loss_weight = getattr(args, "mask_loss_weight", 1.0)
    item_loader = DataLoader(items, batch_sampler=PKSampler(
        items.labels, items.cameras, args.batch_size, instances_per_id,
        getattr(args, "steps_per_epoch", None), items.sample_sources, source_names,
        getattr(args, "source_batch_weights", None)), num_workers=args.workers)
    masks = None if args.reid_only else ForegroundVOC(args.voc_root, "train", args.download_voc, args.augmentation)
    mask_loader = (None if masks is None else DataLoader(
        masks, batch_size=args.batch_size, shuffle=True, num_workers=args.workers))
    model = JointNet(pretrained=args.resume is None, use_mask_pool=not args.reid_only)
    model.compute_mask = not args.reid_only
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        mismatch = model.load_state_dict(saved["model"], strict=False)
        if mismatch.unexpected_keys or any(not key.startswith("mask_context.") for key in mismatch.missing_keys):
            raise ValueError(f"Incompatible resume checkpoint: {mismatch}")
    model.to(device)
    if args.reid_only:
        for parameter in (*model.mask_head.parameters(), *model.mask_context.parameters()):
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    classifier = nn.Linear(EMBEDDING_SIZE, len(set(items.labels))).to(device)
    identity_optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate * 10)
    model_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1)
    identity_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        identity_optimizer, T_max=args.epochs, eta_min=args.learning_rate)
    print(f"ReID: {len(items)} images / {len(set(items.labels))} identities from {len(sources)} dataset(s); "
          f"segmentation: {len(masks) if masks is not None else 'disabled'}; device={device}")
    baseline = None
    best_score = -float("inf")
    history = []
    start_epoch = 0
    signature = training_signature(args)
    last_path = recovery_path(args.checkpoint)
    best_saved = (torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                  if args.checkpoint.is_file() else None)
    if best_saved and best_saved.get("training", {}).get("finished"):
        print(f"Training already finished: {args.checkpoint}", flush=True)
        return
    if last_path.is_file():
        # CUDA RNG state must remain a CPU ByteTensor for set_rng_state_all.
        # Optimizer.load_state_dict moves parameter states to their device.
        recovery = torch.load(last_path, map_location="cpu", weights_only=True)
        if recovery.get("signature") != signature:
            raise ValueError(f"Recovery settings changed; inspect or move {last_path} before restarting")
        if best_saved is None:
            raise FileNotFoundError(f"Recovery exists but best checkpoint is missing: {args.checkpoint}")
        model.load_state_dict(recovery["model"])
        classifier.load_state_dict(recovery["classifier"])
        optimizer.load_state_dict(recovery["optimizer"])
        identity_optimizer.load_state_dict(recovery["identity_optimizer"])
        model_schedule.load_state_dict(recovery["model_schedule"])
        identity_schedule.load_state_dict(recovery["identity_schedule"])
        start_epoch = recovery["epoch"]
        baseline, best_score, history = recovery["baseline"], recovery["best_score"], recovery["history"]
        torch.set_rng_state(recovery["torch_rng"].cpu())
        random.setstate(recovery["python_rng"])
        if device.type == "cuda" and recovery["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in recovery["cuda_rng"]])
        print(f"Resumed full training state after epoch {start_epoch} from {last_path}", flush=True)
    else:
        if selection_settings:
            baseline = selection_metrics(model, selection_data, args, device)
            skipped = set(selection_settings["weights"]) - set(baseline)
            if skipped:
                print(f"Selection metrics unavailable and skipped: {sorted(skipped)}", flush=True)
            print(f"Starting checkpoint validation: {baseline}", flush=True)
            best_score, _ = selection_score(baseline, baseline, selection_settings)
            history.append({"epoch": 0, "metrics": baseline, "score": best_score, "below_floor": []})
        if best_saved and best_saved.get("training", {}).get("epoch", 0) > 0:
            saved_training = best_saved["training"]
            if saved_training.get("signature") not in (None, signature):
                raise ValueError("Existing best checkpoint was made with different training settings")
            start_epoch = saved_training["epoch"]
            model.load_state_dict(best_saved["model"])
            best_score = saved_training.get("selection_score", best_score)
            if selection_settings:
                history.append({"epoch": start_epoch,
                                "metrics": saved_training.get("selection_metrics"),
                                "score": best_score, "below_floor": []})
            print(f"Model-only recovery from epoch {start_epoch}: classifier, optimizer, "
                  "and learning-rate schedule restart because the old checkpoint lacks them", flush=True)
        elif selection_settings:
            save_joint_checkpoint(model, args, epoch=0, selection_metrics=baseline,
                                  selection_score=best_score, instances_per_id=instances_per_id)
    if start_epoch >= args.epochs:
        raise ValueError(f"Saved epoch {start_epoch} has reached configured {args.epochs} epochs")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        mask_iterator = iter(mask_loader) if mask_loader is not None else None
        reid_total = mask_total = 0.0
        mask_batches = 0
        for step, (images, labels) in enumerate(item_loader, 1):
            optimizer.zero_grad()
            identity_optimizer.zero_grad()
            _, features = model(images.to(device))
            vectors = F.normalize(features, dim=1)
            labels = labels.to(device)
            distance = torch.cdist(vectors, vectors, p=2)
            same = labels[:, None] == labels[None, :]
            same.fill_diagonal_(False)
            hardest_positive = distance.masked_fill(~same, -1).max(dim=1).values
            hardest_negative = distance.masked_fill(labels[:, None] == labels[None, :], float("inf")).min(dim=1).values
            triplet_loss = F.relu(hardest_positive - hardest_negative + triplet_margin).mean()
            id_loss = F.cross_entropy(classifier(features), labels, label_smoothing=0.1)
            reid_loss = id_loss + triplet_weight * triplet_loss
            combined_loss = reid_loss
            reid_total += reid_loss.item()

            if mask_loader is not None and step % mask_every == 0:
                try:
                    mask_images, target_masks = next(mask_iterator)
                except StopIteration:
                    mask_iterator = iter(mask_loader)
                    mask_images, target_masks = next(mask_iterator)
                logits, _ = model(mask_images.to(device))
                segmentation_loss = mask_loss(logits, target_masks.to(device))
                combined_loss = combined_loss + mask_loss_weight * segmentation_loss
                mask_total += segmentation_loss.item()
                mask_batches += 1
            combined_loss.backward()
            optimizer.step()
            identity_optimizer.step()
            if step % 50 == 0:
                print(f"Epoch {epoch}: {step}/{len(item_loader)} batches", flush=True)
        summary = f"Epoch {epoch}: ID + triplet loss {reid_total / len(item_loader):.4f}"
        if mask_batches:
            summary += f"; mask loss {mask_total / mask_batches:.4f} ({mask_batches} batches)"
        print(summary, flush=True)
        model_schedule.step()
        identity_schedule.step()
        if selection_settings:
            metrics = selection_metrics(model, selection_data, args, device)
            score, failed = selection_score(metrics, baseline, selection_settings)
            history.append({"epoch": epoch, "metrics": metrics, "score": score,
                            "below_floor": failed})
            print(f"Epoch {epoch} validation: {metrics}; score={score}; "
                  f"below preservation floor={failed}", flush=True)
            if score is not None and score > best_score:
                best_score = score
                save_joint_checkpoint(model, args, epoch, metrics, score, instances_per_id)
                print(f"Saved best balanced checkpoint: {args.checkpoint} (epoch {epoch})", flush=True)
        else:
            save_joint_checkpoint(model, args, epoch, None, None, instances_per_id)
            history.append({"epoch": epoch, "metrics": None, "score": None,
                            "below_floor": []})
        save_recovery(last_path, model, classifier, optimizer, identity_optimizer,
                      model_schedule, identity_schedule, epoch, baseline, best_score,
                      history, signature)
        print(f"Saved recovery checkpoint after epoch {epoch}: {last_path}", flush=True)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved["training"]["completed_epochs"] = args.epochs
    saved["training"]["finished"] = True
    saved["training"]["history"] = history
    temporary = args.checkpoint.with_name(args.checkpoint.name + ".partial")
    torch.save(saved, temporary)
    temporary.replace(args.checkpoint)
    recovery = torch.load(last_path, map_location="cpu", weights_only=True)
    recovery["finished"] = True
    temporary = last_path.with_name(last_path.name + ".partial")
    torch.save(recovery, temporary)
    temporary.replace(last_path)
    print(f"Saved {args.checkpoint}")
