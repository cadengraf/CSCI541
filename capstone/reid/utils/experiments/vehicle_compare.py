"""Matched vehicle-only ReID and binary-mask training on official VRU train IDs."""

import csv
from argparse import Namespace

import torch
import yaml

from train import CONFIG, path_from_config, read_config
from utils.core.checkpointing import training_signature
from utils.core.training import train


def main():
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["vehicle_comparison"]
    identity_train = path_from_config(settings["train_manifest"])
    mask_train = path_from_config(settings["mask_manifest"])
    if not identity_train.is_file() or not mask_train.is_file():
        raise FileNotFoundError("Run `python prepare.py vru` and `python prepare.py vehicle-masks` first")
    if settings["mask_classes"] != 2:
        raise ValueError("Vehicle comparison requires a binary mask head (mask_classes: 2)")
    if settings["epochs"] < 1 or settings["steps_per_epoch"] < 1:
        raise ValueError("Vehicle comparison needs positive epochs and steps_per_epoch")
    if settings["workers"] != 0:
        raise ValueError("Use workers: 0 to keep identity batches identical across arms")
    with identity_train.open(newline="") as file:
        identities = {row["id"] for row in csv.DictReader(file)}
    for key in ("query_manifest", "gallery_manifest"):
        with path_from_config(yaml.safe_load(CONFIG.read_text())["vru"][key]).open(newline="") as file:
            if identities & {row["id"] for row in csv.DictReader(file)}:
                raise ValueError(f"VRU {key} identities overlap training")
    base = read_config()
    for label, reid_only in (("reid_only", True), ("joint", False)):
        args = Namespace(**vars(base))
        args.sop_root = args.abo_root = args.fashion_root = args.voc_root = None
        args.source = [("vru", str(identity_train), None, None, False, None, None)]
        args.source_batch_weights = {"vru": 1}
        args.resume = None  # Same cached ImageNet ShuffleNetV2 initialization in each arm.
        args.checkpoint = path_from_config(settings["output_dir"]) / f"{label}.pt"
        for key in ("epochs", "steps_per_epoch", "batch_size", "instances_per_id",
                    "learning_rate", "augmentation", "mask_every", "mask_loss_weight",
                    "workers", "seed", "triplet_margin", "triplet_weight"):
            setattr(args, key, settings[key])
        args.mask_manifest = mask_train
        args.mask_classes = 2
        args.vehicle_preprocess = True
        args.reid_only = reid_only
        args.selection = None  # Always evaluate the same final epoch.
        args.evaluate_after_train = False
        args.download_voc = False
        if args.checkpoint.is_file():
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if saved.get("training", {}).get("signature") != training_signature(args):
                raise ValueError(f"Existing checkpoint has different comparison settings: {args.checkpoint}")
        print(f"Vehicle comparison: {label}; {args.epochs * args.steps_per_epoch} VRU ReID "
              f"updates; binary iSAID mask batches every {args.mask_every} steps "
              f"for joint; output={args.checkpoint}", flush=True)
        train(args)
    print("Both vehicle arms finished. Run `python benchmark.py vru`.", flush=True)


if __name__ == "__main__":
    main()
