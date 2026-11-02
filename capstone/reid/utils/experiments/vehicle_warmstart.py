"""Matched continuation from the finished VRU ReID-only vehicle checkpoint."""

import csv
from argparse import Namespace

import torch
import yaml

from train import CONFIG, path_from_config, read_config
from utils.core.checkpointing import recovery_path, training_signature
from utils.core.training import train


def _identities(manifest):
    with manifest.open(newline="") as file:
        return {row["id"] for row in csv.DictReader(file)}


def main():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings = config["vehicle_warmstart"]
    source = path_from_config(settings["start_checkpoint"])
    classifier_source = recovery_path(source)
    identity_train = path_from_config(settings["train_manifest"])
    mask_train = path_from_config(settings["mask_manifest"])
    for path in (source, classifier_source, identity_train, mask_train):
        if not path.is_file():
            raise FileNotFoundError(f"Vehicle warm-start input is missing: {path}")
    original = torch.load(source, map_location="cpu", weights_only=True)
    if (not original.get("training", {}).get("finished") or
            original.get("segmentation_trained") or
            original.get("mask_classes") != 2 or
            original.get("preprocessing") != "vehicle_full_resize"):
        raise ValueError("Start checkpoint must be the finished binary-head VRU ReID-only model")
    previous = torch.load(classifier_source, map_location="cpu", weights_only=True)
    if not previous.get("finished") or previous.get("epoch") != original["training"]["epoch"]:
        raise ValueError("Starting ReID checkpoint and classifier recovery do not match")
    if previous["classifier"]["weight"].shape[0] != len(_identities(identity_train)):
        raise ValueError("Starting identity classifier does not match the VRU train IDs")
    if settings["workers"] != 0:
        raise ValueError("Use workers: 0 so identity batches match across arms")
    if not 2 <= settings["mask_pool_start_epoch"] <= settings["epochs"]:
        raise ValueError("mask_pool_start_epoch must be between 2 and epochs")
    if _identities(identity_train) & (
        _identities(path_from_config(config["vru"]["query_manifest"])) |
        _identities(path_from_config(config["vru"]["gallery_manifest"]))
    ):
        raise ValueError("VRU test identities overlap warm-start training")

    base = read_config()
    for label, reid_only in (("reid_only", True), ("joint", False)):
        args = Namespace(**vars(base))
        args.sop_root = args.abo_root = args.fashion_root = args.voc_root = None
        args.source = [("vru", str(identity_train), None, None, False, None, None)]
        args.source_batch_weights = {"vru": 1}
        args.resume = source
        args.resume_classifier = classifier_source
        args.checkpoint = path_from_config(settings["output_dir"]) / f"{label}.pt"
        for key in ("epochs", "steps_per_epoch", "batch_size", "instances_per_id",
                    "learning_rate", "augmentation", "mask_every", "mask_loss_weight",
                    "workers", "seed", "triplet_margin", "triplet_weight"):
            setattr(args, key, settings[key])
        args.mask_manifest = mask_train
        args.mask_classes = 2
        args.vehicle_preprocess = True
        args.reid_only = reid_only
        args.mask_pool_start_epoch = None if reid_only else settings["mask_pool_start_epoch"]
        args.selection = None  # Compare exactly the same final epoch.
        args.evaluate_after_train = False
        args.download_voc = False
        if args.checkpoint.is_file():
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if saved.get("training", {}).get("signature") != training_signature(args):
                raise ValueError(f"Existing checkpoint has different settings: {args.checkpoint}")
        print(f"Warm-start {label}: {args.epochs * args.steps_per_epoch} additional VRU ReID "
              f"updates from {source}; output={args.checkpoint}", flush=True)
        train(args)
    print("Warm-start comparison finished. Run `python benchmark.py vru` and "
          "`python benchmark.py vrai`.", flush=True)


if __name__ == "__main__":
    main()
