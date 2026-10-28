"""Prepare/evaluate VisDrone, then optionally train on it; settings are in config.yaml."""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import yaml

from utils.core.evaluation import evaluate
from utils.core.training import train
from utils.data_sources.prepare_aerial import prepare_visdrone
from train import CONFIG, path_from_config


SOURCE = "visdrone"
PREPARE_KEYS = {"train_root", "val_root", "manifest_dir", "frame_step", "min_box",
                "max_train_per_track", "min_eval_gap", "train_url", "val_url",
                "train_mirror_url", "val_mirror_url", "allow_third_party_mirror",
                "download_if_missing"}
TRAIN_KEYS = {"baseline_checkpoint", "output_checkpoint", "epochs", "batch_size",
              "learning_rate", "workers", "seed", "augmentation"}


def settings():
    with CONFIG.open() as file:
        value = yaml.safe_load(file).get("visdrone")
    if not isinstance(value, dict) or set(value) != PREPARE_KEYS | TRAIN_KEYS:
        raise ValueError(f"config.yaml visdrone needs: {sorted(PREPARE_KEYS | TRAIN_KEYS)}")
    for key in ("train_root", "val_root", "manifest_dir", "baseline_checkpoint", "output_checkpoint"):
        value[key] = path_from_config(value[key])
    return value


def source_paths(config):
    folder = config["manifest_dir"]
    paths = (folder / "train.csv", folder / "query.csv", folder / "gallery.csv")
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Missing manifests under {folder}; run `python prepare.py visdrone` first")
    return (SOURCE, *(str(path) for path in paths))


def extract_dataset(archive, root):
    """Extract the annotation and frame trees regardless of the ZIP's outer folder."""
    print(f"Extracting {archive} to {root} (this can take a while)", flush=True)
    extracted = 0
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            parts = PurePosixPath(member.filename).parts
            subtree = next((index for index, part in enumerate(parts)
                            if part in {"annotations", "sequences"}), None)
            if subtree is None or member.is_dir():
                continue
            relative = parts[subtree:]
            if any(part in {"", ".", ".."} for part in relative):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            destination = root.joinpath(*relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1
    if not (root / "annotations").is_dir() or not (root / "sequences").is_dir() or not extracted:
        raise ValueError(f"{archive} does not contain VisDrone-MOT annotations and sequences")
    (root / ".extracted-ok").write_text(f"{extracted} files extracted\n")
    print(f"Extracted {extracted} files", flush=True)


def ensure_dataset(config, split):
    root = config[f"{split}_root"]
    if (root / "annotations").is_dir() and (root / "sequences").is_dir():
        # Accept archives the user has already extracted manually.
        if (root / ".extracted-ok").exists() or not (root.parent / f"{root.name}.zip").exists():
            return
    archive = root.parent / f"{root.name}.zip"
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        if not config["download_if_missing"]:
            raise FileNotFoundError(f"Missing {root} and {archive}; set download_if_missing: true")
        try:
            import gdown
        except ImportError as error:
            raise RuntimeError("Install the new download dependency: pip install -r requirements.txt") from error
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading official VisDrone-MOT {split} archive to {archive}", flush=True)
        try:
            result = gdown.download(url=config[f"{split}_url"], output=str(archive),
                                    quiet=False, resume=True, retries=3)
            if result is None or not zipfile.is_zipfile(archive):
                raise RuntimeError("The official download did not produce a complete ZIP")
        except Exception as official_error:
            if not config["allow_third_party_mirror"]:
                raise RuntimeError(
                    f"Official download failed: {official_error}. You can download {archive.name} "
                    f"in a browser from {config[f'{split}_url']} and save it as {archive}") from official_error
            mirror_archive = archive.with_name(f"{archive.stem}.mirror.zip")
            mirror_url = config[f"{split}_mirror_url"]
            print(f"Official download unavailable ({official_error}). Trying third-party mirror: "
                  f"{mirror_url}", flush=True)
            try:
                result = gdown.download(url=mirror_url, output=str(mirror_archive),
                                        quiet=False, resume=True, retries=3)
                if result is None or not zipfile.is_zipfile(mirror_archive):
                    raise RuntimeError("Mirror download did not produce a complete ZIP")
                mirror_archive.replace(archive)
            except Exception as mirror_error:
                raise RuntimeError(
                    f"Official and mirror downloads failed. Official: {official_error}; "
                    f"mirror: {mirror_error}. Download {archive.name} in a browser from "
                    f"{mirror_url} and save it as {archive}, then rerun python prepare.py visdrone") from mirror_error
    extract_dataset(archive, root)


def eval_checkpoint(config, checkpoint):
    args = argparse.Namespace(sop_root=None, abo_root=None, fashion_root=None, voc_root=None,
                              abo_max_items=None, source=[source_paths(config)],
                              reid_only=True, checkpoint=checkpoint,
                              batch_size=config["batch_size"], workers=config["workers"],
                              global_pool=True)
    evaluate(args)


def main(mode="prepare"):
    if mode not in ("prepare", "train"):
        raise SystemExit("Usage: python prepare.py visdrone | python train.py visdrone")
    config = settings()
    if mode == "prepare":
        for split in ("train", "val"):
            ensure_dataset(config, split)
        prepare_visdrone(argparse.Namespace(
            train_root=config["train_root"], val_root=config["val_root"],
            output_dir=config["manifest_dir"], frame_step=config["frame_step"],
            min_box=config["min_box"], max_train_per_track=config["max_train_per_track"],
            min_eval_gap=config["min_eval_gap"]))
        print("Baseline aerial Recall@1 (global pooling):", flush=True)
        eval_checkpoint(config, config["baseline_checkpoint"])
        return
    source = source_paths(config)
    if config["output_checkpoint"].resolve() == config["baseline_checkpoint"].resolve():
        raise ValueError("output_checkpoint must differ from baseline_checkpoint")
    args = argparse.Namespace(
        sop_root=None, abo_root=None, fashion_root=None, voc_root=None,
        abo_max_items=None, source=[source], reid_only=True, download_voc=False,
        resume=config["baseline_checkpoint"], checkpoint=config["output_checkpoint"],
        epochs=config["epochs"], batch_size=config["batch_size"],
        learning_rate=config["learning_rate"], workers=config["workers"],
        seed=config["seed"], augmentation=config["augmentation"])
    train(args)
    print("After-training aerial Recall@1:", flush=True)
    eval_checkpoint(config, config["output_checkpoint"])


if __name__ == "__main__":
    main()
