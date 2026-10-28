"""Evaluate two-drone ReID and save a selected-object video from config.yaml."""

from types import SimpleNamespace
import sys

import yaml

from train import CONFIG, demo_checkpoint, path_from_config
from utils.crossdrone import audit_split, evaluate, inspect_archive, prepare_dataset, read_observations, render


def main(mode="video"):
    with CONFIG.open() as file:
        settings = yaml.safe_load(file)["crossdrone"]
    expected = {"archive", "manifest", "checkpoint", "output", "pair", "source_camera", "target_id",
                "max_gallery_per_identity", "video_frames", "video_stride", "fps", "batch_size", "workers",
                "samples_per_identity"}
    if set(settings) != expected:
        raise ValueError(f"config.yaml crossdrone needs: {sorted(expected)}")
    if mode not in ("video", "inspect", "prepare", "audit"):
        raise SystemExit("Usage: python demo.py crossdrone | python prepare.py bucktales [inspect|prepare|audit]")
    if mode == "inspect":
        try:
            inspect_archive(path_from_config(settings["archive"]))
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from None
        return
    if mode == "prepare":
        if not isinstance(settings["samples_per_identity"], int) or settings["samples_per_identity"] < 1:
            raise ValueError("crossdrone.samples_per_identity must be a positive integer")
        prepare_dataset(path_from_config(settings["archive"]),
                        path_from_config(settings["manifest"]),
                        settings["pair"], settings["samples_per_identity"])
        return
    if mode == "audit":
        audit_split(path_from_config(settings["archive"]),
                    read_observations(path_from_config(settings["manifest"])))
        return
    for key in ("max_gallery_per_identity", "video_frames", "video_stride", "fps", "batch_size"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"crossdrone.{key} must be a positive integer")
    if not isinstance(settings["workers"], int) or settings["workers"] < 0:
        raise ValueError("crossdrone.workers must be a nonnegative integer")
    settings["manifest"] = path_from_config(settings["manifest"])
    settings["checkpoint"] = demo_checkpoint(settings["checkpoint"])
    settings["output"] = path_from_config(settings["output"])
    settings["loader"] = SimpleNamespace(batch_size=settings["batch_size"], workers=settings["workers"])
    try:
        rows = read_observations(settings["manifest"])
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    evaluate(settings, rows)
    render(settings, rows)


if __name__ == "__main__":
    main()
