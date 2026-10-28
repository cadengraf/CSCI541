"""Run a matched ReID-only versus joint continuation from config.yaml."""

from argparse import Namespace

import yaml

from train import CONFIG, path_from_config, read_config
from utils.core.training import train


def main():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["comparison"]
    start = path_from_config(config["start_checkpoint"])
    if not start.is_file():
        raise FileNotFoundError(f"Missing common starting checkpoint: {start}")
    if config["epochs"] < 1 or config["steps_per_epoch"] < 1:
        raise ValueError("comparison epochs and steps_per_epoch must be positive")
    base = read_config()
    for name, reid_only in (("reid_only", True), ("joint", False)):
        args = Namespace(**vars(base))
        args.resume = start
        args.checkpoint = path_from_config(config["output_dir"]) / f"{name}.pt"
        args.epochs = config["epochs"]
        args.steps_per_epoch = config["steps_per_epoch"]
        args.reid_only = reid_only
        args.selection = None  # Compare the same final epoch, without choosing on test data.
        args.evaluate_after_train = False
        args.download_voc = False
        if args.checkpoint == start:
            raise ValueError("Comparison output must differ from the starting checkpoint")
        print(f"Matched arm: {name}; start={start}; ReID updates="
              f"{args.epochs * args.steps_per_epoch}; output={args.checkpoint}", flush=True)
        train(args)
    print("Both arms complete. Run `python benchmark.py` for matched test plots.", flush=True)


if __name__ == "__main__":
    main()
