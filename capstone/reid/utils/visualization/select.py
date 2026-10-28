"""Click a VisDrone object, then replay the video with joint-model ReID matches."""

import yaml

from train import CONFIG, demo_checkpoint, path_from_config
from utils.visualization.target_search import run


def main():
    with CONFIG.open() as file:
        config = yaml.safe_load(file)
    settings = config["select"]
    expected = {"checkpoint", "source", "output", "frames", "stride", "fps", "min_box", "target_id"}
    if set(settings) != expected:
        raise ValueError(f"config.yaml select needs: {sorted(expected)}")
    settings["checkpoint"] = demo_checkpoint(settings["checkpoint"])
    settings["output"] = path_from_config(settings["output"])
    run(settings, path_from_config(config["visdrone"]["val_root"]))


if __name__ == "__main__":
    main()
