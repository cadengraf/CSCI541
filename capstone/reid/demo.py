"""Render model videos using the settings in config.yaml."""

import argparse
import yaml

from train import CONFIG, demo_checkpoint, path_from_config
from utils.visualization.video import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("select", "crossdrone"),
                        help="select a VisDrone object or render BuckTales cross-drone matching")
    mode = parser.parse_args().mode
    if mode == "select":
        from utils.visualization.select import main as select
        return select()
    if mode == "crossdrone":
        from utils.crossdrone.command import main as crossdrone
        return crossdrone()
    with CONFIG.open() as file:
        config = yaml.safe_load(file)
    settings = config["video"]
    expected = {"checkpoint", "source", "output", "frames", "stride", "fps", "max_objects", "min_box"}
    if set(settings) != expected:
        raise ValueError(f"config.yaml video needs: {sorted(expected)}")
    settings["checkpoint"] = demo_checkpoint(settings["checkpoint"])
    settings["output"] = path_from_config(settings["output"])
    val_root = path_from_config(config["visdrone"]["val_root"])
    render(settings, val_root)


if __name__ == "__main__":
    main()
