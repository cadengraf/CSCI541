"""Generate model comparisons or evaluate aerial object masks."""

import argparse
import yaml

from train import CONFIG, path_from_config
from utils.benchmarking.benchmark import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("masks",), help="evaluate labeled aerial masks")
    if parser.parse_args().mode == "masks":
        from utils.benchmarking.mask_eval import main as masks
        return masks()
    settings = yaml.safe_load(CONFIG.read_text())["benchmark"]
    paths = ("output", "images_dir", "aicity_query", "aicity_gallery", "aicity_train",
             "bucktales_manifest", "voc_root")
    for key in paths:
        settings[key] = path_from_config(settings[key])
    settings["checkpoints"] = {name: path_from_config(path)
                                for name, path in settings["checkpoints"].items()}
    settings["matched_checkpoints"] = {name: path_from_config(path)
                                        for name, path in settings["matched_checkpoints"].items()}
    run(settings)


if __name__ == "__main__":
    main()
