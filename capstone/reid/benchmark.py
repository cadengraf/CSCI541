"""Generate model comparisons or evaluate aerial object masks."""

import argparse
import yaml

from train import CONFIG, path_from_config
from utils.benchmarking.benchmark import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("masks", "vru", "vru-masks", "vrai", "metrics"),
                        help="evaluate masks, VRU/VRAI retrieval, or refresh metrics HTML")
    mode = parser.parse_args().mode
    if mode == "metrics":
        from utils.benchmarking.scorecard import render
        return render()
    if mode == "masks":
        from utils.benchmarking.mask_eval import main as masks
        return masks()
    if mode == "vru-masks":
        from utils.benchmarking.vehicle_mask_video import main as video
        return video()
    if mode == "vru":
        from utils.benchmarking.vru import run as vru_run
        settings = yaml.safe_load(CONFIG.read_text())["vru"]
        for key in ("train_manifest", "query_manifest", "gallery_manifest", "output"):
            settings[key] = path_from_config(settings[key])
        settings["checkpoints"] = {name: path_from_config(path)
                                   for name, path in settings["checkpoints"].items()}
        settings["matched_checkpoints"] = {name: path_from_config(path)
                                           for name, path in settings.get("matched_checkpoints", {}).items()}
        settings["warmstart_checkpoints"] = {name: path_from_config(path)
                                             for name, path in settings.get("warmstart_checkpoints", {}).items()}
        settings["external_features"] = {name: path_from_config(path)
                                          for name, path in settings.get("external_features", {}).items()}
        if settings.get("msinet"):
            for key in ("repo", "checkpoint"):
                settings["msinet"][key] = path_from_config(settings["msinet"][key])
        vru_run(settings)
        from utils.benchmarking.scorecard import render
        return render()
    if mode == "vrai":
        from utils.benchmarking.vrai_vehicle import run as vrai_run
        settings = yaml.safe_load(CONFIG.read_text())["vrai"]["vehicle_benchmark"]
        for key in ("train_manifest", "query_manifest", "gallery_manifest", "output",
                    "reid_only_checkpoint", "joint_checkpoint",
                    "warmstart_reid_only_checkpoint", "warmstart_joint_checkpoint"):
            settings[key] = path_from_config(settings[key])
        vrai_run(settings)
        from utils.benchmarking.scorecard import render
        return render()
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
