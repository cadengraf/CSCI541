"""Prepare local AICity cross-camera vehicle data using config.yaml."""

import yaml

from train import CONFIG, path_from_config
from utils.data_sources.aicity import prepare


def main():
    with CONFIG.open() as file:
        settings = yaml.safe_load(file)["aicity"]
    expected = {"benchmark_manifest", "output_dir", "max_per_camera"}
    if set(settings) != expected:
        raise ValueError(f"config.yaml aicity needs: {sorted(expected)}")
    prepare(path_from_config(settings["benchmark_manifest"]),
            path_from_config(settings["output_dir"]), settings["max_per_camera"])


if __name__ == "__main__":
    main()
