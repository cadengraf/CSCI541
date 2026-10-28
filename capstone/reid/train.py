"""Train and optionally evaluate the ReID model using config.yaml."""

import argparse
from pathlib import Path

import yaml

from utils.core.evaluation import evaluate
from utils.core.training import train


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"
PATH_KEYS = ("sop_root", "abo_root", "voc_root", "fashion_root", "resume", "checkpoint")
KEYS = set(PATH_KEYS) | {
    "reid_only", "abo_max_items", "sources", "epochs", "batch_size",
    "learning_rate", "workers", "seed", "augmentation", "download_voc",
    "evaluate_after_train", "visdrone", "video", "select", "aicity", "crossdrone",
    "instances_per_id", "steps_per_epoch", "source_batch_weights",
    "triplet_margin", "triplet_weight", "mask_every", "mask_loss_weight",
    "selection", "benchmark", "comparison", "aerial_mask_eval", "vrai", "vru",
}


def path_from_config(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def demo_checkpoint(value):
    if value == "auto":
        candidates = (ROOT / "checkpoints" / "joint_balanced.pt",
                      ROOT / "checkpoints" / "joint_multiview.pt",
                      ROOT / "checkpoints" / "joint_all_current.pt")
        value = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
    path = path_from_config(value)
    if not path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {path}")
    print(f"Using model checkpoint: {path}", flush=True)
    return path


def read_config():
    with CONFIG.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"{CONFIG} must contain a YAML mapping")
    missing, unknown = KEYS - config.keys(), config.keys() - KEYS
    if missing or unknown:
        raise ValueError(f"Config keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
    # BuckTales is the untouched cross-drone test set. Its pair-local IDs do not
    # establish identity separation across the recordings, so using its crops
    # for training or checkpoint selection would invalidate the benchmark.
    bucktales_manifest = path_from_config(config["crossdrone"]["manifest"]).resolve()
    bucktales_folder = bucktales_manifest.parent
    bucktales_source = path_from_config(config["crossdrone"]["archive"]).resolve()
    for source in config["sources"]:
        if source["name"].casefold() == "bucktales":
            raise ValueError("BuckTales is reserved for test-only evaluation")
        for key in ("train", "query", "gallery", "calibration_query", "calibration_gallery"):
            value = source.get(key)
            if value is not None:
                path = path_from_config(value).resolve()
                if (path == bucktales_manifest or bucktales_folder in path.parents
                        or path == bucktales_source or bucktales_source in path.parents):
                    raise ValueError(f"{source['name']}.{key} points into the BuckTales test-only folder")
    del config["visdrone"]  # The aerial workflow reads this section itself.
    del config["video"]  # The MP4 demo reads this section itself.
    del config["select"]  # The target selection demo reads this section itself.
    del config["aicity"]  # The AICity preparation command reads this section itself.
    del config["crossdrone"]  # The cross-drone evaluation reads this section itself.
    del config["benchmark"]  # The comparison report reads this section itself.
    del config["comparison"]  # The matched experiment reads this section itself.
    del config["aerial_mask_eval"]  # The aerial mask check reads this section itself.
    del config["vrai"]  # The VRAI preparation command reads this section itself.
    del config["vru"]  # The VRU preparation command reads this section itself.
    for key in PATH_KEYS:
        config[key] = path_from_config(config[key])
    sources = []
    if not isinstance(config["sources"], list):
        raise ValueError("sources must be a list")
    for source in config["sources"]:
        expected = {"name", "train", "query", "gallery", "calibration_query",
                    "calibration_gallery", "cross_view", "optional"}
        if not isinstance(source, dict) or set(source) != expected:
            raise ValueError(f"Each source needs: {sorted(expected)}")
        paths = tuple(path_from_config(source[key]) for key in
                      ("train", "query", "gallery", "calibration_query", "calibration_gallery"))
        if (paths[3] is None) != (paths[4] is None):
            raise ValueError(f"{source['name']}: calibration query and gallery must both be set")
        required = [path for path in paths if path is not None]
        if not all(path.is_file() for path in required):
            if source["optional"] and not any(path.is_file() for path in required):
                print(f"Optional {source['name']} data not prepared; skipping it", flush=True)
                continue
            raise FileNotFoundError(f"Incomplete {source['name']} manifests: {paths}")
        if paths[3] is not None and not source["cross_view"]:
            raise ValueError(f"{source['name']}: calibration requires cross_view: true")
        sources.append((source["name"], *(str(path) if path else None for path in paths[:3]),
                        source["cross_view"], *(str(path) if path else None for path in paths[3:])))
    config["source"] = sources
    del config["sources"]
    if config["checkpoint"] is None:
        raise ValueError("checkpoint must be a path")
    if config["augmentation"] not in ("none", "basic", "moderate", "moderate_occlusion", "strong"):
        raise ValueError("augmentation must be none, basic, moderate, moderate_occlusion, or strong")
    if config["batch_size"] < 4 or config["batch_size"] % 2:
        raise ValueError("batch_size must be an even number of at least 4")
    if config["instances_per_id"] < 2 or config["batch_size"] % config["instances_per_id"]:
        raise ValueError("instances_per_id must be at least 2 and divide batch_size")
    if config["steps_per_epoch"] is not None and config["steps_per_epoch"] < 1:
        raise ValueError("steps_per_epoch must be positive or null")
    if config["triplet_margin"] <= 0 or config["triplet_weight"] < 0:
        raise ValueError("triplet_margin must be positive and triplet_weight nonnegative")
    if config["mask_every"] < 1 or config["mask_loss_weight"] < 0:
        raise ValueError("mask_every must be positive and mask_loss_weight nonnegative")
    weights = config["source_batch_weights"]
    if not isinstance(weights, dict) or any(not isinstance(value, (int, float)) or value <= 0 for value in weights.values()):
        raise ValueError("source_batch_weights must map names to positive numbers")
    selection = config["selection"]
    if (not isinstance(selection, dict) or set(selection) != {"weights", "minimum_ratio"}
            or not isinstance(selection["weights"], dict) or not selection["weights"]
            or not isinstance(selection["minimum_ratio"], dict)):
        raise ValueError("selection needs nonempty weights and minimum_ratio mappings")
    if any(not isinstance(value, (int, float)) or value <= 0
           for value in selection["weights"].values()):
        raise ValueError("selection weights must be positive")
    if any(not isinstance(value, (int, float)) or not 0 <= value <= 1
           for value in selection["minimum_ratio"].values()):
        raise ValueError("selection minimum ratios must be between 0 and 1")
    if set(selection["minimum_ratio"]) - set(selection["weights"]):
        raise ValueError("selection minimum_ratio keys must have weights")
    if config["resume"] is None or not config["resume"].is_file():
        raise FileNotFoundError("A starting checkpoint is required for relative model selection")
    if config["resume"] == config["checkpoint"]:
        raise ValueError("resume and checkpoint must be different files")
    if not config["reid_only"] and config["voc_root"] is None:
        raise ValueError("voc_root is required unless reid_only is true")
    if config["reid_only"] and config["download_voc"]:
        raise ValueError("download_voc cannot be true with reid_only")
    return argparse.Namespace(**config)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the joint segmentation + ReID model using config.yaml."
    )
    parser.add_argument("mode", nargs="?", choices=("compare", "visdrone"),
                        help="optional matched comparison or separate VisDrone training")
    options = parser.parse_args()
    if options.mode == "compare":
        from utils.experiments.compare import main as compare
        return compare()
    if options.mode == "visdrone":
        from utils.data_sources.visdrone import main as visdrone
        return visdrone("train")
    args = read_config()
    print(f"Using {CONFIG}", flush=True)
    train(args)
    if args.evaluate_after_train:
        evaluate(args)


if __name__ == "__main__":
    main()
