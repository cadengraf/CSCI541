"""Generate model comparisons or evaluate aerial object masks."""

import argparse
import sys
import yaml

from train import CONFIG, path_from_config


def main():
    if len(sys.argv)>1 and sys.argv[1]=='milestones':
        from utils.reporting.milestones import run as milestones
        return milestones()
    if len(sys.argv)>1 and sys.argv[1]=='cityflow':
        from utils.experiments.cityflow.command import main as cityflow
        return cityflow(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='vehicle-recovery':
        from utils.experiments.vehicle_recovery.command import main as recovery
        return recovery(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='vehicle-reference':
        from utils.experiments.vehicle_reference.command import main as reference
        return reference(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='aerial-transfer':
        from utils.experiments.aerial_transfer.command import main as transfer
        return transfer(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='aerial':
        from utils.experiments.aerial_benchmark.command import main as aerial
        return aerial(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='vehicle-vru':
        from utils.vehicle_reid.vru_benchmark import main as vehicle_vru
        return vehicle_vru(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='vehicle-encoder':
        from utils.vehicle_reid.encoder_benchmark import main as encoder_benchmark
        encoder_benchmark(sys.argv[2:])
        from utils.vehicle_reid.encoder_report import finalize
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument('--config', default='configs/vehicle_encoder_benchmark.yaml')
        return finalize(config_parser.parse_args(sys.argv[2:]).config)
    if len(sys.argv)>1 and sys.argv[1]=='vehicle-evidence':
        from utils.vehicle_reid.evidence_experiment import main as evidence
        evidence(sys.argv[2:])
        from utils.vehicle_reid.evidence_scoring import run as score_evidence
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument('--config', default='configs/tracklet_evidence.yaml')
        return score_evidence(config_parser.parse_args(sys.argv[2:]).config)
    if len(sys.argv)>1 and sys.argv[1]=='vehicle-budget':
        from utils.vehicle_reid.budget import main as recovery_budget
        return recovery_budget(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] in ("vehicle", "vehicle-retrieval"):
        if sys.argv[1] == "vehicle-retrieval" or any(flag in sys.argv[2:] for flag in ("--stage", "--limit-identities")):
            from utils.vehicle_reid.retrieval import main as vehicle
        else:
            from utils.vehicle_reid.benchmark import main as vehicle
        return vehicle(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("masks", "vru", "vru-masks", "vrai", "metrics", "kitti-mots", "kitti-mots-check", "kitti-mots-score", "teacher", "trackers", "trackers-check", "trackers-setup", "reid-teacher", "reid-teacher-final", "reid-teacher-setup", "joint-distill", "joint-distill-control", "reid-diagnostics", "compare-reid", "compare-roi", "crop-diagnostics", "crop-validation", "compare-crops", "vrai-adapt"),
                        help="evaluate masks, VRU/VRAI retrieval, or refresh metrics HTML")
    mode = parser.parse_args().mode
    if mode == "vrai-adapt":
        from utils.experiments.vrai_adaptation.command import run as adapt
        return adapt(evaluate_only=True)
    if mode == "compare-crops":
        from utils.experiments.crop_training.command import run as compare_crops
        return compare_crops(evaluate_only=True)
    if mode in ("crop-diagnostics", "crop-validation"):
        from utils.teacher.launcher import launch
        return launch(mode)
    if mode == "compare-roi":
        from utils.experiments.roi_reid.command import run as compare_roi
        return compare_roi(evaluate_only=True)
    if mode == "compare-reid":
        from utils.experiments.reid_balance.command import run as compare_reid
        return compare_reid(evaluate_only=True)
    if mode == "reid-diagnostics":
        from utils.teacher.launcher import launch
        return launch("reid-diagnostics")
    if mode in ("joint-distill", "joint-distill-control"):
        from utils.distillation.command import run as distill
        return distill(mode + '-eval')
    if mode == "reid-teacher-setup":
        from utils.reid_teacher.setup import run as setup
        return setup()
    if mode in ("reid-teacher", "reid-teacher-final"):
        from utils.reid_teacher.evaluation import run as specialist
        return specialist(final=mode.endswith('-final'))
    if mode == "trackers-setup":
        from utils.tracking.setup import run as setup
        return setup()
    if mode in ("trackers", "trackers-check"):
        from utils.teacher.launcher import launch
        from utils.tracking.comparison import run as compare
        check = mode.endswith('-check')
        launch('tracking-cache-check' if check else 'tracking-cache')
        return compare(check=check)
    if mode == "teacher":
        from utils.teacher.launcher import launch
        return launch("eval")
    if mode in ("kitti-mots", "kitti-mots-score", "kitti-mots-check"):
        from utils.full_frame.mots_benchmark import check, run as run_mots
        return check() if mode == "kitti-mots-check" else run_mots(score_only=mode.endswith('-score'))
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
    from utils.benchmarking.benchmark import run
    run(settings)


if __name__ == "__main__":
    main()
