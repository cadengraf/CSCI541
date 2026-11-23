"""Compatibility dispatch for completed and historical training experiments."""
import argparse
import sys
from train import CONFIG, read_config
from utils.core.training import train
from utils.core.evaluation import evaluate

def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='aerial':
        from utils.experiments.aerial_training.command import main as aerial
        return aerial(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='appearance':
        from utils.experiments.appearance.command import main as appearance
        return appearance(sys.argv[2:])
    parser = argparse.ArgumentParser(
        description="Train and evaluate the joint segmentation + ReID model using config.yaml."
    )
    parser.add_argument("mode", nargs="?", choices=("compare", "compare-vehicles", "warmstart-vehicles", "visdrone", "full-frame", "full-frame-smoke", "full-frame-check", "full-frame-eval", "full-frame-vrai", "full-frame-multiview", "full-frame-multiview-check", "full-frame-multiview-eval", "full-frame-multiview-vrai", "kitti-mots", "teacher-setup", "teacher-check", "teacher", "reid-teacher", "reid-teacher-check", "joint-distill", "joint-distill-check", "joint-distill-control", "reid-overfit", "compare-reid", "compare-reid-check", "compare-roi", "compare-roi-check", "compare-crops", "compare-crops-check", "vrai-adapt", "vrai-adapt-check"),
                        help="optional matched comparison or separate VisDrone training")
    options = parser.parse_args()
    if options.mode in ("vrai-adapt", "vrai-adapt-check"):
        from utils.experiments.vrai_adaptation.command import run as adapt
        return adapt(check=options.mode.endswith("-check"))
    if options.mode in ("compare-crops", "compare-crops-check"):
        from utils.experiments.crop_training.command import run as compare_crops
        return compare_crops(check=options.mode.endswith("-check"))
    if options.mode in ("compare-roi", "compare-roi-check"):
        from utils.experiments.roi_reid.command import run as compare_roi
        return compare_roi(check=options.mode.endswith("-check"))
    if options.mode in ("compare-reid", "compare-reid-check"):
        from utils.experiments.reid_balance.command import run as compare_reid
        return compare_reid(check=options.mode.endswith("-check"))
    if options.mode == "reid-overfit":
        from utils.teacher.launcher import launch
        return launch("reid-overfit")
    if options.mode and options.mode.startswith("joint-distill"):
        from utils.distillation.command import run as distill
        return distill(options.mode)
    if options.mode in ("reid-teacher", "reid-teacher-check"):
        from utils.reid_teacher.training import train as specialist
        return specialist(check=options.mode.endswith('-check'))
    if options.mode in ("teacher-setup", "teacher-check", "teacher"):
        from utils.teacher.launcher import setup, launch
        return setup() if options.mode == "teacher-setup" else launch("check" if options.mode == "teacher-check" else "train")
    if options.mode == "kitti-mots":
        from utils.full_frame.protocol import train as train_benchmark
        return train_benchmark()
    if options.mode == "full-frame-smoke":
        from utils.full_frame.training import smoke_test
        return smoke_test()
    if options.mode == "full-frame-check":
        from utils.full_frame.training import check_dataset
        return check_dataset(CONFIG)
    if options.mode == "full-frame":
        from utils.full_frame.training import train as train_full_frame
        return train_full_frame(CONFIG)
    if options.mode == "full-frame-eval":
        from utils.full_frame.evaluate import evaluate as evaluate_full_frame
        return evaluate_full_frame(CONFIG)
    if options.mode == "full-frame-vrai":
        from utils.full_frame.cross_view import evaluate as evaluate_full_frame_vrai
        return evaluate_full_frame_vrai(CONFIG)
    if options.mode in ("full-frame-multiview", "full-frame-multiview-check"):
        from utils.full_frame.multiview import train as train_multiview
        return train_multiview(CONFIG, check_only=options.mode.endswith("-check"))
    if options.mode == "full-frame-multiview-eval":
        from utils.full_frame.evaluate import evaluate as evaluate_full_frame
        return evaluate_full_frame(CONFIG, multiview=True)
    if options.mode == "full-frame-multiview-vrai":
        from utils.full_frame.cross_view import evaluate as evaluate_full_frame_vrai
        return evaluate_full_frame_vrai(CONFIG, multiview=True)
    if options.mode == "compare":
        from utils.experiments.compare import main as compare
        return compare()
    if options.mode == "compare-vehicles":
        from utils.experiments.vehicle_compare import main as compare_vehicles
        return compare_vehicles()
    if options.mode == "warmstart-vehicles":
        from utils.experiments.vehicle_warmstart import main as warmstart_vehicles
        return warmstart_vehicles()
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
