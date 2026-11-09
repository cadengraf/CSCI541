"""Prepare datasets and manifests using config.yaml or explicit source paths."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("aicity", "visdrone", "vrai", "vru", "isaid", "vehicle-masks", "kitti",
                                            "bucktales", "sop", "abo", "folders", "sues",
                                            "visdrone-folders"))
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="source options for sop, abo, folders, sues, or visdrone-folders")
    args = parser.parse_args()
    if args.dataset == "kitti":
        if args.rest:
            parser.error('kitti uses config.yaml and accepts no extra arguments')
        from pathlib import Path
        import yaml
        from utils.full_frame.kitti import unpack_kitti
        config_path = Path(__file__).with_name('config.yaml')
        settings = yaml.safe_load(config_path.read_text())['full_frame']
        root = Path(settings['data_root']).expanduser()
        if not root.is_absolute():
            root = config_path.parent / root
        def archive_path(value):
            path = Path(value).expanduser()
            return path if path.is_absolute() else config_path.parent / path
        return unpack_kitti(root, archive_path(settings['images_archive']),
                            archive_path(settings['labels_archive']))
    if args.dataset == "aicity":
        from utils.data_sources.prepare_aicity import main as run
    elif args.dataset == "visdrone":
        from utils.data_sources.visdrone import main as run
    elif args.dataset == "vrai":
        from utils.data_sources.vrai import main as run
    elif args.dataset == "vru":
        from utils.data_sources.vru import main as run
    elif args.dataset == "isaid":
        from utils.data_sources.isaid_masks import main as run
    elif args.dataset == "vehicle-masks":
        from utils.data_sources.vehicle_masks import main as run
    elif args.dataset == "bucktales":
        from utils.crossdrone.command import main as run
        mode = args.rest or ["prepare"]
        if len(mode) != 1 or mode[0] not in ("inspect", "prepare", "audit"):
            parser.error("bucktales accepts one optional mode: inspect, prepare, or audit")
        return run(mode[0])
    elif args.dataset in ("sop", "abo"):
        from utils.data_sources.fetch_data import main as run
        return run([args.dataset, *args.rest])
    else:
        from utils.data_sources.prepare_aerial import main as run
        source = "visdrone" if args.dataset == "visdrone-folders" else args.dataset
        return run([source, *args.rest])
    if args.rest:
        parser.error(f"{args.dataset} uses config.yaml and accepts no extra arguments")
    return run()


if __name__ == "__main__":
    main()
