"""Prepare datasets and manifests using config.yaml or explicit source paths."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("aicity", "visdrone", "vrai", "vru", "isaid",
                                            "bucktales", "sop", "abo", "folders", "sues",
                                            "visdrone-folders"))
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="source options for sop, abo, folders, sues, or visdrone-folders")
    args = parser.parse_args()
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
