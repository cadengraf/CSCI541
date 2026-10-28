"""Download reproducible public subsets of SOP and Amazon Berkeley Objects."""

import argparse
import csv
import gzip
import hashlib
import json
import random
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SOP_URL = "https://huggingface.co/datasets/pawlo2013/StanfordOnlineProducts/resolve/main/Stanford_Online_Products/"
ABO_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/"


def download(url: str, destination: Path):
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response, temp.open("wb") as out:
                shutil.copyfileobj(response, out)
            temp.replace(destination)
            return
        except Exception:
            temp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def download_many(jobs, workers):
    jobs = list(dict(jobs).items())  # deduplicate destinations
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(download, url, path): path for path, url in jobs}
        for number, future in enumerate(as_completed(pending), 1):
            future.result()
            if number % 100 == 0 or number == len(jobs):
                print(f"Downloaded {number}/{len(jobs)} files", flush=True)


def choose_items(by_item, count, seed):
    eligible = [item for item, rows in by_item.items() if len(rows) >= 2]
    random.Random(seed).shuffle(eligible)
    return eligible[:count]


def fetch_sop(args):
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    jobs = {}
    archive = zipfile.ZipFile(args.archive) if args.archive else None
    archive_names = set(archive.namelist()) if archive else set()

    def archive_member(relative):
        matches = [name for name in (relative, f"Stanford_Online_Products/{relative}") if name in archive_names]
        if not matches:
            raise FileNotFoundError(f"{relative} is not in {args.archive}")
        return matches[0]

    for split, count in (("train", args.train_items), ("test", args.test_items)):
        full_manifest = root / f"Ebay_{split}_full.txt"
        if archive:
            if not full_manifest.is_file():
                with archive.open(archive_member(f"Ebay_{split}.txt")) as source, full_manifest.open("wb") as out:
                    shutil.copyfileobj(source, out)
        else:
            download(SOP_URL + full_manifest.name.replace("_full", ""), full_manifest)
        with full_manifest.open(newline="") as file:
            rows = list(csv.DictReader(file, delimiter=" ", skipinitialspace=True))
        by_item = defaultdict(list)
        for row in rows:
            by_item[row["class_id"]].append(row)
        selected = choose_items(by_item, count, args.seed)
        chosen_rows = []
        for item in selected:
            chosen_rows.extend(by_item[item][:args.images_per_item])
        manifest = root / f"Ebay_{split}.txt"
        with manifest.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["image_id", "class_id", "super_class_id", "path"], delimiter=" ")
            writer.writeheader()
            writer.writerows(chosen_rows)
        for row in chosen_rows:
            image_path = row["path"]
            jobs[root / image_path] = SOP_URL + urllib.parse.quote(image_path, safe="/")
        print(f"SOP {split}: {len(selected)} items, {len(chosen_rows)} images")
    if archive:
        for number, destination in enumerate(jobs, 1):
            if not destination.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                relative = destination.relative_to(root).as_posix()
                with archive.open(archive_member(relative)) as source, destination.open("wb") as out:
                    shutil.copyfileobj(source, out)
            if number % 100 == 0 or number == len(jobs):
                print(f"Prepared {number}/{len(jobs)} SOP images", flush=True)
        archive.close()
    else:
        download_many(jobs, args.workers)


def fetch_abo(args):
    root = args.root
    metadata = root / "images" / "metadata" / "images.csv.gz"
    download(ABO_URL + "images/metadata/images.csv.gz", metadata)
    listing_files = []
    for number in "0123456789abcdef":
        path = root / "listings" / "metadata" / f"listings_{number}.json.gz"
        download(ABO_URL + f"listings/metadata/listings_{number}.json.gz", path)
        listing_files.append(path)
    with gzip.open(metadata, "rt", newline="") as file:
        image_paths = {row["image_id"]: row["path"] for row in csv.DictReader(file)}
    groups = {"train": [], "test": []}
    for listing_file in listing_files:
        with gzip.open(listing_file, "rt") as file:
            for line in file:
                listing = json.loads(line)
                item_id = f"{listing['domain_name']}:{listing['item_id']}"
                split = "test" if hashlib.sha256(item_id.encode()).digest()[0] < 26 else "train"
                image_ids = dict.fromkeys([listing.get("main_image_id"), *(listing.get("other_image_id") or [])])
                paths = list(dict.fromkeys(image_paths[image_id] for image_id in image_ids if image_id in image_paths))
                if len(paths) >= 2:
                    groups[split].append((item_id, paths[:args.images_per_item]))
    jobs = {}
    used_images = set()
    for split, count in (("train", args.train_items), ("test", args.test_items)):
        candidates = groups[split][:]
        random.Random(args.seed + (0 if split == "train" else 1)).shuffle(candidates)
        selected = []
        for item, paths in candidates:
            if any(path in used_images for path in paths):
                continue
            selected.append((item, paths))
            used_images.update(paths)
            if len(selected) >= count:
                break
        (root / f"selected_{split}.txt").write_text("\n".join(item for item, _ in selected) + "\n")
        with (root / f"selected_{split}_images.tsv").open("w") as manifest:
            for item, paths in selected:
                for relative in paths:
                    manifest.write(f"{item}\t{relative}\n")
        for item, paths in selected:
            for relative in paths:
                destination = root / "images" / "small" / relative
                jobs[destination] = ABO_URL + "images/small/" + urllib.parse.quote(relative, safe="/")
        print(f"ABO {split}: {len(selected)} items")
    download_many(jobs, args.workers)
    download(ABO_URL + "LICENSE-CC-BY-4.0.txt", root / "LICENSE-CC-BY-4.0.txt")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)
    for name in ("sop", "abo"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--root", type=Path, required=True)
        cmd.add_argument("--train-items", type=int, default=1000)
        cmd.add_argument("--test-items", type=int, default=200)
        cmd.add_argument("--images-per-item", type=int, default=3)
        cmd.add_argument("--workers", type=int, default=8)
        cmd.add_argument("--seed", type=int, default=42)
        if name == "sop":
            cmd.add_argument("--archive", type=Path, help="local original Stanford ZIP; avoids mirror rate limits")
    args = parser.parse_args(argv)
    if min(args.train_items, args.test_items, args.workers) < 1 or args.images_per_item < 2:
        parser.error("item counts/workers must be positive and images-per-item must be at least 2")
    if args.dataset == "sop":
        fetch_sop(args)
    else:
        fetch_abo(args)


if __name__ == "__main__":
    main()
