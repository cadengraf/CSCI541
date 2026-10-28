"""Convert the existing AICity handoff benchmark into cross-camera crop manifests."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image


def _probe_size(video):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
                            capture_output=True, text=True, check=True)
    width, height = map(int, result.stdout.strip().split(","))
    return width, height


def _read_exact(stream, count):
    chunks = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise EOFError(f"ffmpeg stopped with {remaining} bytes still needed")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _safe_box(raw):
    if len(raw) != 4:
        raise ValueError(f"Expected x1,y1,x2,y2 box, got {raw}")
    return tuple(round(float(value)) for value in raw)


def _crop_path(output, record):
    partition, identity, camera, video, frame, box = record
    scenario, vehicle = identity.split(":", 1)
    camera_name = camera.rsplit("/", 1)[-1]
    fingerprint = hashlib.sha1(f"{video}:{frame}:{box}".encode()).hexdigest()[:10]
    return output / "crops" / partition / scenario / camera_name / f"{vehicle}_{frame:06d}_{fingerprint}.jpg"


def _collect(manifest, max_trials=None):
    records = set()
    seen = 0
    with manifest.open() as file:
        for line in file:
            trial = json.loads(line)
            if trial["dataset_partition"] not in {"train", "validation"}:
                continue
            seen += 1
            if max_trials and seen > max_trials:
                break
            partition = ("train" if trial["dataset_partition"] == "train"
                         else trial["benchmark_split"])
            if partition not in {"train", "calibration", "test"}:
                raise ValueError(f"Unexpected AICity split: {partition}")
            identity = f"{trial['scenario']}:{trial['target_id']}"
            source = Path(trial["source_video"])
            camera = trial["source_camera"]
            for sample in trial["source_samples"]:
                records.add((partition, identity, camera, source, int(sample["frame"]),
                             _safe_box(sample["bbox_xyxy"])))
            if trial["present"] and trial["destination_bbox_xyxy"] is not None:
                records.add((partition, identity, trial["destination_camera"],
                             Path(trial["destination_video"]), int(trial["destination_frame"]),
                             _safe_box(trial["destination_bbox_xyxy"])))
    return records


def _add_training_tracks(records):
    """Expand training identities with labeled frames from the same AICity videos."""
    wanted = defaultdict(set)
    for partition, identity, camera, video, *_ in records:
        if partition == "train":
            wanted[(video, camera)].add(identity)
    expanded = set(records)
    for (video, camera), identities in wanted.items():
        ground_truth = video.parent / "gt" / "gt.txt"
        if not ground_truth.is_file():
            raise FileNotFoundError(f"Missing ground truth for {video}: {ground_truth}")
        scenario = camera.split("/", 1)[0]
        with ground_truth.open(newline="") as file:
            for row in csv.reader(file):
                if len(row) < 6:
                    continue
                frame, object_id = int(float(row[0])), int(float(row[1]))
                identity = f"{scenario}:{object_id}"
                if identity not in identities or (len(row) > 6 and float(row[6]) != 1):
                    continue
                x, y, width, height = (round(float(value)) for value in row[2:6])
                if width >= 16 and height >= 16:
                    expanded.add(("train", identity, camera, video, frame,
                                  (x, y, x + width, y + height)))
    return expanded


def _sample(records, maximum):
    groups = defaultdict(list)
    for record in records:
        groups[(record[0], record[1], record[2])].append(record)
    sampled = set()
    for group in groups.values():
        group.sort(key=lambda record: (record[4], record[3].as_posix(), record[5]))
        if len(group) <= maximum:
            sampled.update(group)
        else:
            indices = {round(i * (len(group) - 1) / (maximum - 1)) for i in range(maximum)}
            sampled.update(group[index] for index in indices)
    return sampled


def _eligible(records):
    by_identity = defaultdict(set)
    for partition, identity, camera, *_ in records:
        by_identity[(partition, identity)].add(camera)
    return {record for record in records if len(by_identity[record[:2]]) >= 2}


def _extract(records, output):
    by_video = defaultdict(lambda: defaultdict(list))
    for record in records:
        destination = _crop_path(output, record)
        if not destination.is_file():
            by_video[record[3]][record[4]].append((record, destination))
    for number, (video, frames) in enumerate(sorted(by_video.items()), 1):
        if not video.is_file():
            raise FileNotFoundError(f"AICity video is missing: {video}")
        width, height = _probe_size(video)
        selected = sorted(frames)
        expression = "+".join(f"eq(n\\,{frame - 1})" for frame in selected)
        command = ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", f"select={expression}",
                   "-vsync", "0", "-frames:v", str(len(selected)), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in selected:
                image = Image.frombytes("RGB", (width, height), _read_exact(process.stdout, width * height * 3))
                for record, destination in frames[frame]:
                    left, top, right, bottom = record[5]
                    box = (max(0, left), max(0, top), min(width, right), min(height, bottom))
                    if box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError(f"Invalid box {record[5]} in {video}:{frame}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image.crop(box).save(destination, "JPEG", quality=92)
            process.stdout.close()
            error = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed on {video}: {error}")
        except BaseException:
            process.kill()
            process.wait()
            raise
        print(f"Extracted video {number}/{len(by_video)}: {video.name}", flush=True)


def _write(path, records, output):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("path", "id", "camera"))
        for record in sorted(records, key=lambda row: (row[1], row[2], row[4], row[5])):
            image = _crop_path(output, record)
            writer.writerow((os.path.relpath(image, path.parent), record[1], record[2]))


def prepare(manifest, output, max_per_camera, max_trials=None):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required to read the AICity videos")
    if max_per_camera < 2:
        raise ValueError("max_per_camera must be at least 2")
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing benchmark manifest: {manifest}")
    records = _eligible(_sample(_add_training_tracks(_collect(manifest, max_trials)), max_per_camera))
    if not records:
        raise ValueError("No identities appear in two cameras")
    train = {record for record in records if record[0] == "train"}
    calibration = {record for record in records if record[0] == "calibration"}
    test = {record for record in records if record[0] == "test"}
    if not train or not calibration or not test:
        raise ValueError("The benchmark needs train, calibration, and test identities")
    train_ids = {record[1] for record in train}
    calibration_ids = {record[1] for record in calibration}
    test_ids = {record[1] for record in test}
    if train_ids & calibration_ids or train_ids & test_ids or calibration_ids & test_ids:
        raise ValueError("Train, calibration, and test physical IDs must be disjoint")

    def queries(rows):
        by_id_camera = defaultdict(list)
        for record in rows:
            by_id_camera[(record[1], record[2])].append(record)
        return {sorted(group, key=lambda row: row[4])[0] for group in by_id_camera.values()}

    calibration_query = queries(calibration)
    test_query = queries(test)
    print(f"AICity: {len(train)} train crops / {len(train_ids)} IDs, "
          f"{len(calibration_query)} calibration queries / {len(calibration_ids)} IDs, "
          f"{len(test_query)} test queries / {len(test_ids)} IDs", flush=True)
    _extract(train | calibration | test, output)
    _write(output / "train.csv", train, output)
    _write(output / "calibration_query.csv", calibration_query, output)
    _write(output / "calibration_gallery.csv", calibration, output)
    _write(output / "query.csv", test_query, output)
    _write(output / "gallery.csv", test, output)
    print(f"Prepared {output}")
