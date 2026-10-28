"""BuckTales pair metadata, crop preparation, and observation validation."""

import csv
import math
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


def inspect_archive(path):
    if path.is_dir():
        print(f"Dataset: {path}")
        print("Drone pairs:")
        for name, _, _, _ in dataset_pairs(path):
            print(f"  {name}")
        readme = path / "README.txt"
        if readme.is_file():
            print(f"\n===== {readme} =====")
            print(readme.read_text(errors="replace"))
        return
    if not path.is_file():
        raise FileNotFoundError(f"Missing BuckTales dataset: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        print(f"Archive: {path} ({len(names)} entries)")
        print("First entries:")
        for name in names[:40]:
            print(f"  {name}")
        readmes = [name for name in names if Path(name).name.lower().startswith("readme")
                   and name.lower().endswith((".txt", ".md"))]
        if not readmes:
            print("No README found in archive")
        for name in readmes:
            info = archive.getinfo(name)
            if info.file_size > 100_000:
                print(f"Skipping large README: {name} ({info.file_size} bytes)")
                continue
            print(f"\n===== {name} =====")
            print(archive.read(name).decode("utf-8-sig", errors="replace"))


def dataset_pairs(root):
    """Return the published two-video pair list in a stable order."""
    index = defaultdict(int)
    with (root / "id_matching.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            set_id = row["set_id"]
            index[set_id] += 1
            name = f"set{set_id.split()[-1]}_pair{index[set_id]}"
            yield name, set_id, row["video_1"], row["video_2"]


def _integer(value):
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"Expected integer annotation, got {value!r}")
    return int(number)


def _pair_matches(path):
    """Keep only unambiguous one-to-one local-ID links."""
    raw = []
    with path.open(newline="") as file:
        for row in csv.reader(file):
            if not row or row[0] == "" or row[0].lower() == "serialno":
                continue
            if len(row) < 7:
                continue
            try:
                start_a, end_a, id_a = map(_integer, row[1:4])
                start_b, end_b, id_b = map(_integer, row[4:7])
            except ValueError:
                continue
            if start_a <= end_a and start_b <= end_b:
                raw.append((id_a, id_b, start_a, end_a, start_b, end_b))
    links_a, links_b = defaultdict(set), defaultdict(set)
    for id_a, id_b, *_ in raw:
        links_a[id_a].add(id_b)
        links_b[id_b].add(id_a)
    valid = [(a, b, sa, ea, sb, eb) for a, b, sa, ea, sb, eb in raw
             if len(links_a[a]) == len(links_b[b]) == 1]
    print(f"{path.name}: {len(valid)} unambiguous matches of {len(raw)} rows", flush=True)
    return valid


def _pick_frames(track_path, ranges, count):
    available = defaultdict(set)
    with track_path.open(newline="") as file:
        for row in csv.DictReader(file):
            try:
                identity, frame = _integer(row["id"]), _integer(row["frame"])
            except (ValueError, KeyError):
                continue
            for label, start, end in ranges.get(identity, ()):
                if start <= frame <= end:
                    available[label].add(frame)
    frames = set()
    for label, found in available.items():
        ordered = sorted(found)
        for i in range(min(count, len(ordered))):
            index = round((i + .5) * len(ordered) / min(count, len(ordered)) - .5)
            frames.add(ordered[index])
    return frames


def _boxes_at_frames(track_path, frames, ranges):
    boxes = defaultdict(list)
    with track_path.open(newline="") as file:
        for row in csv.DictReader(file):
            try:
                frame, identity = _integer(row["frame"]), _integer(row["id"])
                x, y = float(row["bb_left"]), float(row["bb_top"])
                w, h = float(row["bb_width"]), float(row["bb_height"])
            except (ValueError, KeyError):
                continue
            if frame in frames and w > 0 and h > 0:
                labels = {label for label, start, end in ranges.get(identity, ())
                          if start <= frame <= end}
                if len(labels) == 1:
                    boxes[frame].append((labels.pop(), (x, y, w, h)))
    return boxes


def _extract_crops(video, boxes, output, pair, camera, writer):
    """Decode selected frames once and retain only small object JPEGs."""
    if not boxes:
        return 0
    output.mkdir(parents=True, exist_ok=True)
    frames = sorted(boxes)
    expression = "+".join(f"eq(n\\,{frame})" for frame in frames)
    saved = 0
    with tempfile.TemporaryDirectory(prefix="bucktales_frames_") as temp:
        temporary = Path(temp)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                   "-vf", f"select='{expression}'", "-vsync", "0", "-q:v", "3",
                   str(temporary / "%06d.jpg")]
        subprocess.run(command, check=True)
        extracted = sorted(temporary.glob("*.jpg"))
        if len(extracted) != len(frames):
            raise RuntimeError(f"{video}: requested {len(frames)} frames, got {len(extracted)}")
        for source, frame in zip(extracted, frames):
            with Image.open(source) as image:
                for number, (identity, (x, y, w, h)) in enumerate(boxes[frame]):
                    # Include some context while keeping the animal prominent.
                    margin = .15
                    left = max(0, math.floor(x - w * margin))
                    top = max(0, math.floor(y - h * margin))
                    right = min(image.width, math.ceil(x + w * (1 + margin)))
                    bottom = min(image.height, math.ceil(y + h * (1 + margin)))
                    if right <= left or bottom <= top:
                        continue
                    destination = output / f"{camera}_{frame:06d}_{number:03d}.jpg"
                    image.crop((left, top, right, bottom)).convert("RGB").save(destination, quality=90)
                    writer.writerow([pair, camera, identity, frame, str(destination.resolve()), "", "", "", ""])
                    saved += 1
            source.unlink()
    return saved


def prepare_dataset(root, manifest, pair=None, samples_per_identity=4):
    if not root.is_dir():
        raise FileNotFoundError(f"Expected extracted BuckTales dataset at {root}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to extract BuckTales crops")
    pairs = list(dataset_pairs(root))
    selected = [entry for entry in pairs if pair is None or entry[0] == pair]
    if not selected:
        raise ValueError(f"Unknown pair {pair!r}; choices: {[entry[0] for entry in pairs]}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    columns = ["pair", "camera", "id", "frame", "path", "x", "y", "w", "h"]
    existing = []
    if manifest.is_file():
        read_observations(manifest)  # Check that every previously prepared crop still exists.
        with manifest.open(newline="") as file:
            existing = list(csv.DictReader(file))
    # Earlier preparation could include an ID's crop outside its annotated time
    # range when that ID appeared in a frame sampled for another animal.
    if existing:
        allowed = defaultdict(list)
        for name, set_id, camera_a, camera_b in pairs:
            folder = root / "data" / set_id
            for a, b, sa, ea, sb, eb in _pair_matches(folder / f"{camera_a}_{camera_b}.csv"):
                label = f"animal_{a}_{b}"
                allowed[name, camera_a, label].append((sa, ea))
                allowed[name, camera_b, label].append((sb, eb))
        clean = []
        for row in existing:
            if not row["id"].startswith("animal_"):
                clean.append(row)
                continue
            spans = allowed[row["pair"], row["camera"], row["id"]]
            if any(start <= int(row["frame"]) <= end for start, end in spans):
                clean.append(row)
        if len(clean) != len(existing):
            print(f"Discarding {len(existing) - len(clean)} old crops outside their "
                  "published ID time ranges", flush=True)
            partial = manifest.with_suffix(".partial.csv")
            with partial.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(clean)
            partial.replace(manifest)
            existing = clean
    pair_id_cameras = defaultdict(set)
    for row in existing:
        pair_id_cameras[row["pair"], row["id"]].add(row["camera"])
    completed = {pair for (pair, _), cameras in pair_id_cameras.items() if len(cameras) == 2}
    total = 0
    for name, set_id, camera_a, camera_b in selected:
        if pair is None and name in completed:
            print(f"{name}: already prepared; skipping", flush=True)
            continue
        pair_partial = manifest.parent / f".{name}.partial.csv"
        try:
            with pair_partial.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                folder = root / "data" / set_id
                matches = _pair_matches(folder / f"{camera_a}_{camera_b}.csv")
                if not matches:
                    raise ValueError(f"No unambiguous identities in {name}")
                for camera, side in ((camera_a, 0), (camera_b, 1)):
                    ranges = defaultdict(list)
                    for match in matches:
                        local_id = match[side]
                        start, end = match[2:4] if side == 0 else match[4:6]
                        label = f"animal_{match[0]}_{match[1]}"
                        ranges[local_id].append((label, start, end))
                    track = folder / f"{camera}_tracks.csv"
                    frames = _pick_frames(track, ranges, samples_per_identity)
                    boxes = _boxes_at_frames(track, frames, ranges)
                    print(f"{name} {camera}: {len(frames)} sampled frames", flush=True)
                    total += _extract_crops(folder / f"{camera}.MP4", boxes,
                                            manifest.parent / "crops" / name, name, camera, writer)
            with pair_partial.open(newline="") as file:
                new_rows = list(csv.DictReader(file))
            identities = defaultdict(set)
            for row in new_rows:
                identities[row["id"]].add(row["camera"])
            if not any(cameras == {camera_a, camera_b} for cameras in identities.values()):
                raise ValueError(f"{name} did not produce an identity seen by both drones")
            merged = [row for row in existing if row["pair"] != name] + new_rows
            partial = manifest.with_suffix(".partial.csv")
            try:
                with partial.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(merged)
                partial.replace(manifest)
            finally:
                partial.unlink(missing_ok=True)
            existing = merged
            print(f"{name}: saved {len(new_rows)} crops; progress is resumable", flush=True)
        finally:
            pair_partial.unlink(missing_ok=True)
    print(f"Prepared {total} new crops in {manifest} across "
          f"{len({row['pair'] for row in existing})} pairs", flush=True)


@dataclass(frozen=True)
class Observation:
    path: Path
    pair: str
    identity: str
    camera: str
    frame: int
    box: tuple[int, int, int, int] | None


def read_observations(manifest):
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing {manifest}. Download BuckTales ReIdentification_Dataset.zip and "
            "prepare the paired-drone observation CSV described in README.md")
    rows = []
    with manifest.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"pair", "camera", "id", "frame", "path"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{manifest} needs columns: {sorted(required)}")
        for number, row in enumerate(reader, 2):
            if any(not (row.get(key) or "").strip() for key in required):
                raise ValueError(f"Missing required value at {manifest}:{number}")
            path = Path(row["path"]).expanduser()
            if not path.is_absolute():
                path = manifest.parent / path
            if not path.is_file():
                raise FileNotFoundError(f"Missing image at {manifest}:{number}: {path}")
            box_fields = [(row.get(key) or "").strip() for key in ("x", "y", "w", "h")]
            if any(box_fields) and not all(box_fields):
                raise ValueError(f"Crop at {manifest}:{number} needs x,y,w,h together")
            box = None
            if all(box_fields):
                x, y, width, height = (int(float(value)) for value in box_fields)
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid crop at {manifest}:{number}")
                box = x, y, x + width, y + height
            frame = int(row["frame"])
            if frame < 0:
                raise ValueError(f"Negative frame number at {manifest}:{number}")
            rows.append(Observation(path.resolve(), row["pair"].strip(), row["id"].strip(),
                                    row["camera"].strip(), frame, box))
    if not rows:
        raise ValueError(f"No observations in {manifest}")
    if len(set(rows)) != len(rows):
        raise ValueError(f"Duplicate observations in {manifest}")
    pairs = defaultdict(set)
    for row in rows:
        pairs[row.pair].add(row.camera)
    invalid = {pair: cameras for pair, cameras in pairs.items() if len(cameras) != 2}
    if invalid:
        raise ValueError(f"Each drone pair must have exactly two camera labels: {invalid}")
    return rows


def audit_split(root, rows):
    """Report recording links that make naive pair-level splits leak data."""
    pairs = list(dataset_pairs(root))
    observed = {row.pair for row in rows}
    names = {name for name, *_ in pairs}
    if observed - names:
        raise ValueError(f"Unknown BuckTales pairs in manifest: {sorted(observed - names)}")
    if names - observed:
        print(f"Unprepared pairs: {sorted(names - observed)}", flush=True)
    by_video = defaultdict(set)
    by_recording = defaultdict(set)
    for name, _, camera_a, camera_b in pairs:
        if name not in observed:
            continue
        for camera in (camera_a, camera_b):
            by_video[camera].add(name)
            # The source filenames encode recording date and location.
            by_recording["_".join(camera.split("_")[:3])].add(name)
    shared = {camera: sorted(linked) for camera, linked in by_video.items()
              if len(linked) > 1}
    print(f"BuckTales: {len(rows)} crops across {len(observed)} prepared drone pairs", flush=True)
    for recording, linked in sorted(by_recording.items()):
        print(f"  recording {recording}: {', '.join(sorted(linked))}", flush=True)
    for camera, linked in sorted(shared.items()):
        print(f"  SHARED VIDEO {camera}: {', '.join(linked)}", flush=True)
    print("Pair-disjoint train/test splits can reuse an identical video. "
          "The supplied IDs are pair-local and do not certify identity "
          "separation across recording dates.", flush=True)
