"""Combine paired AI Camera captures into a frozen query/gallery evaluation.

Each scene has one query object from camera A and one or more candidate objects
from camera B. Record an absent scene by omitting the query identity from B.
This builder chooses the earliest valid capture per physical object per scene.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def build(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    if not rows:
        raise ValueError("no capture records")
    models = {row["model_sha256"] for row in rows}
    if len(models) != 1:
        raise ValueError("captures use different RPK model hashes")
    group_splits: dict[str, str] = {}
    identity_splits: dict[str, str] = {}
    scenes: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    seen_ids = set()
    for row in rows:
        if row["observation_id"] in seen_ids:
            raise ValueError(f"duplicate observation: {row['observation_id']}")
        seen_ids.add(row["observation_id"])
        split = row["split"]
        if split not in ("validation", "test"):
            raise ValueError(f"unsupported split: {split}")
        group = row["capture_group"]
        identity = row["identity_id"]
        if group_splits.setdefault(group, split) != split:
            raise ValueError(f"capture group {group} crosses splits")
        if identity_splits.setdefault(identity, split) != split:
            raise ValueError(f"physical identity {identity} crosses splits")
        if len(row["embedding"]) != 64:
            raise ValueError(f"wrong embedding length for {row['observation_id']}")
        scenes[(split, group, row["scene_id"])].append(row)
    observations = []
    trials = []
    embeddings = []
    selected_ids = set()
    for (split, group, scene), members in sorted(scenes.items()):
        queries = [row for row in members if row["role"] == "query"]
        gallery = [row for row in members if row["role"] == "candidate"]
        if not queries or not gallery:
            raise ValueError(f"scene {group}/{scene} needs query and candidate captures")
        query_identities = {row["identity_id"] for row in queries}
        query_cameras = {row["camera_id"] for row in queries}
        gallery_cameras = {row["camera_id"] for row in gallery}
        if len(query_identities) != 1 or len(query_cameras) != 1 or len(gallery_cameras) != 1:
            raise ValueError(f"scene {group}/{scene} needs one query and one gallery camera")
        if query_cameras == gallery_cameras:
            raise ValueError(f"scene {group}/{scene} is same-camera")
        query = min(queries, key=lambda row: row["timestamp_us"])
        candidates_by_identity: dict[str, dict] = {}
        for row in sorted(gallery, key=lambda member: member["timestamp_us"]):
            candidates_by_identity.setdefault(row["identity_id"], row)
        candidates = list(candidates_by_identity.values())
        chosen = [query, *candidates]
        for row in chosen:
            key = row["observation_id"]
            if key in selected_ids:
                raise ValueError(f"observation reused in two scenes: {key}")
            selected_ids.add(key)
            observations.append({key: value for key, value in row.items()
                                 if key not in ("embedding", "embedding_size", "output_shapes",
                                                "model_sha256", "kpi_ms", "role", "scene_id")})
            embeddings.append({"observation_id": key, "embedding": row["embedding"],
                               "model_sha256": row["model_sha256"],
                               "kpi_ms": row.get("kpi_ms")})
        trials.append({
            "trial_id": f"{group}/{scene}",
            "split": split,
            "query": query["observation_id"],
            "candidates": [row["observation_id"] for row in candidates],
            "target_identity_id": query["identity_id"],
            "present": query["identity_id"] in candidates_by_identity,
        })
    for split in ("validation", "test"):
        subset = [row for row in trials if row["split"] == split]
        if not any(row["present"] for row in subset) or not any(not row["present"] for row in subset):
            raise ValueError(f"{split} needs at least one present and one absent scene")
    return observations, trials, embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "on_sensor_handoff/data/live_eval_v1")
    args = parser.parse_args()
    records = []
    for capture_file in args.captures:
        records.extend(json.loads(line) for line in capture_file.read_text().splitlines()
                       if line.strip())
    observations, trials, embeddings = build(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "observations.jsonl", observations)
    write_jsonl(args.output_dir / "trials.jsonl", trials)
    write_jsonl(args.output_dir / "imx500_embeddings.jsonl", embeddings)
    print(f"wrote {len(trials)} trials, {len(observations)} observations to {args.output_dir}")


if __name__ == "__main__":
    main()
