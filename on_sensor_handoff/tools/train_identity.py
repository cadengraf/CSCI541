"""Fit a 64-D cross-camera projection on frozen EfficientNet-B0 features.

The backbone remains frozen for this first model. Training pairs must contain
physical instance identities across cameras; category labels are not enough.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from on_sensor_handoff.reid.model import (
    BACKBONE, EMBEDDING_SIZE, INPUT_SIZE, IdentityEncoder, observation_input,
)


ROOT = Path(__file__).resolve().parents[2]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def feature_matrix(model: IdentityEncoder, rows: list[dict], base_dir: Path,
                   batch_size: int, device: torch.device) -> np.ndarray:
    features = []
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = np.stack([observation_input(row, base_dir) for row in rows[start:start + batch_size]])
            tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
            features.append(model.features(tensor).cpu().numpy())
            print(f"backbone features {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(features).astype(np.float32)


def cross_camera_groups(rows: list[dict]) -> dict[str, dict[str, list[int]]]:
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        groups[row["identity_id"]][row["camera_id"]].append(index)
    eligible = {identity: cameras for identity, cameras in groups.items()
                if len(cameras) >= 2}
    if len(eligible) < 2:
        raise ValueError("need at least two identities observed by multiple cameras")
    return eligible


def sample_pair(cameras: dict[str, list[int]], rng: random.Random) -> tuple[int, int]:
    first, second = rng.sample(sorted(cameras), 2)
    return rng.choice(cameras[first]), rng.choice(cameras[second])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path,
                        default=ROOT / "on_sensor_handoff/data/aicity_train_v1/observations.jsonl")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "on_sensor_handoff/artifacts/identity_b0_64.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-identities", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=541)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_identities < 2 or args.temperature <= 0:
        parser.error("epochs >= 1, batch-identities >= 2, and temperature > 0 are required")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows(args.observations)
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("training observations must all have split=train")
    if len({row["observation_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate observation_id in training manifest")
    groups = cross_camera_groups(rows)
    model = IdentityEncoder(pretrained=True)
    features = feature_matrix(model, rows, args.base_dir, args.feature_batch_size, device)
    feature_tensor = torch.from_numpy(features).to(device)
    # Only the small projection trains; preserving the pretrained backbone is
    # useful when the training identities cover vehicles alone.
    model.projection.to(device).train()
    optimizer = torch.optim.AdamW(model.projection.parameters(),
                                   lr=args.learning_rate, weight_decay=0.0001)
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for identity in groups:
        by_scenario[identity.split("/", 1)[0]].append(identity)
    for epoch in range(args.epochs):
        losses = []
        for scenario, identities in sorted(by_scenario.items()):
            if len(identities) < 2:
                continue
            shuffled = identities.copy()
            rng.shuffle(shuffled)
            for start in range(0, len(shuffled), args.batch_identities):
                chosen = shuffled[start:start + args.batch_identities]
                if len(chosen) < 2:
                    continue
                a, b = zip(*(sample_pair(groups[identity], rng) for identity in chosen))
                left = F.normalize(model.projection(feature_tensor[list(a)]), dim=1)
                right = F.normalize(model.projection(feature_tensor[list(b)]), dim=1)
                logits = left @ right.T / args.temperature
                labels = torch.arange(len(chosen), device=device)
                loss = (F.cross_entropy(logits, labels) +
                        F.cross_entropy(logits.T, labels)) / 2
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError("no training updates: each scenario needs at least two cross-camera identities")
        print(f"epoch {epoch + 1}/{args.epochs} loss={np.mean(losses):.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    checkpoint = {
        "backbone": BACKBONE,
        "input_size": INPUT_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "model_state": model.state_dict(),
        "training_manifest_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
        "identities": len(groups),
        "epochs": args.epochs,
        "seed": args.seed,
        "preprocessing": "RGB float 0..1; square resize INTER_LINEAR; ImageNet normalization in model",
    }
    torch.save(checkpoint, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
