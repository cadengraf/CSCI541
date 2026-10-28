"""Shared crop normalization, augmentation, SOP reader, and identity samplers."""

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.models import ShuffleNet_V2_X1_5_Weights


WEIGHTS = ShuffleNet_V2_X1_5_Weights.DEFAULT
NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def item_transform(augmentation: str):
    if augmentation == "none":
        return WEIGHTS.transforms()
    operations = [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation in ("moderate", "moderate_occlusion", "strong"):
        operations.extend([
            transforms.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.9, 1.1), fill=255),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)
            ], p=0.8),
        ])
    if augmentation == "strong":
        # Keep geometry mild: small aerial crops can lose identity cues easily.
        operations.extend([
            transforms.RandomGrayscale(p=0.08),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.12),
        ])
    operations.extend([transforms.ToTensor(), NORMALIZE])
    if augmentation in ("moderate_occlusion", "strong"):
        # A small, occasional occlusion; keep most identity details visible.
        operations.append(transforms.RandomErasing(p=0.12, scale=(0.02, 0.07), value=0))
    return transforms.Compose(operations)


class PKSampler(Sampler):
    """Sample P identities x K crops, preferring distinct cameras for positives."""

    def __init__(self, labels, cameras, batch_size, instances_per_id, steps_per_epoch=None,
                 sample_sources=None, source_names=None, source_weights=None):
        if instances_per_id < 2 or batch_size % instances_per_id:
            raise ValueError("batch_size must be divisible by instances_per_id >= 2")
        self.by_id = defaultdict(list)
        for index, identity in enumerate(labels):
            self.by_id[identity].append(index)
        self.ids = [identity for identity, indices in self.by_id.items() if len(indices) >= 2]
        self.ids_per_batch = batch_size // instances_per_id
        if len(self.ids) < self.ids_per_batch:
            raise ValueError("Too few identities with multiple images for a P x K batch")
        self.cameras = cameras
        self.instances_per_id = instances_per_id
        self.steps = steps_per_epoch or math.ceil(len(labels) / batch_size)
        self.by_source = defaultdict(list)
        for identity in self.ids:
            index = self.by_id[identity][0]
            source = sample_sources[index] if sample_sources is not None else 0
            self.by_source[source].append(identity)
        self.source_names = source_names or [str(index) for index in range(max(self.by_source) + 1)]
        self.source_weights = source_weights or {}

    def _choose_ids(self):
        active = list(self.by_source)
        weights = [self.source_weights.get(self.source_names[index], 1.0) for index in active]
        total = sum(weights)
        exact = [self.ids_per_batch * weight / total for weight in weights]
        quota = [min(len(self.by_source[source]), math.floor(value)) for source, value in zip(active, exact)]
        remaining = self.ids_per_batch - sum(quota)
        for index in sorted(range(len(active)), key=lambda i: exact[i] - math.floor(exact[i]), reverse=True):
            available = len(self.by_source[active[index]]) - quota[index]
            extra = min(available, remaining)
            quota[index] += extra
            remaining -= extra
            if remaining == 0:
                break
        chosen = []
        for source, count in zip(active, quota):
            chosen.extend(random.sample(self.by_source[source], count))
        if remaining:
            chosen.extend(random.sample([identity for identity in self.ids if identity not in chosen], remaining))
        random.shuffle(chosen)
        return chosen

    def __iter__(self):
        for _ in range(self.steps):
            batch = []
            for identity in self._choose_ids():
                indices = self.by_id[identity]
                by_camera = defaultdict(list)
                for index in indices:
                    by_camera[self.cameras[index]].append(index)
                chosen = []
                if len(by_camera) > 1:
                    for camera in random.sample(list(by_camera), 2):
                        chosen.append(random.choice(by_camera[camera]))
                available = [index for index in indices if index not in chosen]
                remaining = self.instances_per_id - len(chosen)
                chosen.extend(random.sample(available, min(remaining, len(available))))
                if len(chosen) < self.instances_per_id:
                    chosen.extend(random.choices(indices, k=self.instances_per_id - len(chosen)))
                batch.extend(chosen)
            yield batch

    def __len__(self):
        return self.steps


class Products(Dataset):
    def __init__(self, root: Path, split: str, training: bool, augmentation: str = "basic"):
        manifest = root / f"Ebay_{split}.txt"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing {manifest}. See README.md for the dataset layout.")
        with manifest.open(newline="") as file:
            rows = list(csv.DictReader(file, delimiter=" ", skipinitialspace=True))
        self.paths = [root / row["path"] for row in rows]
        self.labels = [int(row["class_id"]) for row in rows]
        if not self.paths:
            raise ValueError(f"No images listed in {manifest}")
        missing = next((path for path in self.paths if not path.is_file()), None)
        if missing:
            raise FileNotFoundError(f"Image listed in {manifest} is missing: {missing}")
        self.transform = item_transform(augmentation) if training else WEIGHTS.transforms()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index]


class PairsByItem(Sampler):
    """Visit every image each epoch, with two images per distinct item in a batch."""

    def __init__(self, labels, batch_size):
        if batch_size < 4 or batch_size % 2:
            raise ValueError("--batch-size must be an even number of at least 4")
        self.by_item = defaultdict(list)
        for index, label in enumerate(labels):
            self.by_item[label].append(index)
        self.items = [item for item, indices in self.by_item.items() if len(indices) >= 2]
        self.items_per_batch = batch_size // 2
        if len(self.items) < self.items_per_batch:
            raise ValueError("Dataset has too few items with two images for this batch size")
        self.rounds = max(math.ceil(len(self.by_item[item]) / 2) for item in self.items)
        self.num_batches = sum(
            math.ceil(sum(len(self.by_item[item]) > 2 * round_number for item in self.items)
                      / self.items_per_batch)
            for round_number in range(self.rounds)
        )

    def __iter__(self):
        shuffled = {item: random.sample(self.by_item[item], len(self.by_item[item]))
                    for item in self.items}
        for round_number in range(self.rounds):
            scheduled = [item for item in self.items if len(shuffled[item]) > 2 * round_number]
            random.shuffle(scheduled)
            for start in range(0, len(scheduled), self.items_per_batch):
                chosen = scheduled[start:start + self.items_per_batch]
                scheduled_count = len(chosen)
                if scheduled_count < self.items_per_batch:
                    available = [item for item in self.items if item not in chosen]
                    chosen += random.sample(available, self.items_per_batch - scheduled_count)
                batch = []
                for position, item in enumerate(chosen):
                    images = shuffled[item]
                    if position < scheduled_count:
                        first = 2 * round_number
                        pair = [images[first], images[(first + 1) % len(images)]]
                    else:
                        pair = random.sample(images, 2)
                    batch.extend(pair)
                yield batch

    def __len__(self):
        return self.num_batches
