"""Identity crop readers, source mixing, evaluation splits, and VOC masks."""

import csv
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models import ShuffleNet_V2_X1_5_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .identity_data import NORMALIZE, item_transform


class InShop(Dataset):
    """DeepFashion In-shop images with item IDs from the official partition file."""

    def __init__(self, root: Path, split: str, augmentation: str = "basic"):
        annotation = root / "Eval" / "list_eval_partition.txt"
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing {annotation}")
        boxes_file = root / "Anno" / "list_bbox_inshop.txt"
        self.boxes = {}
        if boxes_file.is_file():
            for line in boxes_file.read_text().splitlines()[2:]:
                fields = line.split()
                if len(fields) >= 7:
                    self.boxes[fields[0]] = tuple(map(int, fields[-4:]))
        self.samples = []
        for line in annotation.read_text().splitlines()[2:]:
            fields = line.split()
            if len(fields) != 3 or fields[2] != split:
                continue
            name, item_id, _ = fields
            path = root / "Img" / name
            if not path.is_file():
                path = root / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing DeepFashion image: {name}")
            self.samples.append((path, item_id, name))
        if not self.samples:
            raise ValueError(f"No DeepFashion images found for split {split}")
        self.labels = [item_id for _, item_id, _ in self.samples]
        self.transform = (item_transform(augmentation) if split == "train"
                          else ShuffleNet_V2_X1_5_Weights.DEFAULT.transforms())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, item_id, name = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if name in self.boxes:
                x1, y1, x2, y2 = self.boxes[name]
                if x2 > x1 and y2 > y1:
                    image = image.crop((x1, y1, x2, y2))
            return self.transform(image), item_id


class AmazonBerkeleyObjects(Dataset):
    """ABO small catalog photos, split by product listing rather than image."""

    def __init__(self, root: Path, split: str, max_items: int | None = None, augmentation: str = "basic"):
        metadata = root / "images" / "metadata" / "images.csv.gz"
        listings = sorted((root / "listings" / "metadata").glob("listings_*.json.gz"))
        if not metadata.is_file() or not listings:
            raise FileNotFoundError("ABO needs images/metadata/images.csv.gz and listings/metadata/listings_*.json.gz")
        selection = root / f"selected_{split}.txt"
        image_selection = root / f"selected_{split}_images.tsv"
        allowed_paths = None
        if image_selection.is_file():
            allowed_paths = defaultdict(set)
            for line in image_selection.read_text().splitlines():
                item_id, relative = line.split("\t", 1)
                allowed_paths[item_id].add(root / "images" / "small" / relative)
            missing = next((path for paths in allowed_paths.values() for path in paths if not path.is_file()), None)
            if missing:
                raise FileNotFoundError(f"Selected ABO image is missing: {missing}; rerun python prepare.py abo")
        selected_ids = (set(allowed_paths) if allowed_paths is not None else
                        set(selection.read_text().splitlines()) if selection.is_file() else None)
        with gzip.open(metadata, "rt", newline="") as file:
            paths = {row["image_id"]: root / "images" / "small" / row["path"] for row in csv.DictReader(file)}
        self.samples = []
        item_count = 0
        for listing_file in listings:
            with gzip.open(listing_file, "rt") as file:
                for line in file:
                    item = json.loads(line)
                    item_id = f"{item['domain_name']}:{item['item_id']}"
                    # The same product never appears in both splits.
                    is_test = hashlib.sha256(item_id.encode()).digest()[0] < 26
                    if is_test != (split == "test"):
                        continue
                    if selected_ids is not None and item_id not in selected_ids:
                        continue
                    image_ids = dict.fromkeys([item.get("main_image_id"), *(item.get("other_image_id") or [])])
                    images = [paths[image_id] for image_id in image_ids
                              if image_id in paths and paths[image_id].is_file()
                              and (allowed_paths is None or paths[image_id] in allowed_paths[item_id])]
                    if len(images) < 2:
                        continue
                    self.samples.extend((path, item_id) for path in images)
                    item_count += 1
                    if max_items and item_count >= max_items:
                        break
            if max_items and item_count >= max_items:
                break
        if not self.samples:
            raise ValueError(f"No ABO product pairs found in {split} split")
        if allowed_paths is not None and max_items is None and item_count != len(allowed_paths):
            raise ValueError(f"Loaded {item_count}/{len(allowed_paths)} selected ABO items in {split}; rerun python prepare.py abo")
        self.labels = [item_id for _, item_id in self.samples]
        self.transform = (item_transform(augmentation) if split == "train"
                          else ShuffleNet_V2_X1_5_Weights.DEFAULT.transforms())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, item_id = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), item_id


class ManifestItems(Dataset):
    """Repeated identities from a CSV; optional x,y,w,h crops for tracking data."""

    def __init__(self, manifest: Path, training: bool = False, augmentation: str = "basic"):
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing identity manifest: {manifest}")
        self.samples = []
        self.cameras = []
        with manifest.open(newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not {"path", "id"}.issubset(reader.fieldnames):
                raise ValueError(f"{manifest} must have path,id columns")
            for row in reader:
                if not row.get("path") or not row.get("id"):
                    raise ValueError(f"Every row in {manifest} needs a path and identity")
                path = Path(row["path"])
                if not path.is_absolute():
                    path = manifest.parent / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Missing image listed in {manifest}: {path}")
                label = row["id"].strip()
                if not label:
                    raise ValueError(f"Empty identity in {manifest}")
                box = None
                box_fields = [(row.get(key) or "").strip() for key in ("x", "y", "w", "h")]
                if any(box_fields) and not all(box_fields):
                    raise ValueError(f"Crop in {manifest} must have x,y,w,h together")
                if all(box_fields):
                    box = tuple(int(float(row[key])) for key in ("x", "y", "w", "h"))
                    if box[2] <= 0 or box[3] <= 0:
                        raise ValueError(f"Invalid crop box in {manifest}: {box}")
                self.samples.append((path, label, box))
                self.cameras.append((row.get("camera") or "").strip() or None)
        if not self.samples:
            raise ValueError(f"No images in {manifest}")
        self.labels = [label for _, label, _ in self.samples]
        if training and any(count < 2 for count in Counter(self.labels).values()):
            raise ValueError(f"Every training identity in {manifest} needs at least two images")
        self.transform = (item_transform(augmentation) if training
                          else ShuffleNet_V2_X1_5_Weights.DEFAULT.transforms())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, box = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
            if box is not None:
                x, y, width, height = box
                left, top = max(x, 0), max(y, 0)
                right, bottom = min(x + width, image.width), min(y + height, image.height)
                if right <= left or bottom <= top:
                    raise ValueError(f"Crop box lies outside {path}: {box}")
                image = image.crop((left, top, right, bottom))
            return self.transform(image), label


class MixedItems(Dataset):
    """Combine item datasets without letting IDs from different sources collide."""

    def __init__(self, datasets):
        self.datasets = datasets
        self.locations = []
        self.labels = []
        self.cameras = []
        self.sample_sources = []
        item_to_number = {}
        for source, dataset in enumerate(datasets):
            for index, original_id in enumerate(dataset.labels):
                key = (source, original_id)
                if key not in item_to_number:
                    item_to_number[key] = len(item_to_number)
                self.locations.append((source, index))
                self.labels.append(item_to_number[key])
                cameras = getattr(dataset, "cameras", None)
                self.cameras.append(cameras[index] if cameras is not None else None)
                self.sample_sources.append(source)

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        source, local_index = self.locations[index]
        image, _ = self.datasets[source][local_index]
        return image, self.labels[index]


class IdentityPartition(Dataset):
    """Deterministic, disjoint identity half of an evaluation dataset."""

    def __init__(self, dataset, name, partition, seed):
        def in_selection(label):
            digest = hashlib.sha256(f"{seed}:{name}:{label}".encode()).digest()
            return digest[0] < 128

        self.dataset = dataset
        self.indices = [index for index, label in enumerate(dataset.labels)
                        if in_selection(label) == (partition == "selection")]
        if not self.indices:
            raise ValueError(f"{name} has no {partition} identities")
        self.labels = [dataset.labels[index] for index in self.indices]
        if hasattr(dataset, "cameras"):
            self.cameras = [dataset.cameras[index] for index in self.indices]
        if hasattr(dataset, "samples"):
            self.samples = [dataset.samples[index] for index in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def index_partition(dataset, name, partition, seed):
    indices = [index for index in range(len(dataset))
               if (hashlib.sha256(f"{seed}:{name}:{index}".encode()).digest()[0] < 128)
               == (partition == "selection")]
    if not indices:
        raise ValueError(f"{name} has no {partition} examples")
    return Subset(dataset, indices)


class ForegroundVOC(Dataset):
    """VOC pixels 1..20 are foreground; 0 is background; 255 is ignored."""

    def __init__(self, root: Path, split: str, download: bool, augmentation: str = "basic"):
        self.data = VOCSegmentation(root=str(root), year="2012", image_set=split, download=download)
        self.augmentation = augmentation if split == "train" else "none"
        self.image_resize = transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, mask = self.data[index]
        image = image.convert("RGB")
        if self.augmentation in ("moderate", "moderate_occlusion", "strong"):
            top, left, height, width = transforms.RandomResizedCrop.get_params(
                image, scale=(0.85, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(image, top, left, height, width, (224, 224), InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, top, left, height, width, (224, 224), InterpolationMode.NEAREST)
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            image = self.color_jitter(image)
            if self.augmentation == "strong":
                if random.random() < 0.08:
                    image = TF.rgb_to_grayscale(image, num_output_channels=3)
                if random.random() < 0.12:
                    image = TF.gaussian_blur(image, kernel_size=3, sigma=random.uniform(0.1, 1.0))
        else:
            image = self.image_resize(image)
            mask = self.mask_resize(mask)
        return NORMALIZE(transforms.ToTensor()(image)), torch.as_tensor(np.array(mask).copy(), dtype=torch.long)
