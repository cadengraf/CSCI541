"""BuckTales crop loading and cross-drone embedding retrieval."""

from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ShuffleNet_V2_X1_5_Weights

from utils.core.metrics import cross_view_metrics, image_vectors
from utils.core.model import JointNet


def crop(row):
    with Image.open(row.path) as source:
        image = source.convert("RGB")
        if row.box is not None:
            left, top, right, bottom = row.box
            box = max(0, left), max(0, top), min(image.width, right), min(image.height, bottom)
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Crop outside image: {row.path}, {row.box}")
            image = image.crop(box)
        return image


class Observations(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.labels = [f"{row.pair}:{row.identity}" for row in rows]
        self.cameras = [row.camera for row in rows]
        self.transform = ShuffleNet_V2_X1_5_Weights.DEFAULT.transforms()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.transform(crop(self.rows[index])), self.labels[index]


def sample_rows(rows, maximum):
    if maximum < 1:
        raise ValueError("max_gallery_per_identity must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[(row.pair, row.identity, row.camera)].append(row)
    sampled = []
    queries = []
    for group in groups.values():
        group.sort(key=lambda row: (row.frame, str(row.path), str(row.box)))
        if len(group) > maximum:
            indices = {round(i * (len(group) - 1) / (maximum - 1)) for i in range(maximum)} if maximum > 1 else {0}
            group = [group[index] for index in sorted(indices)]
        sampled.extend(group)
        queries.append(group[0])
    return sampled, queries


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not saved.get("use_mask_pool", True):
        raise ValueError("Cross-drone ablation requires a joint model with a mask head")
    model = JointNet(pretrained=False, use_mask_pool=True)
    model.load_state_dict(saved["model"])
    return model.to(device).eval()


def evaluate(config, rows):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config["checkpoint"], device)
    sampled, queries = sample_rows(rows, config["max_gallery_per_identity"])
    pair_names = sorted({row.pair for row in rows})
    results = {}
    for mask_pool in (True, False):
        model.use_mask_pool = mask_pool
        title = "mask-guided" if mask_pool else "global pooling"
        correct_total = query_total = 0
        ap_total = 0.0
        with torch.inference_mode():
            for pair in pair_names:
                gallery_rows = [row for row in sampled if row.pair == pair]
                cameras = defaultdict(set)
                for row in gallery_rows:
                    cameras[row.identity].add(row.camera)
                query_rows = [row for row in queries if row.pair == pair and len(cameras[row.identity]) == 2]
                if not query_rows:
                    raise ValueError(f"{pair} has no physical IDs mapped across both drones")
                query_data = Observations(query_rows)
                gallery_data = Observations(gallery_rows)
                qv = image_vectors(model, query_data, config["loader"], device)
                gv = image_vectors(model, gallery_data, config["loader"], device)
                correct, total, mean_ap = cross_view_metrics(
                    qv, query_data.labels, query_data.cameras,
                    gv, gallery_data.labels, gallery_data.cameras)
                correct_total += correct
                query_total += total
                ap_total += total * mean_ap
                print(f"{pair} {title}: Rank-1 {correct / total:.2%} ({correct}/{total}), "
                      f"mAP {mean_ap:.2%}", flush=True)
        results[title] = (correct_total / query_total, ap_total / query_total, query_total)
        print(f"All pairs {title}: Rank-1 {correct_total / query_total:.2%}, "
              f"mAP {ap_total / query_total:.2%} ({query_total} queries)", flush=True)
    return results
