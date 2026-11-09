"""Audit partitions and sample identity-balanced, two-camera vehicle batches."""

import csv
from collections import defaultdict
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms as T
import yaml
from utils.tracking.common import ROOT


def read():
    return yaml.safe_load((ROOT / 'configs/reid_teacher.yaml').read_text())


def rows(path):
    with Path(path).open() as file:
        records = list(csv.DictReader(file))
    if not records or any(not {'path', 'id', 'camera'} <= set(row) for row in records):
        raise ValueError(f'Missing path/id/camera records: {path}')
    for row in records:
        row['path'] = str(Path(row['path']).resolve())
        if not Path(row['path']).is_file():
            raise FileNotFoundError(row['path'])
    return records


def partitions(settings):
    data = {name: rows(ROOT / path) for name, path in settings['data'].items()}
    groups = [data['train'], data['calibration_query'] + data['calibration_gallery'], data['query'] + data['gallery']]
    for first in range(3):
        for second in range(first+1, 3):
            for key in ('id', 'path'):
                if {row[key] for row in groups[first]} & {row[key] for row in groups[second]}:
                    raise ValueError(f'{key} overlap across train/calibration/final partitions')
    for query, gallery in ((data['calibration_query'], data['calibration_gallery']), (data['query'], data['gallery'])):
        if {row['id'] for row in query} != {row['id'] for row in gallery}:
            raise ValueError('Query/gallery identity sets differ')
        if {row['camera'] for row in query} & {row['camera'] for row in gallery}:
            raise ValueError('Query/gallery cameras overlap')
        if {row['path'] for row in query} & {row['path'] for row in gallery}:
            raise ValueError('Query/gallery images overlap')
    return data


def transform(size, train=False, normalization='transreid'):
    operations = [T.Resize((size, size))]
    if train:
        operations += [T.RandomHorizontalFlip(), T.Pad(10), T.RandomCrop((size, size)),
                       T.ColorJitter(.15, .15, .1, .02)]
    operations += [T.ToTensor()]
    mean, std = ([.5]*3, [.5]*3) if normalization == 'transreid' else ([.485, .456, .406], [.229, .224, .225])
    operations += [T.Normalize(mean, std)]
    if train:
        operations += [T.RandomErasing(p=.25, scale=(.02, .15), value=0)]
    return T.Compose(operations)


class Batches:
    def __init__(self, records, settings):
        self.grouped = defaultdict(lambda: defaultdict(list))
        for row in records:
            self.grouped[row['id']][row['camera']].append(row['path'])
        self.identities = sorted(self.grouped)
        if not 2 <= settings['training']['identities_per_batch'] <= len(self.identities):
            raise ValueError('identities_per_batch must be between two and the training identity count')
        self.mapping = {identity: label for label, identity in enumerate(self.identities)}
        if any(len(cameras) < 2 for cameras in self.grouped.values()):
            raise ValueError('Each training identity needs at least two cameras')
        self.settings = settings
        self.transform = transform(settings['model']['image_size'], train=True)

    def sample(self, rng):
        spec = self.settings['training']
        count = spec['images_per_identity']
        if count < 2 or count % 2:
            raise ValueError('images_per_identity must be even and at least two')
        images, labels = [], []
        for identity in rng.sample(self.identities, spec['identities_per_batch']):
            for camera in rng.sample(sorted(self.grouped[identity]), 2):
                paths = self.grouped[identity][camera]
                chosen = rng.sample(paths, count//2) if len(paths) >= count//2 else rng.choices(paths, k=count//2)
                for path in chosen:
                    with Image.open(path) as raw:
                        images.append(self.transform(raw.convert('RGB')))
                    labels.append(self.mapping[identity])
        return torch.stack(images), torch.tensor(labels)
