"""Clean KITTI training and identity-disjoint VRAI sampling/calibration."""

import csv
import random
from pathlib import Path

from utils.full_frame.kitti import KittiMOTSCars
from utils.full_frame.multiview import VRAICrossViewBatches
from .settings import resolve


def rows(path):
    with Path(path).open(newline='') as file:
        reader = csv.DictReader(file)
        if not {'path', 'id', 'camera'} <= set(reader.fieldnames or ()):
            raise ValueError(f'Expected path,id,camera in {path}')
        return list(reader)


def build(settings, root):
    data = settings['data']
    kitti = KittiMOTSCars(resolve(root, data['kitti_root']), data['train_sequences'])
    vrai = VRAICrossViewBatches(*(resolve(root, data[key]) for key in ('vrai_train', 'vrai_query', 'vrai_gallery')))
    calibration = [rows(resolve(root, data[key])) for key in ('vrai_calibration_query', 'vrai_calibration_gallery')]
    final = [rows(resolve(root, data[key])) for key in ('vrai_query', 'vrai_gallery')]
    train_ids = set(vrai.identity_to_label)
    calibration_ids = {row['id'] for split in calibration for row in split}
    final_ids = {row['id'] for split in final for row in split}
    if train_ids & (calibration_ids | final_ids) or calibration_ids & final_ids:
        raise ValueError('VRAI train/calibration/final identities overlap')
    for query, gallery in (calibration, final):
        if {row['id'] for row in query} != {row['id'] for row in gallery}:
            raise ValueError('Query/gallery identity sets differ')
        if {row['path'] for row in query} & {row['path'] for row in gallery}:
            raise ValueError('Query/gallery image files overlap')
        if {row['camera'] for row in query} & {row['camera'] for row in gallery}:
            raise ValueError('Expected separate query and gallery cameras')
    limit = settings['training'].get('calibration_max_ids')
    if limit is not None and len(calibration_ids) > limit:
        chosen = set(random.Random(settings['training']['seed']).sample(sorted(calibration_ids), limit))
        calibration = [[row for row in split if row['id'] in chosen] for split in calibration]
    return kitti, vrai, calibration, final
