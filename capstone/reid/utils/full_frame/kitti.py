"""KITTI MOTS car frames and compressed COCO-RLE instance labels."""

from collections import defaultdict
from pathlib import Path
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def decode_rle(encoded, height, width):
    """Decode the compressed COCO RLE stored in KITTI instances_txt."""
    counts = []
    pos = 0
    while pos < len(encoded):
        value = 0
        shift = 0
        while True:
            code = ord(encoded[pos]) - 48
            pos += 1
            value |= (code & 0x1f) << shift
            shift += 5
            if not code & 0x20:
                if code & 0x10:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError('Negative COCO RLE run length')
        counts.append(value)
    if sum(counts) != height * width:
        raise ValueError(f'RLE pixel count {sum(counts)} != {height}x{width}')
    pixels = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    foreground = False
    for length in counts:
        if foreground:
            pixels[cursor:cursor + length] = 1
        cursor += length
        foreground = not foreground
    return pixels.reshape((height, width), order='F')


def read_car_annotations(annotation_dir, sequences):
    frames = defaultdict(list)
    for sequence in sequences:
        path = Path(annotation_dir) / f'{sequence}.txt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing KITTI MOTS annotations: {path}')
        for line in path.read_text().splitlines():
            parts = line.split(' ', 5)
            if len(parts) != 6:
                raise ValueError(f'Malformed KITTI annotation in {path}: {line[:80]}')
            frame, track, category, height, width = map(int, parts[:5])
            if category == 1:  # KITTI MOTS: car=1, pedestrian=2, ignore=10
                frames[(sequence, frame)].append((track, height, width, parts[5]))
    return frames


class KittiMOTSCars(Dataset):
    """Return consecutive frame pairs so identity positives occur in a batch."""

    def __init__(self, root, sequences, max_pairs=None):
        self.root = Path(root)
        self.sequences = tuple(f'{int(s):04d}' for s in sequences)
        self.annotations = read_car_annotations(self.root / 'instances_txt', self.sequences)
        self.identity_to_label = {
            (sequence, track): index
            for index, (sequence, track) in enumerate(sorted({
                (sequence, record[0])
                for (sequence, _), records in self.annotations.items()
                for record in records
            }))
        }
        self.pairs = []
        for sequence in self.sequences:
            folder = self.root / 'training' / 'image_02' / sequence
            if not folder.is_dir():
                raise FileNotFoundError(f'Missing KITTI MOTS images: {folder}')
            frames = sorted(int(path.stem) for path in folder.glob('*.png'))
            self.pairs += [((sequence, first), (sequence, second))
                           for first, second in zip(frames, frames[1:])
                           if second == first + 1 and (sequence, first) in self.annotations
                           and (sequence, second) in self.annotations]
        if max_pairs is not None:
            self.pairs = self.pairs[:max_pairs]
        if not self.pairs:
            raise ValueError('No consecutive labeled car frames found in KITTI MOTS')

    def __len__(self):
        return len(self.pairs)

    def _frame(self, sequence, frame):
        path = self.root / 'training' / 'image_02' / sequence / f'{frame:06d}.png'
        with Image.open(path) as raw:
            image = pil_to_tensor(raw.convert('RGB')).float() / 255
        height, width = image.shape[-2:]
        boxes, masks, tracks, identity_labels = [], [], [], []
        for track, annotated_height, annotated_width, encoded in self.annotations.get((sequence, frame), []):
            if (annotated_height, annotated_width) != (height, width):
                raise ValueError(f'Image/annotation size mismatch: {path}')
            mask = decode_rle(encoded, height, width)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            masks.append(torch.from_numpy(mask.copy()))
            tracks.append(track)
            identity_labels.append(self.identity_to_label[(sequence, track)])
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.ones(len(boxes), dtype=torch.int64),
            'masks': torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.uint8),
            'track_ids': torch.tensor(tracks, dtype=torch.int64),
            'identity_labels': torch.tensor(identity_labels, dtype=torch.int64),
            'image_id': torch.tensor(int(sequence) * 1000000 + frame),
        }
        return image, target

    def __getitem__(self, index):
        first, second = self.pairs[index]
        return self._frame(*first), self._frame(*second)


def unpack_kitti(root, images_archive, labels_archive):
    """Copy complete ZIP members into data/; safe to resume after interruption."""
    root = Path(root)
    for archive_path, marker in ((Path(images_archive).expanduser(), 'training/image_02/'),
                                 (Path(labels_archive).expanduser(), 'instances_txt/')):
        if not zipfile.is_zipfile(archive_path):
            raise FileNotFoundError(f'Complete ZIP archive is required: {archive_path}')
        with zipfile.ZipFile(archive_path) as archive:
            members = [item for item in archive.infolist()
                       if not item.is_dir() and marker in item.filename.replace('\\', '/')]
            if not members:
                raise ValueError(f'{archive_path} contains no {marker} files')
            copied = 0
            for item in members:
                name = item.filename.replace('\\', '/')
                relative = name[name.index(marker):]
                destination = (root / relative).resolve()
                if root.resolve() not in destination.parents:
                    raise ValueError(f'Unsafe archive entry: {name}')
                if destination.is_file() and destination.stat().st_size == item.file_size:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + '.partial')
                with archive.open(item) as source, temporary.open('wb') as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(destination)
                copied += 1
                if copied % 100 == 0:
                    print(f'Copied {copied} files from {archive_path.name}', flush=True)
            print(f'{archive_path.name}: copied {copied}, skipped {len(members)-copied} complete files', flush=True)
