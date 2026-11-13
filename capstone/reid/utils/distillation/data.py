"""Aligned augmented student crops and clean, cached specialist identity targets."""

from PIL import Image
import torch
from torchvision.transforms.functional import hflip, pil_to_tensor


def sample(vrai, indices, targets, rng, count):
    images, labels, teacher = [], [], []
    for identity in rng.sample(vrai.identities, count):
        for camera in rng.sample(sorted(vrai.by_identity[identity]), 2):
            path = rng.choice(vrai.by_identity[identity][camera])
            with Image.open(path) as raw:
                image = vrai.grayscale(vrai.jitter(raw.convert('RGB')))
                if rng.random() < .5:
                    image = hflip(image)
                images.append(pil_to_tensor(image).float() / 255)
            labels.append(vrai.identity_to_label[identity])
            teacher.append(targets[indices[str(path.resolve())]])
    return images, torch.tensor(labels), torch.stack(teacher).float()


def audit(vrai, indices, check=False):
    expected = {str(path.resolve()) for cameras in vrai.by_identity.values() for paths in cameras.values() for path in paths}
    if not set(indices) <= expected or (not check and set(indices) != expected):
        raise ValueError('Specialist targets must cover exactly the joint VRAI training images')
    if check:
        vrai.identities = [identity for identity in vrai.identities
                           if all(str(path.resolve()) in indices for paths in vrai.by_identity[identity].values() for path in paths)]
        if len(vrai.identities) < 2:
            raise ValueError('Check cache needs complete multi-camera training identities')
