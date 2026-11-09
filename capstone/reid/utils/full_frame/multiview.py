"""Continue the KITTI teacher with VRAI cross-camera vehicle identities."""

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, RandomGrayscale
from torchvision.transforms.functional import hflip, pil_to_tensor

from .kitti import KittiMOTSCars
from .model import FullFrameVehicleNet, batch_hard_triplet
from .training import _pair_collate


def _path(config_path, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(config_path).parent / path


def _read_manifest(path):
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        if not {'path', 'id', 'camera'} <= set(reader.fieldnames or ()):
            raise ValueError(f'{path} needs path, id, camera columns')
        return list(reader)


class VRAICrossViewBatches:
    """Sample two camera views per physical car and several cars per step."""

    def __init__(self, train_manifest, query_manifest, gallery_manifest):
        rows = _read_manifest(train_manifest)
        query = _read_manifest(query_manifest)
        gallery = _read_manifest(gallery_manifest)
        if {row['id'] for row in query} != {row['id'] for row in gallery}:
            raise ValueError('VRAI query and gallery identity sets differ')
        heldout = {row['id'] for row in query + gallery}
        self.by_identity = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row['id'] in heldout:
                raise ValueError(f'VRAI training identity {row["id"]} is in held-out query/gallery')
            path = Path(row['path'])
            if not path.is_file():
                raise FileNotFoundError(path)
            self.by_identity[row['id']][row['camera']].append(path)
        self.identities = sorted(self.by_identity)
        if not self.identities or any(len(self.by_identity[identity]) < 2 for identity in self.identities):
            raise ValueError('Every VRAI training identity needs at least two cameras')
        self.identity_to_label = {identity: index for index, identity in enumerate(self.identities)}
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04)
        self.grayscale = RandomGrayscale(p=0.05)

    def sample(self, rng, identities_per_batch, augment=True):
        if identities_per_batch > len(self.identities) or identities_per_batch < 2:
            raise ValueError('identities_per_batch must be between 2 and the VRAI training ID count')
        images, labels = [], []
        for identity in rng.sample(self.identities, identities_per_batch):
            cameras = rng.sample(sorted(self.by_identity[identity]), 2)
            for camera in cameras:
                path = rng.choice(self.by_identity[identity][camera])
                with Image.open(path) as raw:
                    image = raw.convert('RGB')
                    if augment:
                        image = self.grayscale(self.jitter(image))
                        if rng.random() < 0.5:
                            image = hflip(image)
                    images.append(pil_to_tensor(image).float() / 255)
                labels.append(self.identity_to_label[identity])
        return images, torch.tensor(labels, dtype=torch.long)


def _save(checkpoint, model, optimizer, kitti, vrai, settings, epoch, step, rng):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + '.partial')
    torch.save({'epoch': epoch, 'step': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'identity_to_label': kitti.identity_to_label,
                'vrai_identity_to_label': vrai.identity_to_label,
                'settings': settings, 'vrai_rng': rng.getstate(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, temporary)
    temporary.replace(checkpoint)
    print(f'Saved {checkpoint} (completed epochs {epoch}, current step {step})', flush=True)


def train(config_path, check_only=False):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    settings = config['full_frame']
    extra = settings['multiview']
    if set(settings['train_sequences']) & set(settings['val_sequences']):
        raise ValueError('KITTI train and validation sequences overlap')
    kitti = KittiMOTSCars(_path(config_path, settings['data_root']), settings['train_sequences'],
                          max_pairs=settings.get('max_pairs'))
    manifest = config['vrai']['vehicle_benchmark']
    vrai = VRAICrossViewBatches(*(_path(config_path, manifest[key]) for key in
                                 ('train_manifest', 'query_manifest', 'gallery_manifest')))
    if extra['identities_per_batch'] < 2 or extra['kitti_steps_per_vrai_step'] < 1:
        raise ValueError('VRAI needs at least two identities and a positive KITTI/VRAI step ratio')
    if extra['save_every_steps'] < 1:
        raise ValueError('save_every_steps must be positive')
    teacher_path = _path(config_path, settings['checkpoint'])
    output = _path(config_path, extra['checkpoint'])
    if teacher_path.resolve() == output.resolve():
        raise ValueError('Multiview checkpoint must be separate from the KITTI teacher checkpoint')
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False, mmap=True)
    if teacher['identity_to_label'] != kitti.identity_to_label:
        raise ValueError('KITTI identity mapping changed since teacher training')
    if teacher['settings']['architecture'] != settings['architecture'] or teacher['settings']['image_size'] != settings['image_size']:
        raise ValueError('Teacher architecture or image size differs from config')
    kitti_count = len(kitti.identity_to_label)
    torch.manual_seed(settings['seed'])
    model = FullFrameVehicleNet(num_identities=kitti_count + len(vrai.identity_to_label),
                                pretrained=False, image_size=settings['image_size'],
                                architecture=settings['architecture'],
                                crop_image_size=extra['crop_image_size'])
    old = teacher['model']
    shared = {key: value for key, value in old.items() if not key.startswith('id_classifier.')}
    missing, unexpected = model.load_state_dict(shared, strict=False)
    if set(missing) != {'id_classifier.weight', 'id_classifier.bias'} or unexpected:
        raise ValueError(f'Unexpected teacher/model mismatch: missing={missing}, unexpected={unexpected}')
    with torch.no_grad():
        model.id_classifier.weight[:kitti_count].copy_(old['id_classifier.weight'])
        model.id_classifier.bias[:kitti_count].copy_(old['id_classifier.bias'])
    del teacher, old, shared
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=extra['learning_rate'], weight_decay=1e-4)
    start_epoch = start_step = 0
    resumed = None
    if output.is_file() and not check_only:
        resumed = torch.load(output, map_location='cpu', weights_only=False)
        if resumed['identity_to_label'] != kitti.identity_to_label or resumed['vrai_identity_to_label'] != vrai.identity_to_label:
            raise ValueError('Identity mappings differ from multiview checkpoint')
        for key in ('crop_image_size', 'identities_per_batch', 'kitti_steps_per_vrai_step'):
            if resumed['settings']['multiview'][key] != extra[key]:
                raise ValueError(f'{key} differs from multiview checkpoint')
        model.load_state_dict(resumed['model'])
        optimizer.load_state_dict(resumed['optimizer'])
        start_epoch, start_step = resumed['epoch'], resumed['step']
        print(f'Resuming epoch {start_epoch + 1}, KITTI step {start_step + 1}', flush=True)
    print(f'KITTI: {len(kitti)} frame pairs / {kitti_count} IDs; '
          f'VRAI: {len(vrai.identities)} two-camera training IDs; device={device}', flush=True)

    def vrai_step(rng, augment=True):
        images, local_labels = vrai.sample(rng, extra['identities_per_batch'], augment=augment)
        images = [image.to(device) for image in images]
        labels = local_labels.to(device) + kitti_count
        embeddings = model.crop_embeddings(images)
        losses = {'vrai_id': F.cross_entropy(model.id_classifier(embeddings), labels),
                  'vrai_triplet': batch_hard_triplet(embeddings, labels)}
        return losses

    if check_only:
        model.train()
        images, targets = _pair_collate([kitti[0]])
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        with torch.no_grad():
            kitti_losses = model(images, targets)
        if not all(torch.isfinite(value) for value in kitti_losses.values()):
            raise RuntimeError('Non-finite KITTI loss')
        del images, targets, kitti_losses
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        losses = vrai_step(random.Random(settings['seed']), augment=False)
        loss = sum(losses.values())
        loss.backward()
        if (not torch.isfinite(loss) or model.embed_head.weight.grad is None
                or model.detector.roi_heads.mask_predictor.mask_fcn_logits.weight.grad is None):
            raise RuntimeError('VRAI embedding/mask gradient check failed')
        print('Multiview check passed:', {name: round(value.item(), 4) for name, value in losses.items()}, flush=True)
        return

    for epoch in range(start_epoch, extra['epochs']):
        generator = torch.Generator().manual_seed(settings['seed'] + epoch)
        loader = DataLoader(kitti, batch_size=settings['batch_pairs'], shuffle=True,
                            generator=generator, num_workers=settings['workers'],
                            collate_fn=_pair_collate)
        rng = random.Random(settings['seed'] + 10000 + epoch)
        if resumed is not None and epoch == start_epoch and start_step:
            rng.setstate(resumed['vrai_rng'])
            torch.set_rng_state(resumed['torch_rng'].byte())
            if device.type == 'cuda' and resumed['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all([state.byte() for state in resumed['cuda_rng']])
        model.train()
        for step, (images, targets) in enumerate(loader, 1):
            if epoch == start_epoch and step <= start_step:
                continue
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            kitti_losses = model(images, targets)
            kitti_loss = sum(kitti_losses.values())
            if not torch.isfinite(kitti_loss):
                raise RuntimeError(f'Non-finite KITTI loss at epoch {epoch+1} step {step}')
            kitti_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if step % extra['kitti_steps_per_vrai_step'] == 0:
                optimizer.zero_grad(set_to_none=True)
                losses = vrai_step(rng)
                loss = extra['vrai_loss_weight'] * sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite VRAI loss at epoch {epoch+1} step {step}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            if step % 50 == 0 or step == len(loader):
                print(f'Epoch {epoch+1}/{extra["epochs"]} KITTI {step}/{len(loader)} '
                      f'loss={kitti_loss.item():.3f}' +
                      (f' VRAI loss={loss.item():.3f}' if step % extra['kitti_steps_per_vrai_step'] == 0 else ''),
                      flush=True)
            if step % extra['save_every_steps'] == 0:
                _save(output, model, optimizer, kitti, vrai, settings, epoch, step, rng)
        _save(output, model, optimizer, kitti, vrai, settings, epoch + 1, 0, rng)
        start_step = 0
        resumed = None
