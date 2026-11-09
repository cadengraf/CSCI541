"""Train the shared full-frame KITTI MOTS car model."""

from pathlib import Path
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .kitti import KittiMOTSCars
from .model import FullFrameVehicleNet
from .protocol import project_root


def _pair_collate(batch):
    frames = [frame for pair in batch for frame in pair]
    return [frame[0] for frame in frames], [frame[1] for frame in frames]


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def smoke_test():
    """Exercise box, mask, ReID losses and inference without the dataset."""
    torch.manual_seed(7)
    model = FullFrameVehicleNet(num_identities=2, pretrained=False, image_size=128)
    images = [torch.rand(3, 128, 128) for _ in range(2)]
    masks = torch.zeros((2, 128, 128), dtype=torch.uint8)
    masks[0, 25:70, 20:65] = 1
    masks[1, 50:100, 55:110] = 1
    target = {
        'boxes': torch.tensor([[20., 25., 65., 70.], [55., 50., 110., 100.]]),
        'labels': torch.ones(2, dtype=torch.int64),
        'masks': masks,
        'track_ids': torch.tensor([1, 2]),
        'identity_labels': torch.tensor([0, 1]),
    }
    model.train()
    losses = model(images, [target, target])
    total = sum(losses.values())
    total.backward()
    if not torch.isfinite(total):
        raise RuntimeError('Non-finite full-frame training loss')
    print('Training losses:', {key: round(value.item(), 4) for key, value in losses.items()})
    model.eval()
    with torch.no_grad():
        predictions = model(images[:1])
    result = predictions[0]
    assert set(('boxes', 'scores', 'masks', 'embeddings')) <= result.keys()
    assert result['embeddings'].shape == (len(result['boxes']), 128)
    print('Inference outputs:', {key: tuple(result[key].shape)
                                 for key in ('boxes', 'scores', 'masks', 'embeddings')})


def check_dataset(config_path):
    """Check a real KITTI pair and all model losses before a long run."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    settings = config['full_frame']
    base = project_root(config_path, config)
    root = Path(settings['data_root']).expanduser()
    if not root.is_absolute():
        root = base / root
    if set(settings['train_sequences']) & set(settings['val_sequences']):
        raise ValueError('KITTI train and validation sequences overlap')
    train_data = KittiMOTSCars(root, settings['train_sequences'])
    val_data = KittiMOTSCars(root, settings['val_sequences'])
    images, targets = _pair_collate([train_data[0]])
    print(f'KITTI check: {len(train_data)} train pairs / {len(val_data)} validation pairs; '
          f'{len(train_data.identity_to_label)} train IDs; '
          f'first pair has {[len(t["boxes"]) for t in targets]} cars', flush=True)
    model = FullFrameVehicleNet(num_identities=len(train_data.identity_to_label),
                                pretrained=False, image_size=settings['image_size'],
                                architecture=settings['architecture'])
    model.train()
    with torch.no_grad():
        losses = model(images, targets)
    if not all(torch.isfinite(loss).item() for loss in losses.values()):
        raise RuntimeError(f'Non-finite losses on KITTI pair: {losses}')
    print('Real KITTI pair passed:', {key: round(value.item(), 4)
                                      for key, value in losses.items()}, flush=True)


def train(config_path):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    settings = config['full_frame']
    base = project_root(config_path, config)
    root = Path(settings['data_root']).expanduser()
    if not root.is_absolute():
        root = base / root
    train_sequences = settings['train_sequences']
    val_sequences = settings['val_sequences']
    if set(train_sequences) & set(val_sequences):
        raise ValueError('KITTI train and validation sequences overlap')
    dataset = KittiMOTSCars(root, train_sequences, max_pairs=settings.get('max_pairs'))
    device = _device()
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    model = FullFrameVehicleNet(num_identities=len(dataset.identity_to_label),
                                pretrained=settings['pretrained'],
                                image_size=settings['image_size'],
                                architecture=settings['architecture']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=1e-4)
    checkpoint = Path(settings['checkpoint']).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = base / checkpoint
    start_epoch = 0
    start_step = 0
    saved = None
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['identity_to_label'] != dataset.identity_to_label:
            raise ValueError('Checkpoint identity mapping differs from current KITTI split')
        if saved['settings']['architecture'] != settings['architecture']:
            raise ValueError('Checkpoint architecture differs from config.yaml')
        for key in ('train_sequences', 'val_sequences', 'image_size', 'protocol'):
            if saved['settings'].get(key) != settings.get(key):
                raise ValueError(f'Checkpoint {key} differs from the selected config')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch = saved['epoch']
        start_step = saved.get('step', 0)
        print(f'Resuming epoch {start_epoch + 1} from {checkpoint}', flush=True)
    print(f'KITTI cars: {len(dataset.pairs)} adjacent frame pairs, '
          f'{len(dataset.identity_to_label)} track identities; device={device}', flush=True)

    def save(epoch, step):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temp = checkpoint.with_suffix(checkpoint.suffix + '.partial')
        torch.save({'epoch': epoch, 'step': step, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'identity_to_label': dataset.identity_to_label,
                    'settings': settings, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}, temp)
        temp.replace(checkpoint)
        print(f'Saved {checkpoint} (completed epochs {epoch}, step {step})', flush=True)

    for epoch in range(start_epoch, settings['epochs']):
        loader = DataLoader(dataset, batch_size=settings['batch_pairs'], shuffle=True,
                            generator=torch.Generator().manual_seed(settings['seed'] + epoch),
                            num_workers=settings['workers'], collate_fn=_pair_collate)
        if epoch == start_epoch and start_step and saved is not None:
            torch.set_rng_state(saved['torch_rng'].byte())
            if device.type == 'cuda' and saved.get('cuda_rng'):
                torch.cuda.set_rng_state_all([state.byte() for state in saved['cuda_rng']])
        model.train()
        total_loss = 0.0
        steps_seen = 0
        for step, (images, targets) in enumerate(loader, 1):
            if epoch == start_epoch and step <= start_step:
                continue
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss on epoch {epoch+1}, step {step}: {losses}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            steps_seen += 1
            if step % 20 == 0 or step == len(loader):
                print(f'Epoch {epoch+1}/{settings["epochs"]} step {step}/{len(loader)} '
                      f'loss={total_loss/steps_seen:.4f}', flush=True)
            if settings.get('save_every_steps') and step % settings['save_every_steps'] == 0:
                save(epoch, step)
        save(epoch + 1, 0)
        start_step = 0
