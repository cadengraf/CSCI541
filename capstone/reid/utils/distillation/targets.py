"""Cache frozen specialist embeddings for training images only, across environments."""

import random
import torch
import yaml
from PIL import Image
from utils.teacher.assets import sha256
from .settings import read


def expected(settings, root):
    return {'teacher_sha256': settings['distillation']['teacher_sha256'],
            'train_manifest_sha256': sha256(root / settings['data']['vrai_train']),
            'dimensions': settings['distillation']['teacher_dimensions'],
            'chunk_size': settings['distillation']['cache_chunk_size'], 'format': 1}


def prepare(check=False):
    from utils.reid_teacher.evaluation import load_selected
    from utils.reid_teacher.data import partitions, read as specialist_settings
    from utils.tracking.common import save
    settings, root = read()
    spec = settings['distillation']
    cache = root / spec['cache']
    metadata = expected(settings, root)
    data = partitions(specialist_settings())
    if sha256(root / settings['data']['vrai_train']) != sha256(root / specialist_settings()['data']['train']):
        raise ValueError('Joint and specialist training manifests differ')
    records = data['train']
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / 'source.yaml'
    if manifest.is_file() and yaml.safe_load(manifest.read_text()) != metadata:
        raise ValueError('Teacher cache belongs to other weights/data/settings; choose a new cache path')
    manifest.write_text(yaml.safe_dump(metadata))
    teacher, conversion, selected = load_selected(selection_path=root / spec['teacher_selection'])
    if selected['candidate']['checkpoint_sha256'] != spec['teacher_sha256']:
        raise ValueError('Selected specialist differs from the pinned distillation teacher')
    # A check must cover randomly sampled identities, not merely the first rows.
    if check:
        identities = sorted({row['id'] for row in records})
        selected_ids = set(random.Random(settings['training']['seed']).sample(identities, 2 * settings['training']['identities_per_batch']))
        filename = cache / 'check.pt'
        chosen = [row for row in records if row['id'] in selected_ids]
        batches = [(filename, chosen)]
    else:
        size = spec['cache_chunk_size']
        batches = [(cache / f'{start:06d}.pt', records[start:start+size]) for start in range(0, len(records), size)]
    for filename, chosen in batches:
        if filename.is_file() and not check:
            existing = torch.load(filename, map_location='cpu', weights_only=True)
            if existing['paths'] != [row['path'] for row in chosen] or existing['targets'].shape != (len(chosen), spec['teacher_dimensions']):
                raise ValueError(f'Invalid teacher cache chunk {filename}')
            continue
        vectors = []
        for start in range(0, len(chosen), spec['cache_batch_size']):
            images = []
            for row in chosen[start:start+spec['cache_batch_size']]:
                with Image.open(row['path']) as raw:
                    images.append(conversion(raw.convert('RGB')))
            with torch.no_grad():
                target = teacher(torch.stack(images).cuda())
            if target.shape[1] != spec['teacher_dimensions'] or not torch.isfinite(target).all():
                raise ValueError('Invalid specialist targets')
            vectors.append(target.half().cpu())
        save({'paths': [row['path'] for row in chosen], 'targets': torch.cat(vectors)}, filename)
        print(f'Cached specialist targets: {filename.name}, {len(chosen)} training images', flush=True)
    if not check:
        (cache / 'complete.yaml').write_text(yaml.safe_dump({**metadata, 'images': len(records),
            'chunks': {path.name: sha256(path) for path, _ in batches}}))
    print('Specialist training targets ready', flush=True)


def load(settings, root, check=False):
    """Legacy Torch reader: no specialist network or modern dependencies required."""
    cache = root / settings['distillation']['cache']
    if yaml.safe_load((cache / 'source.yaml').read_text()) != expected(settings, root):
        raise ValueError('Teacher target cache provenance differs')
    if check:
        files = [cache / 'check.pt']
    else:
        complete = yaml.safe_load((cache / 'complete.yaml').read_text())
        if any(complete[key] != value for key, value in expected(settings, root).items()):
            raise ValueError('Incomplete or incompatible teacher targets')
        files = [cache / name for name in complete['chunks']]
        if any(sha256(path) != complete['chunks'][path.name] for path in files):
            raise ValueError('Teacher target cache is corrupt')
    paths, vectors = [], []
    for path in files:
        saved = torch.load(path, map_location='cpu')
        paths.extend(saved['paths']); vectors.append(saved['targets'])
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate paths in teacher target cache')
    return {path: index for index, path in enumerate(paths)}, torch.cat(vectors)
