"""Joint crop-embedding recovery on the same frozen aerial observations as TransReID."""
import argparse
import fcntl
from pathlib import Path
import os
import subprocess
import time
import yaml

ROOT = Path(__file__).resolve().parents[3]


def extract(config, selection, settings, checkpoint, device):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from utils.tracking.common import load, save
    from utils.teacher.assets import sha256
    from utils.teacher.model import CoDINOJoint
    digest = sha256(ROOT/config['output']/'selection.pt')
    model = None
    for sequence, case in selection['cases'].items():
        directory = ROOT/config['state_root']/'features'/sequence
        path = directory/'state.pt'
        state = load(path) if path.exists() else dict(signature=digest, parts=[], frames=0,
                    seconds=0., complete=False, device=str(device))
        if state['signature'] != digest:
            raise ValueError('Joint recovery inputs changed; use a new suite run name')
        if state['complete']:
            print(f'{sequence}: completed joint features reused', flush=True)
            continue
        if model is None:
            saved = load(checkpoint)
            count = len(saved['identity_to_label']) + len(saved['vrai_identity_to_label'])
            model = CoDINOJoint(settings, ROOT, count, pretrained=False).to(device)
            model.load_state_dict(saved['model'])
            model.eval().requires_grad_(False)
            del saved
        directory.mkdir(parents=True, exist_ok=True)
        pending = []
        started = time.perf_counter()
        for index in range(state['frames'], len(case['frames'])):
            row = case['frames'][index]
            outputs = []
            if row['observations']:
                with Image.open(row['path']) as image:
                    rgb = image.convert('RGB')
                for offset in range(0, len(row['observations']), 2):
                    images = [pil_to_tensor(rgb.crop(tuple(o['box']))).float().to(device)/255
                              for o in row['observations'][offset:offset+2]]
                    with torch.inference_mode(), torch.cuda.amp.autocast(
                        enabled=settings['training']['mixed_precision'] and device.type == 'cuda'
                    ):
                        outputs.append(F.normalize(model.crop_embeddings(images).float(), dim=1).cpu())
            dimension = settings['model']['embedding_dim']
            vectors = torch.cat(outputs) if outputs else torch.empty((0, dimension))
            if vectors.shape != (len(row['observations']), dimension) or not torch.isfinite(vectors).all():
                raise ValueError('Invalid joint recovery embeddings')
            pending.append(vectors)
            if len(pending) >= config['checkpoint_frames'] or index+1 == len(case['frames']):
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                part = directory/f'chunk_{len(state["parts"]):04d}.pt'
                save(pending, part)
                state['parts'].append(dict(file=part.name, frames=len(pending)))
                state.update(frames=index+1, complete=index+1 == len(case['frames']),
                             seconds=state['seconds']+time.perf_counter()-started)
                save(state, path)
                pending = []
                started = time.perf_counter()
                print(f'{sequence}: joint embedding frames {index+1}/{len(case["frames"])} committed', flush=True)


def worker(spec):
    import torch
    from utils.tracking.common import load, save
    from utils.teacher.assets import sha256
    from utils.experiments.vehicle_recovery.data import freeze
    from utils.experiments.vehicle_recovery.command import features, replay
    from utils.experiments.vehicle_recovery.report import report, render
    from .command import checkpoint_settings
    source = yaml.safe_load((ROOT/spec['source_config']).read_text())
    original = freeze(source)
    joint = yaml.safe_load((ROOT/spec['joint_config']).read_text())
    settings = checkpoint_settings(spec['joint_config'])
    last = load(ROOT/joint['last_checkpoint'])
    if not last['finished']:
        raise ValueError('Complete joint training before final recovery evaluation')
    run_id = last['run_id']
    del last
    best = load(ROOT/joint['checkpoint'])
    if best['run_id'] != run_id:
        raise ValueError('Joint best/last checkpoints belong to different runs')
    del best
    config = {**source, **spec}
    # Same observations, reliability gates, association and blackout schedule.
    signed = dict(cases=original['cases'], config=config,
        source_selection_sha256=sha256(ROOT/source['output']/'selection.pt'),
        checkpoint_sha256=sha256(ROOT/joint['checkpoint']),
        implementation={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
            ROOT/'utils/teacher/model.py', ROOT/'utils/teacher/data.py']},
        limitations=[*filter(lambda s: not s.startswith('GT vehicle boxes'), original['limitations']),
            'GT vehicle boxes replace discovery; joint soft-mask pooling runs on RGB crops',
            'Fixed specialist association thresholds are transferred unchanged, not calibrated for joint embeddings',
            'This crop-path diagnostic does not measure shared full-frame inference speed or cross-camera recovery'])
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output/'selection.pt'
    if selection_path.exists():
        if load(selection_path) != signed:
            raise ValueError('Joint recovery provenance changed; use a new suite run name')
    else:
        save(signed, selection_path)
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('device:auto selected CPU; the existing Co-DINO CUDA operators require a CUDA host')
    extract(config, signed, settings, ROOT/joint['checkpoint'], device)
    for sequence in signed['cases']:
        bank = features(config, sequence)
        for scenario in config['scenarios']:
            replay(config, signed, sequence, scenario, bank)
        del bank
    render(config, signed)
    report(config, signed)
    summary = output/'summary.md'
    summary.write_text(summary.read_text().replace('Original TransReID;', 'Shared joint crop encoder;').replace(
        'No detector, masks, training, threshold search or automatic promotion.',
        'GT boxes replace discovery; joint soft-mask pooling is active. No training, threshold search or automatic promotion.'))
    result = yaml.safe_load((output/'report.yaml').read_text())
    result.update(encoder='joint', checkpoint_sha256=signed['checkpoint_sha256'], limitations=signed['limitations'])
    (output/'report.yaml').write_text(yaml.safe_dump(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    spec = yaml.safe_load((ROOT/args.config).read_text())
    if spec['device'] != 'auto':
        raise ValueError('device must be auto')
    root = ROOT/spec['state_root']
    root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        with (root/'queue.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Joint recovery is already running') from None
            return worker(spec)
    config = yaml.safe_load((ROOT/'configs/teacher.yaml').read_text())
    python = (ROOT/config['environment']/'bin/python').resolve()
    subprocess.run([str(python), '-u', '-m', 'utils.experiments.joint_evaluation.recovery',
                    '--config', args.config, '--worker'], cwd=ROOT,
                   env={**os.environ, 'PYTHONUNBUFFERED': '1'}, check=True)


if __name__ == '__main__':
    main()
