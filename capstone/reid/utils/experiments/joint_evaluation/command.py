"""Evaluate the existing square-crop joint checkpoint; never train or select on test."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def evaluate_selected(config_path=None):
    # Preserve the exact saved training/model settings and existing KITTI cache.
    # The evaluator checks training completion and matching best/last run IDs.
    from utils.teacher.evaluation import run
    if config_path:
        settings = checkpoint_settings(config_path)
    else:
        from utils.experiments.crop_training.settings import read
        settings, _ = read('square_256')
    run(experiment=settings, calibration_only=False)
    if config_path:
        import yaml
        from utils.benchmarking.suite.scorecard import joint_protocol
        output = ROOT/settings['tracking']['metrics_root']
        (output.parent/'evaluation_protocol.yaml').write_text(yaml.safe_dump(joint_protocol(settings)))


def checkpoint_settings(config_path):
    """Use the checkpoint's training recipe; isolate evaluation output paths."""
    from copy import deepcopy
    import torch
    import yaml
    spec = yaml.safe_load((ROOT/config_path).read_text())
    if spec['device'] != 'auto':
        raise ValueError('device must be auto')
    saved = torch.load(ROOT/spec['checkpoint'], map_location='cpu', weights_only=False)
    settings = deepcopy(saved['settings'])
    del saved
    settings['training'].update(best_checkpoint=spec['checkpoint'], checkpoint=spec['last_checkpoint'])
    settings['tracking'].update(name=spec['tracking_name'], predictions_root=spec['predictions_root'],
                                metrics_root=spec['output']+'/kitti_mots')
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Suite evaluation config; saved checkpoint settings define the model')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return evaluate_selected(args.config)

    import yaml
    config = yaml.safe_load((ROOT/'configs/teacher.yaml').read_text())
    python = (ROOT/config['environment']/'bin/python').resolve()
    if not python.is_file():
        raise FileNotFoundError(f'Existing joint environment is missing: {python}')
    spec = yaml.safe_load((ROOT/args.config).read_text()) if args.config else None
    paths = [ROOT/spec[key] for key in ('checkpoint', 'last_checkpoint')] if spec else [
        ROOT/'checkpoints/crop_training'/name for name in ('square_256.best.pt', 'square_256.last.pt')]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Complete the existing square-crop run first: {path}')
    print('Evaluating the configured joint checkpoint: full held-out VRAI '
          '(mask/global pooling) and KITTI validation. No training. '
          'Existing calibration scores are retained; completed KITTI sequences are reused.', flush=True)
    command = [str(python), '-m', 'utils.experiments.joint_evaluation.command', '--worker']
    if args.config:
        command += ['--config', args.config]
    subprocess.run(command,
                   cwd=ROOT, env={**os.environ, 'PYTHONUNBUFFERED': '1'}, check=True)
    if not args.config:
        from utils.reporting.milestones import run
        run()


if __name__ == '__main__':
    main()
