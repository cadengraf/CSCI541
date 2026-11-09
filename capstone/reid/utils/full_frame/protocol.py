"""Paths and leakage guards for the conventional KITTI MOTS validation split."""

from pathlib import Path
import yaml

CONFIG = Path(__file__).resolve().parents[2] / 'configs/kitti_mots.yaml'
TRAIN = (0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20)
VALIDATION = (2, 6, 7, 8, 10, 13, 14, 16, 18)
PROTOCOL = 'kitti_mots_standard_val_v1'


def project_root(config_path, config):
    return (Path(config_path).parent / config.get('project_root', '.')).resolve()


def load(config_path=CONFIG):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    check_settings(config['full_frame'])
    return config, project_root(config_path, config)


def check_settings(settings):
    if (settings.get('protocol') != PROTOCOL
            or sorted(settings['train_sequences']) != list(TRAIN)
            or sorted(settings['val_sequences']) != list(VALIDATION)):
        raise ValueError('This benchmark requires the conventional 12/9 KITTI MOTS split')
    if settings.get('max_pairs') is not None:
        raise ValueError('The research benchmark must use the entire training partition')


def check_checkpoint(saved):
    check_settings(saved['settings'])
    if saved.get('step', 0) or saved['epoch'] < saved['settings']['epochs']:
        raise ValueError('Training is incomplete; finish the clean training run before benchmarking')


def train(check_only=False):
    from .training import check_dataset, train as run
    config, _ = load()
    if check_only:
        return check_dataset(CONFIG)
    run(CONFIG)
    if config.get('evaluate_after_train', True):
        from .mots_benchmark import run as benchmark
        return benchmark()
