"""Teacher experiment paths and train/evaluation identity separation."""

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/teacher.yaml'


def read(config_path=CONFIG):
    config_path = Path(config_path)
    settings = yaml.safe_load(config_path.read_text())
    root = (config_path.parent / settings.get('project_root', '.')).resolve()
    from utils.full_frame.protocol import TRAIN, VALIDATION
    if (sorted(settings['data']['train_sequences']) != list(TRAIN)
            or sorted(settings['data']['val_sequences']) != list(VALIDATION)):
        raise ValueError('Teacher must use the conventional KITTI MOTS split')
    return settings, root


def resolve(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()
