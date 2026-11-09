"""Prepare the pinned official vehicle ReID code and authored vehicle weights."""

from pathlib import Path
import subprocess
import sys
from utils.teacher.assets import sha256
from utils.tracking.common import ROOT
from .data import read


def run():
    settings = read()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'utils/reid_teacher/requirements.txt')], check=True)
    source = ROOT / settings['source']['directory']
    if not source.is_dir():
        subprocess.run(['git', 'clone', settings['source']['repository'], str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', settings['source']['revision']], check=True)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != settings['source']['revision']:
        raise ValueError('Unexpected TransReID source revision')
    path = ROOT / settings['pretrained']['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        import gdown
        temporary = path.with_suffix(path.suffix + '.partial')
        result = gdown.download(id=settings['pretrained']['google_drive_id'], output=str(temporary), resume=True)
        if not result or sha256(temporary) != settings['pretrained']['sha256']:
            raise ValueError('Official vehicle checkpoint download failed verification')
        temporary.replace(path)
    if sha256(path) != settings['pretrained']['sha256']:
        raise ValueError('Vehicle pretrained checkpoint hash mismatch')
    for key in ('compact', 'msinet_source', 'msinet_weights'):
        if not (ROOT / settings['references'][key]).exists():
            raise FileNotFoundError(f'Reference missing: {settings["references"][key]}')
    print('Vehicle-trained TransReID source and weights verified. Run `python train.py reid-teacher-check`.', flush=True)
