"""Fetch pinned tracker source and verified released propagation weights."""

import subprocess
import sys
from .common import ROOT, read, vendor


def run():
    from utils.teacher.assets import weights
    settings = read()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'utils/tracking/requirements.txt')], check=True)
    for name, repository in (('mcbyte', 'https://github.com/tstanczyk95/McBytePlusPlus.git'),
                             ('cutie', 'https://github.com/hkchengrex/Cutie.git')):
        location = ROOT / settings[f'{name}_source']
        if not location.exists():
            subprocess.run(['git', 'clone', repository, str(location)], check=True)
            subprocess.run(['git', '-C', str(location), 'checkout', settings[f'{name}_revision']], check=True)
        vendor(settings, name)
    releases = [
        ('edgetam', 'https://raw.githubusercontent.com/facebookresearch/EdgeTAM/main/checkpoints/edgetam.pt',
         56116523, 'ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df'),
        ('cutie', 'https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth',
         140443788, '9c05402ee36d3a356fb72715d263ba7e1ea06ad3bada48c1306491792da43023')]
    for name, url, size, sha in releases:
        weights({'pretrained': {'url': url, 'path': settings[f'{name}_weights'], 'bytes': size, 'sha256': sha}}, ROOT)
    print('Tracker source and released weights ready. Run `python benchmark.py trackers-check`.', flush=True)
