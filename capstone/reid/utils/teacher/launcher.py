"""Keep the simple root commands while isolating the teacher's legacy stack."""

import os
import shutil
import subprocess
from .settings import read, resolve


def setup():
    from .assets import prepare
    settings, root = read()
    environment = resolve(root, settings['environment'])
    python = environment / 'bin/python'
    if not python.is_file():
        conda = shutil.which('conda')
        if not conda:
            raise RuntimeError('Miniconda is needed to create the isolated Python 3.10 teacher environment')
        subprocess.run([conda, 'create', '--prefix', str(environment), 'python=3.10', 'pip', '-y',
                        '--override-channels', '-c', 'conda-forge'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', 'setuptools==69.5.1', 'numpy==1.26.4',
                    'torch==1.13.1+cu117', 'torchvision==0.14.1+cu117',
                    '--extra-index-url', 'https://download.pytorch.org/whl/cu117'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(root / 'utils/teacher/requirements.txt'),
                    '--no-build-isolation', '-f', 'https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html'], check=True)
    prepare()
    launch('check')


def launch(mode):
    settings, root = read()
    python = resolve(root, settings['environment']) / 'bin/python'
    if not python.is_file():
        raise FileNotFoundError('Run `python train.py teacher-setup` first')
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    subprocess.run([str(python), '-m', 'utils.teacher.runtime', mode], cwd=root,
                   env=environment, check=True)
