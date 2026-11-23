"""Isolate RF-DETR dependencies while sharing the existing large Torch install."""
import hashlib
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[3]


def prepare(spec):
    environment=(ROOT/spec['environment']).resolve()
    if environment in {(ROOT.parent/'.venv').resolve(),(ROOT.parent/'.teacher-env').resolve()}:
        raise ValueError('RF-DETR dependencies require their own environment')
    python=environment/'bin/python'
    if not python.exists():
        subprocess.run([sys.executable,'-m','venv',str(environment)],check=True)
    # Reuse modern Torch/CUDA binaries without reinstalling them. Locally
    # installed RF-DETR dependencies take precedence over shared packages.
    site=next(environment.glob('lib/python*/site-packages'))
    modern=ROOT.parent/'.venv/bin/python'
    if not modern.is_file():raise FileNotFoundError('Prepare the existing modern capstone/.venv first')
    shared=subprocess.check_output([str(modern),'-c',"import sysconfig;print(sysconfig.get_paths()['purelib'])"],text=True).strip()
    (site/'shared-modern-environment.pth').write_text(shared+'\n')
    subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check',
                    f'rfdetr=={spec["package_version"]}'],check=True)
    target=ROOT/spec['checkpoint']
    target.parent.mkdir(parents=True,exist_ok=True)
    if spec.get('checkpoint_format')=='finetuned':
        from utils.teacher.assets import sha256
        if not target.exists() or sha256(target)!=spec['sha256']:
            raise ValueError('Selected fine-tuned weights are missing or changed; do not replace them with COCO weights')
        return
    if target.exists():
        if hashlib.md5(target.read_bytes()).hexdigest()!=spec['md5']:
            raise ValueError('Existing RF-DETR weights differ from the pinned upstream asset')
        return
    import requests
    temporary=target.with_suffix('.partial')
    with requests.get(spec['url'],stream=True,timeout=(20,120)) as response:
        response.raise_for_status()
        with temporary.open('wb') as stream:
            for block in response.iter_content(1024*1024):stream.write(block)
    if hashlib.md5(temporary.read_bytes()).hexdigest()!=spec['md5']:
        temporary.unlink()
        raise ValueError('RF-DETR weight download checksum mismatch')
    temporary.replace(target)


if __name__=='__main__':
    prepare(yaml.safe_load((ROOT/'configs/detector_compare.yaml').read_text())['rfdetr'])
