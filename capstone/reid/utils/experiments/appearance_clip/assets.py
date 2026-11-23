"""Checksum-verified CLIP initialization stored inside the workspace."""
from pathlib import Path
import urllib.request
import yaml
from utils.tracking.common import ROOT
from utils.teacher.assets import sha256


def prepare(experiment):
    spec=experiment['clip'];path=ROOT/spec['weights']
    if path.exists():
        if sha256(path)!=spec['sha256']:raise ValueError('Existing CLIP initialization hash differs')
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.partial')
    offset=partial.stat().st_size if partial.exists() else 0
    request=urllib.request.Request(spec['url'],headers={'Range':f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request,timeout=60) as response:
        resumed=offset>0 and response.status==206
        with partial.open('ab' if resumed else 'wb') as file:
            received=offset if resumed else 0
            while block:=response.read(1024*1024):
                file.write(block);received+=len(block)
                if received//(50*1024*1024)!=(received-len(block))//(50*1024*1024):
                    print(f'CLIP initialization: {received/2**20:.0f} MiB received',flush=True)
    if sha256(partial)!=spec['sha256']:raise ValueError('Downloaded CLIP initialization checksum mismatch')
    partial.replace(path)
    print(f'CLIP initialization verified: {path}',flush=True)
    return path


if __name__=='__main__':prepare(yaml.safe_load((ROOT/'configs/appearance_training.yaml').read_text()))
