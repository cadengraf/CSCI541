"""Reuse the frozen temporal protocol; keep source calibration separate."""
import csv
import hashlib
from pathlib import Path
import subprocess
from PIL import Image
import yaml
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256
from utils.experiments.aerial_transfer.data import crop


def image(row):
    if 'box' in row:return crop(row)
    with Image.open(row['path']) as value:return value.convert('RGB')


def manifest(path):
    with (ROOT/path).open() as file:return list(csv.DictReader(file))


def signature(config):
    source=load(ROOT/config['transfer_selection'])
    for path,digest in source['implementations'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Original transfer implementation changed')
    for path,digest in source['audit']['annotations'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Original transfer annotations changed')
    data={f'calibration_{group}':manifest(config[f'calibration_{group}']) for group in ['query','gallery']}
    data.update({f'target_{group}':rows for group,rows in source['data'].items()})
    audit=yaml.safe_load((ROOT/config['prepared_audit']).read_text())
    for group in ['query','gallery']:
        path=config[f'calibration_{group}']
        if sha256(ROOT/path)!=audit['prepared_manifests'][Path(path).name]:raise ValueError('Calibration manifest changed')
    train={row['id'] for row in manifest(config['training_manifest'])}
    if train & {row['id'] for row in data['calibration_query']+data['calibration_gallery']}:raise ValueError('Training/calibration ID overlap')
    q=data['calibration_query'];g=data['calibration_gallery']
    if {r['id'] for r in q}!={r['id'] for r in g} or {r['path'] for r in q}&{r['path'] for r in g}:raise ValueError('Invalid calibration partitions')
    if {r['camera'] for r in q}&{r['camera'] for r in g}:raise ValueError('Calibration cameras overlap')
    hashes=[]
    for row in data['target_gallery']+data['target_query']:
        value=image(row);w,h=value.size
        hashes.append(hashlib.sha256(f'{w},{h}:'.encode()+value.tobytes()).hexdigest())
    if hashes!=source['audit']['crop_pixels']:raise ValueError('Target pixels changed since frozen transfer benchmark')
    settings=yaml.safe_load((ROOT/config['base_config']).read_text());models={}
    for name in config['models']:
        if name=='msinet':
            ref=settings['references'];path=ref['msinet_weights'];digest=ref['msinet_sha256']
            revision=subprocess.check_output(['git','-C',str(ROOT/ref['msinet_source']),'rev-parse','HEAD'],text=True).strip()
            if revision!=ref['msinet_revision']:raise ValueError('MSINet revision changed')
            models[name]=dict(checkpoint=path,sha256=digest,kind='msinet',dimension=768,
                preprocessing='256-square RGB, ImageNet normalization',training='Author clean-VRU checkpoint; different training recipe/budget; upstream pretraining not independently audited',
                source_revision=revision)
        else:
            spec=source['models'][name];path=spec['checkpoint'];digest=spec['sha256']
            models[name]=dict(checkpoint=path,sha256=digest,kind='transreid',dimension=3840,
                preprocessing='256-square RGB, TransReID normalization',training='VeRi-pretrained; '+('VRAI fine-tuning' if name=='original' else 'balanced VRAI+VRU fine-tuning with AdaSP'))
        if sha256(ROOT/path)!=digest:raise ValueError('Frozen checkpoint checksum mismatch')
    signed=dict(config=config,models=models,target_protocol=source['config']['protocol'],
        files={key:sha256(ROOT/config[key]) for key in ['transfer_selection','base_config','calibration_query','calibration_gallery','prepared_audit','training_manifest']},
        implementation={str(path.relative_to(ROOT)):sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
        shared={str(path.relative_to(ROOT)):sha256(path) for path in [ROOT/'utils/experiments/aerial_transfer/evaluation.py',ROOT/'utils/vehicle_reid/operating_points.py',ROOT/'utils/benchmarking/msinet_baseline.py',ROOT/'utils/reid_teacher/model.py',ROOT/'utils/reid_teacher/data.py']},
        source_pixels=[(r['path'],Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns) for r in q+g],data=data,
        calibration='Same source VRAI fit identities for every model; no VisDrone fitting or target-selected operating points',
        limitations=['Previously observed VisDrone protocol; this is an added baseline, not a new blind target test',
            'Same-video GT boxes, not cross-camera or full tracking',
            'Training recipes and data exposure differ; comparison does not isolate architecture',
            'Historical joint models used VisDrone and are excluded from unseen-data comparison'])
    path=ROOT/config['output']/'selection.pt'
    if path.exists():
        if load(path)!=signed:raise ValueError('Reference inputs changed; use new output/state paths')
    else:save(signed,path)
    (path.parent/'protocol.yaml').write_text(yaml.safe_dump({k:v for k,v in signed.items() if k not in ['data','source_pixels']}))
    return signed
