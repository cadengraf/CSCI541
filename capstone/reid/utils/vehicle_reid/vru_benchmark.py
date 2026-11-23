"""Frozen specialist retrieval on the existing full VRU single-gallery split."""
import argparse
from copy import deepcopy
import csv
import gc
from pathlib import Path
import time
from PIL import Image
import torch
import yaml
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256
from utils.benchmarking.vru import single_gallery_metrics
from .models import Embeddings


def read_data(config,limit=None):
    data={}
    for name in ['train','query','gallery']:
        with (ROOT/config[name+'_manifest']).open() as file:data[name]=list(csv.DictReader(file))
    train={row['id'] for row in data['train']}
    query={row['id'] for row in data['query']};gallery=[row['id'] for row in data['gallery']]
    if train&query or query!=set(gallery) or len(gallery)!=len(set(gallery)):
        raise ValueError('Expected disjoint VRU train/test identities and one gallery image per test ID')
    if limit:
        ids=set(sorted(query)[:limit])
        for name in ['query','gallery']:data[name]=[row for row in data[name] if row['id'] in ids]
    paths={name:{str(Path(row['path']).resolve()) for row in rows} for name,rows in data.items()}
    if paths['query']&paths['gallery'] or paths['train']&(paths['query']|paths['gallery']):raise ValueError('VRU image paths overlap')
    if any(not Path(row['path']).is_file() for name in ['query','gallery'] for row in data[name]):raise FileNotFoundError('VRU crop is missing')
    return data


def prepare_reference(config):
    from utils.reid_teacher.data import read
    settings=read();destination=ROOT/config['models']['pretrained']
    selection=yaml.safe_load((ROOT/config['models']['original']).read_text())
    manifest=dict(selection='Explicit frozen VeRi-pretrained reference; no VRU model selection',finalized_selection=True,
        calibration_manifests=selection['calibration_manifests'],candidate=dict(kind='transreid_pretrained',
        model='TransReID VeRi pretrained, camera-agnostic',checkpoint=settings['pretrained']['path'],
        checkpoint_sha256=settings['pretrained']['sha256']))
    if destination.exists() and yaml.safe_load(destination.read_text())!=manifest:raise ValueError('Pretrained reference manifest changed')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(yaml.safe_dump(manifest))
    if not (ROOT/config['models']['degradation']).exists():
        from .appearance_encoder import export
        export('checkpoints/appearance_training/degradation/last.pt',config['models']['degradation'])


def extract(config,data,name,selection):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest=yaml.safe_load((ROOT/selection).read_text())
    signature=dict(model=manifest,manifests={key:sha256(ROOT/config[key+'_manifest']) for key in ['train','query','gallery']},
        subset_ids=sorted({row['id'] for row in data['query']}),batch_size=config['batch_size'],
        implementation={key:sha256(Path(__file__).parent/key) for key in ['vru_benchmark.py','models.py','appearance_encoder.py']},
        metric_sha256=sha256(ROOT/'utils/benchmarking/vru.py'))
    destination=ROOT/config['state_root']/name/'state.pt'
    banks={group:[] for group in ['query','gallery']};seconds=0.
    if destination.exists():
        state=load(destination)
        if state['signature']!=signature:raise ValueError(f'{name}: VRU inputs changed; use new state/output paths')
        if state['complete']:print(f'{name}: reusing complete VRU score',flush=True);return state['result']
        banks,seconds=state['banks'],state['seconds']
    encoder=Embeddings(dict(selection=selection,batch_size=config['batch_size']),device)
    def checkpoint(complete=False,result=None):
        save(dict(signature=signature,banks=banks,seconds=seconds,complete=complete,result=result,device=str(device)),destination)
    for group in ['query','gallery']:
        rows=data[group];offset=sum(len(chunk) for chunk in banks[group]);started=time.perf_counter()
        for start in range(offset,len(rows),config['batch_size']):
            images=[]
            for row in rows[start:start+config['batch_size']]:
                with Image.open(row['path']) as image:images.append(encoder.transform(image.convert('RGB')))
            with torch.inference_mode():vectors=encoder.model(torch.stack(images).to(device)).float().cpu()
            if vectors.shape[1]!=3840 or not torch.isfinite(vectors).all():raise ValueError('Invalid specialist features')
            banks[group].append(vectors)
            if (start//config['batch_size']+1)%10==0:
                print(f'{name} VRU {group}: {min(start+config["batch_size"],len(rows))}/{len(rows)} ({device})',flush=True)
            if (start//config['batch_size']+1)%config['checkpoint_batches']==0 or start+config['batch_size']>=len(rows):
                if device.type=='cuda':torch.cuda.synchronize()
                seconds+=time.perf_counter()-started;checkpoint();started=time.perf_counter()
    del encoder;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    query,gallery=[torch.cat(banks[group]) for group in ['query','gallery']]
    result=single_gallery_metrics(query,[row['id'] for row in data['query']],gallery,[row['id'] for row in data['gallery']])
    result.update(encoder=name,extraction_seconds=seconds,device=str(device),checkpoint_sha256=manifest['candidate']['checkpoint_sha256'])
    checkpoint(True,result)
    print(f"{name}: VRU Rank-1 {result['rank1']:.2%}, Rank-5 {result['rank5']:.2%}, mAP {result['map']:.2%}",flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/vehicle_vru_benchmark.yaml')
    parser.add_argument('--check',action='store_true',help='Real-input smoke check on 16 identities; separate outputs')
    args=parser.parse_args(argv);config=yaml.safe_load((ROOT/args.config).read_text())
    if config['device']!='auto':raise ValueError('Device must remain auto')
    torch.set_num_threads(config['cpu_threads']);prepare_reference(config)
    if args.check:
        config=deepcopy(config);config['output']+='/check16';config['state_root']+='/check16'
    data=read_data(config,16 if args.check else None)
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,selection in config['models'].items():
        results.append(extract(config,data,name,selection))
        (output/'report.yaml').write_text(yaml.safe_dump(dict(protocol='VRU fixed local single-gallery split; seed 42',
            smoke_check=args.check,models=results,training_performed=False,threshold_tuning=False,
            note='Not camera-filtered: VRU manifests have no camera IDs. Single-positive mAP is reciprocal rank. Extraction timing excludes detection/tracking.')))
    with (output/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    lines=['# Frozen encoders on VRU','',f"{'SMOKE CHECK ONLY. ' if args.check else ''}{len(data['query'])} queries, {len(data['gallery'])} gallery identities. One fixed gallery image per identity. No training or threshold selection.",
        '', '| Encoder | Rank-1 | Rank-5 | mAP |','|---|---:|---:|---:|']
    for row in results:lines.append(f"| {row['encoder']} | {100*row['rank1']:.2f} | {100*row['rank5']:.2f} | {100*row['map']:.2f} |")
    lines+=['','Single-positive mAP is reciprocal rank and is not directly comparable to multi-positive VRAI mAP. This evaluates retrieval, not video return recovery.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
