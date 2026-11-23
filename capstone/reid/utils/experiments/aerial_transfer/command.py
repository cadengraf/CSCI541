"""Run/resume frozen aerial dataset transfer: benchmark.py aerial-transfer."""
import argparse
import csv
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import torch
import torch.nn.functional as F
import yaml
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256
from utils.reid_teacher.model import build
from utils.reid_teacher.data import transform
from .data import prepare,crop
from .evaluation import evaluate


def freeze(config):
    mixed=yaml.safe_load((ROOT/config['mixed_selection']).read_text())
    original=yaml.safe_load((ROOT/config['original_selection']).read_text())['candidate']
    points=yaml.safe_load((ROOT/config['original_operating_points']).read_text())
    calibration=load(ROOT/config['original_calibration_state'])
    if calibration['signature']['model']['checkpoint_sha256']!=original['checkpoint_sha256']:raise ValueError('Original threshold/model mismatch')
    del calibration
    models={}
    for name in config['models']:
        source=original if name=='original' else mixed['models'][name]
        path=ROOT/source['checkpoint'];digest=sha256(path)
        if digest!=source.get('sha256',source.get('checkpoint_sha256')):raise ValueError('Frozen source weights changed')
        checkpoint=load(path)
        if name=='original':
            point=next(p['calibration'] for p in points['operating_points'] if p['empirical_calibration_budget']==.01)
            thresholds={'vrai_fit':{key:point[key] for key in ['threshold','margin']}}
        else:
            if checkpoint['signature']['inputs']!=mixed['training_signature']:raise ValueError('Mixed training provenance changed')
            thresholds={domain+'_fit':{key:checkpoint['best_metrics'][domain]['fit'][key] for key in ['threshold','margin']} for domain in ['vrai','vru']}
        models[name]=dict(checkpoint=source['checkpoint'],sha256=digest,thresholds=thresholds)
        del checkpoint
    data,audit=prepare(config)
    implementations={str(p.relative_to(ROOT)):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    selected=dict(config=config,models=models,implementations=implementations,audit=audit,data=data,
        selection='All five candidates and source-fitted thresholds fixed before VisDrone evaluation; no target fitting',
        limitations=['GT vehicle boxes isolate appearance; no detector, masks, tracker, or spatial gating',
            'Local chronological same-video protocol, not an official VisDrone ReID leaderboard',
            'No cross-camera identity annotation; cannot measure cross-camera transfer',
            'VisDrone excluded from these encoders fine-tuning; other historical joint models used VisDrone',
            'Small correlated track samples; report per-sequence counts and do not infer field leadership'])
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True);path=output/'selection.pt'
    if path.exists() and load(path)!=selected:raise ValueError('Frozen transfer selection changed; use new output/state paths')
    if not path.exists():save(selected,path)
    (output/'protocol.yaml').write_text(yaml.safe_dump({key:value for key,value in selected.items() if key!='data'}))
    return selected


def worker(config,name):
    torch.set_num_threads(config['cpu_threads']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selection=load(ROOT/config['output']/'selection.pt');spec=selection['models'][name]
    path=ROOT/config['state_root']/name/'state.pt';path.parent.mkdir(parents=True,exist_ok=True)
    signature=sha256(ROOT/config['output']/'selection.pt')
    state=load(path) if path.exists() else dict(signature=signature,parts={'gallery':[],'query':[]},seconds=0.,complete=False)
    if state['signature']!=signature:raise ValueError('Transfer signature changed')
    if state['complete']:print(name+': completed result reused',flush=True);return
    checkpoint=load(ROOT/spec['checkpoint']);settings=yaml.safe_load((ROOT/config['base_config']).read_text())
    model=build(settings,len(checkpoint['identity_mapping']),pretrained=False).to(device)
    model.load_state_dict(checkpoint['model']);model.eval().requires_grad_(False);del checkpoint
    conversion=transform(settings['model']['image_size']);data=selection['data']
    print(f'{name}: inference on {device}',flush=True)
    for group,rows in data.items():
        start=sum(len(part) for part in state['parts'][group]);pending=[];began=time.perf_counter()
        for offset in range(start,len(rows),config['batch_size']):
            images=torch.stack([conversion(crop(row)) for row in rows[offset:offset+config['batch_size']]]).to(device)
            with torch.inference_mode():vectors=F.normalize(model(images).float(),dim=1).cpu()
            if vectors.shape[1]!=3840 or not torch.isfinite(vectors).all():raise ValueError('Invalid embeddings')
            pending.append(vectors)
            if len(pending)>=config['checkpoint_batches'] or offset+config['batch_size']>=len(rows):
                state['parts'][group].append(torch.cat(pending));state['seconds']+=time.perf_counter()-began
                save(state,path);pending=[];began=time.perf_counter()
                print(f'{name} {group}: {min(offset+config["batch_size"],len(rows))}/{len(rows)}',flush=True)
    vectors={group:torch.cat(parts) for group,parts in state['parts'].items()}
    result,decisions=evaluate(vectors,data,spec['thresholds'],config['protocol'])
    result.update(model=name,checkpoint_sha256=spec['sha256'],extraction_seconds=state['seconds'],
                  timing_scope='Image reading, crop, transform, embedding and transfer; excludes model loading and all detector/tracker costs')
    (ROOT/config['output']/f'{name}.yaml').write_text(yaml.safe_dump(result))
    state.update(complete=True,result=result,decisions=decisions);save(state,path)
    print(f'{name}: {result["retrieval"]}',flush=True)


def summarize(config):
    output=ROOT/config['output'];lines=['# Frozen encoders on unseen VisDrone vehicles','',
        'Local chronological same-video GT-crop retrieval. No target training or threshold fitting. Not cross-camera evaluation.','',
        '| Model | Known queries | Rank-1 | mAP | Rejection threshold source | Correct known acceptance | Unknown FAR | Wrong known acceptance |',
        '|---|---:|---:|---:|---|---:|---:|---:|'];flat=[]
    for name in config['models']:
        path=output/f'{name}.yaml'
        if not path.exists():continue
        result=yaml.safe_load(path.read_text());r=result['retrieval']
        for source,point in result['open_set'].items():
            lines.append(f'| {name} | {r["queries"]} | {r["rank1"]:.2%} | {r["map"]:.2%} | {source} | {point["known_recovery"]:.2%} | {point["unknown_far"]:.2%} | {point["wrong_known_rate"]:.2%} |')
            flat.append(dict(model=name,**r,threshold_source=source,**{key:point[key] for key in ['known_recovery','unknown_far','wrong_known_rate']}))
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    if flat:
        with (output/'summary.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/aerial_transfer.yaml')
    parser.add_argument('--worker',help=argparse.SUPPRESS);args=parser.parse_args(argv)
    config=yaml.safe_load((ROOT/args.config).read_text())
    if config['device']!='auto':raise ValueError('Device must be auto')
    if args.worker:return worker(config,args.worker)
    root=ROOT/config['state_root'];root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Transfer benchmark is already running') from None
        freeze(config);status=dict(status='running',completed=[],current=None);output=ROOT/config['output']
        def progress():
            temporary=output/'queue.partial';temporary.write_text(yaml.safe_dump(status));temporary.replace(output/'queue.yaml')
        try:
            for name in config['models']:
                status['current']=name;progress()
                subprocess.run([sys.executable,str(ROOT/'benchmark.py'),'aerial-transfer','--config',args.config,'--worker',name],cwd=ROOT,check=True)
                status['completed'].append(name);summarize(config);progress()
            from .review import render
            render(config);status.update(status='complete',current=None);progress()
        except BaseException as error:
            status.update(status='failed',error=str(error));progress();raise
