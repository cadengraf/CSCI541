"""Compare Co-DINO and RF-DETR Seg with one frozen specialist and local tracker."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import yaml
from utils.benchmarking.suite.command import digest, execute, write, discard_duplicate_pages

ROOT=Path(__file__).resolve().parents[3]
NAMES=('codino','rfdetr')


def read(path):
    config=yaml.safe_load((ROOT/path).read_text())
    from utils.full_frame.protocol import VALIDATION
    if config['device']!='auto' or config['sequences']!=list(VALIDATION):
        raise ValueError('Use device:auto and all nine conventional KITTI validation sequences')
    if not config['run'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in config['run']):
        raise ValueError('run must be a single safe folder name')
    if config['checkpoint_frames']<1 or config['video']['frames']<1:raise ValueError('Frame counts must be positive')
    if config['video']['sequence'] not in [f'{n:04d}' for n in config['sequences']]:
        raise ValueError('Choose a validation sequence for the common video workload')
    return config


def freeze(config):
    from utils.full_frame.mots_evaluation import TRACKEVAL_REVISION
    root=ROOT/config['dataset_root'];frames={};annotations={}
    for number in config['sequences']:
        seq=f'{number:04d}';paths=sorted((root/'training/image_02'/seq).glob('*.png'))
        if not paths or [int(p.stem) for p in paths]!=list(range(len(paths))):raise ValueError(f'Incomplete {seq}')
        frames[seq]=[[p.name,p.stat().st_size,p.stat().st_mtime_ns] for p in paths]
        annotations[seq]=digest(root/'instances_txt'/f'{seq}.txt')
    specialist=yaml.safe_load((ROOT/config['specialist_selection']).read_text())
    checkpoint=ROOT/specialist['candidate']['checkpoint']
    if digest(checkpoint)!=specialist['candidate']['checkpoint_sha256']:raise ValueError('Specialist weights changed')
    local=yaml.safe_load((ROOT/config['tracking_config']).read_text())['association']['local']
    files=[*Path(__file__).parent.glob('*.py'), ROOT/'utils/full_frame/tracker.py',
        ROOT/'utils/full_frame/mots_evaluation.py',ROOT/'utils/vehicle_reid/models.py',
        ROOT/'utils/vehicle_reid/recovery.py',ROOT/'utils/reid_teacher/model.py',
        ROOT/'utils/reid_teacher/data.py',ROOT/'utils/reid_teacher/evaluation.py',ROOT/'utils/teacher/model.py']
    protocol=dict(dataset=str(root),frames=frames,annotations=annotations,tracking=local,
        specialist=specialist['candidate'],score_threshold=config['score_threshold'],
        mask_threshold=config['mask_threshold'],minimum_mask_pixels=config['minimum_mask_pixels'],
        specialist_batch_size=config['specialist_batch_size'],video=config['video'],evaluator=TRACKEVAL_REVISION)
    environments={}
    script='''import sys,json,importlib.metadata as m
packages={}
for p in ["torch","torchvision","numpy","timm","rfdetr","supervision","transformers","opencv-python","opencv-python-headless"]:
    try:packages[p]=m.version(p)
    except m.PackageNotFoundError:pass
print(json.dumps(dict(python=sys.version,packages=packages)))
'''
    compile(script,'environment_snapshot','exec')
    for name in NAMES:
        python=(ROOT/config[name]['environment']).resolve()/'bin/python'
        environments[name]=json.loads(subprocess.check_output([str(python),'-c',script],text=True))
    signed=dict(config=config,protocol=protocol,environments=environments,implementation={str(p.relative_to(ROOT)):digest(p) for p in files},
        detectors={name:digest(ROOT/config[name]['checkpoint']) for name in NAMES})
    base=ROOT/config['output']/config['run'];base.mkdir(parents=True,exist_ok=True)
    path=base/'manifest.yaml'
    if path.exists():
        if yaml.safe_load(path.read_text())!=signed:raise ValueError('Comparison inputs changed; set a new run name')
    else:write(path,signed)
    return signed


def worker(config,name,check=False):
    import cv2
    import numpy as np
    import torch
    from utils.full_frame.tracker import VehicleTracker
    from utils.full_frame.mots_evaluation import evaluate
    from utils.teacher.evaluation import write_csv
    from utils.vehicle_reid.models import Embeddings
    from .models import Detector, identity_predictions
    from .evaluation import sequence,live_video
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name=='codino' and device.type!='cuda':raise RuntimeError('device:auto selected CPU; Co-DINO requires a CUDA host')
    torch.set_num_threads(4)
    signed=None if check else yaml.safe_load((ROOT/config['output']/config['run']/'manifest.yaml').read_text())
    token=None if check else hashlib.sha256(yaml.safe_dump(signed,sort_keys=True).encode()).hexdigest()
    base=ROOT/config['output']/config['run']/'detectors';result_path=base/f'{name}.yaml'
    if not check and result_path.exists():
        result=yaml.safe_load(result_path.read_text())
        if result['signature']!=token:raise ValueError('Stored detector result signature changed')
        for seq,hash_value in result['predictions'].items():
            if digest(base/'predictions'/name/'data'/f'{seq}.txt')!=hash_value:raise ValueError('Stored prediction output changed')
        if not (base/f'{name}.mp4').exists():raise ValueError('Completed review video is missing')
        print(name+': complete validation and live-video measurement reused',flush=True);return
    detector=Detector(name,config,ROOT,device)
    encoder=Embeddings(dict(selection=config['specialist_selection'],batch_size=config['specialist_batch_size']),device)
    tracking=yaml.safe_load((ROOT/config['tracking_config']).read_text())['association']['local']
    if check:
        tracker=VehicleTracker(tracking)
        paths=sorted((ROOT/config['dataset_root']/'training/image_02'/config['video']['sequence']).glob('*.png'))[:2]
        for index,path in enumerate(paths):
            image=cv2.imread(str(path));pred=identity_predictions(image,detector(image),encoder,config)
            ids=tracker.update(pred,index)
            assert pred['embeddings'].shape==(len(ids),3840)
            assert torch.isfinite(pred['embeddings']).all()
            assert pred['masks'].shape[1:]==image.shape[:2]
            print(f'CHECK ONLY {name}/{index}: {len(ids)} vehicles; masks {tuple(pred["masks"].shape)}; embeddings {tuple(pred["embeddings"].shape)}',flush=True)
        return
    # Same unscored warm-up image for both methods; no GT or future state enters tracking.
    warm=ROOT/config['dataset_root']/'training/image_02'/config['video']['sequence']/'000000.png'
    image=cv2.imread(str(warm));identity_predictions(image,detector(image),encoder,config)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    summaries={}
    for number in config['sequences']:
        seq=f'{number:04d}'
        summaries[seq]=sequence(ROOT,config,name,seq,detector,encoder,tracking,token,device)
    seq_info={seq:len(state['times']) for seq,state in summaries.items()}
    metrics=evaluate(ROOT/config['dataset_root'],base/'predictions',[name],seq_info,base/name)[0]
    video=live_video(ROOT,config,name,detector,encoder,tracking,token,device)
    counts=Counter()
    for state in summaries.values():
        counts.update({k:v for k,v in state['recovery_summary'].items() if isinstance(v,int)})
    frames=sum(seq_info.values());times=[v for state in summaries.values() for v in state['times']]
    checkpoint_seconds=sum(state['checkpoint_seconds'] for state in summaries.values())
    result=dict(signature=token,model=name,metrics=metrics,video=video,recovery=dict(counts),frames=frames,
        mean_ms=float(np.mean(times))+1000*checkpoint_seconds/frames,checkpoint_seconds=checkpoint_seconds,
        hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor() or 'CPU',
        torch=str(torch.__version__),peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
        predictions={seq:digest(base/'predictions'/name/'data'/f'{seq}.txt') for seq in seq_info},
        scope='Decode, detector forward, mask filtering, frozen TransReID, local association, RLE and checkpoint writes; GT audit/TrackEval excluded. Fresh video timing separately includes annotation/encoding.',
        caveat='Co-DINO uses KITTI/VRAI-finetuned weights and computes unused joint embeddings. RF-DETR exposure: '+
            config['rfdetr'].get('training','COCO-pretrained segmentation weights')+
            '. Different training exposure, resolutions, precision and software environments.')
    write(result_path,result);discard_duplicate_pages(base/name)
    print(f'{name}: HOTA {metrics["HOTA"]:.2f}, IDF1 {metrics["IDF1"]:.2f}, live-video {video["mean_ms"]:.2f} ms/frame',flush=True)


def summarize(config):
    from utils.teacher.evaluation import write_csv
    from utils.benchmarking.suite.report import render
    base=ROOT/config['output']/config['run'];folder=base/'detectors';rows=[]
    for name in NAMES:
        result=yaml.safe_load((folder/f'{name}.yaml').read_text());m=result['metrics']
        rows.append(dict(detector='Co-DINO / RefineMask' if name=='codino' else 'RF-DETR Seg Large',
            reid='Frozen TransReID',HOTA=m['HOTA'],DetA=m['DetA'],AssA=m['AssA'],IDF1=m['IDF1'],sMOTSA=m['sMOTSA'],
            IDSW=m['IDSW'],pipeline_ms=result['mean_ms'],video_ms=result['video']['mean_ms'],hardware=result['hardware'],
            recovered=result['recovery'].get('recovered',0),eligible=result['recovery'].get('eligible_returns',0),
            false_reconnections=result['recovery'].get('false_reconnections',0)))
    write_csv(folder/'summary.csv',rows)
    render({**config,'joint_checkpoint':config['codino']['checkpoint']},dict(status='complete',
        config={**config,'joint_checkpoint':config['codino']['checkpoint']},stages=dict(detectors=dict(status='complete'))))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/detector_compare.yaml')
    parser.add_argument('--check',action='store_true',help='Two-frame integration check only; no benchmark scores or scorecard changes')
    parser.add_argument('--worker',choices=NAMES,help=argparse.SUPPRESS)
    args=parser.parse_args(argv);config=read(args.config)
    if args.worker:return worker(config,args.worker,args.check)
    if not (ROOT/config['rfdetr']['environment']/'bin/python').exists() or not (ROOT/config['rfdetr']['checkpoint']).exists():
        from .setup import prepare
        prepare(config['rfdetr'])
    base=ROOT/config['output']/config['run'];base.mkdir(parents=True,exist_ok=True)
    with (base/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Detector comparison is already running') from None
        if not args.check:freeze(config)
        status=dict(status='running',completed=[],current=None)
        try:
            for name in NAMES:
                status['current']=name
                if not args.check:write(base/'queue.yaml',status)
                python=(ROOT/config[name]['environment']).resolve()/'bin/python'
                command=[str(python),'-u','-m','utils.experiments.detector_compare.command','--config',args.config,'--worker',name]
                if args.check:command.append('--check')
                execute(command,base/'logs'/f'{name}{"_check" if args.check else ""}.log')
                status['completed'].append(name)
            if not args.check:summarize(config)
            status.update(status='complete',current=None)
        except BaseException as error:
            status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
        finally:
            if not args.check:write(base/'queue.yaml',status)


if __name__=='__main__':main()
