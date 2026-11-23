"""Full CityFlowV2 validation queue: benchmark.py cityflow."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256
from utils.experiments.vehicle_reference.inference import build_model
from .data import freeze,capture,schedule,detections
from .association import MultiCamera


def track(config,selection,name):
    torch.set_num_threads(config['cpu_threads']);cv2.setNumThreads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');spec=selection['models'][name]
    digest=sha256(ROOT/config['output']/'selection.pt');model=None;conversion=None;all_events=[];timings={}
    for scene_index,(scene,cameras) in enumerate(selection['scenes'].items()):
        directory=ROOT/config['state_root']/name/scene;directory.mkdir(parents=True,exist_ok=True);path=directory/'state.pt'
        state=load(path) if path.exists() else dict(signature=digest,complete=False,next_frames={cam:1 for cam in cameras},
            algorithm=MultiCamera(config,spec['point'],cameras,namespace=scene_index*100_000_000),chunks=[],events=[],
            frames=0,timing=dict(decode=0.,embedding=0.,association=0.,checkpoint=0.,preparation=0.,initialization=0.),last_timestamp=-float('inf'))
        if state['signature']!=digest:raise ValueError('CityFlow run signature changed')
        if not state['complete']:
            if model is None:
                started=time.perf_counter();model,conversion=build_model(config,spec,device);state['timing']['initialization']+=time.perf_counter()-started
            print(f'{name}/{scene}: {device}, resume {state["frames"]}/{sum(c["frames"] for c in cameras.values())}',flush=True)
            started=time.perf_counter();prepared={cam:detections(camera,config) for cam,camera in cameras.items()};state['timing']['preparation']+=time.perf_counter()-started
            caps={};pending=[];interval=0
            def commit(complete=False):
                started=time.perf_counter();part=directory/f'predictions_{len(state["chunks"]):05d}.txt';temporary=part.with_suffix('.partial')
                temporary.write_text(''.join(pending));temporary.replace(part);state['chunks'].append(part.name)
                state['complete']=complete;save(state,path);state['timing']['checkpoint']+=time.perf_counter()-started;pending.clear()
            try:
                for timestamp,cam,frame in schedule(cameras,state['next_frames']):
                    if timestamp<state['last_timestamp']-1e-8:raise ValueError('Resumed global clock moved backwards')
                    started=time.perf_counter()
                    if cam not in caps:caps[cam]=capture(cameras[cam]['path'],frame)
                    okay,image=caps[cam].read()
                    if not okay or int(round(caps[cam].get(cv2.CAP_PROP_POS_FRAMES)))!=frame:raise ValueError(f'Video decode/cursor mismatch {cam}/{frame}')
                    state['timing']['decode']+=time.perf_counter()-started
                    boxes,scores,quality=prepared[cam][frame];started=time.perf_counter();outputs=[]
                    if len(boxes):
                        rgb=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                        for offset in range(0,len(boxes),config['batch_size']):
                            batch=torch.stack([conversion(rgb.crop(tuple(map(int,box)))) for box in boxes[offset:offset+config['batch_size']]]).to(device)
                            with torch.inference_mode():outputs.append(F.normalize(model(batch).float(),dim=1).cpu())
                    vectors=torch.cat(outputs).numpy() if outputs else np.empty((0,spec['dimension']),np.float32)
                    if vectors.shape!=(len(boxes),spec['dimension']) or not np.isfinite(vectors).all():raise ValueError('Invalid online embeddings')
                    state['timing']['embedding']+=time.perf_counter()-started;started=time.perf_counter()
                    rows=state['algorithm'].update(cam,frame,timestamp,boxes,vectors,quality)
                    state['timing']['association']+=time.perf_counter()-started
                    for box,row in zip(boxes,rows):
                        x,y,x2,y2=map(int,box);identity=row['identity']
                        pending.append(f'{int(cam[1:])} {identity} {frame} {x} {y} {x2-x} {y2-y} -1 -1\n')
                        if row['action'] in ['new','cross_camera','same_camera_recovery']:
                            state['events'].append(dict(scene=scene,camera=cam,frame=frame,identity=identity,action=row['action'],box=[x,y,x2-x,y2-y],timestamp=timestamp))
                    state['next_frames'][cam]=frame+1;state['last_timestamp']=timestamp;state['frames']+=1;interval+=1
                    if interval>=config['checkpoint_frames']:
                        commit();interval=0
                        elapsed=sum(state['timing'].values());print(f'{name}/{scene}: {state["frames"]} frames committed; {elapsed:.1f}s decode/embedding/association/checkpoint; profiles={len(state["algorithm"].profiles)}',flush=True)
                commit(complete=True)
            finally:
                for cap in caps.values():cap.release()
        all_events+=state['events'];timings[scene]=dict(frames=state['frames'],**state['timing'],actions=dict(state['algorithm'].counts))
    output=ROOT/config['output']/name;output.mkdir(parents=True,exist_ok=True);temporary=output/'predictions.partial'
    with temporary.open('w') as file:
        for scene in selection['scenes']:
            directory=ROOT/config['state_root']/name/scene;state=load(directory/'state.pt')
            for part in state['chunks']:
                with (directory/part).open() as source:shutil.copyfileobj(source,file)
    temporary.replace(output/'predictions.txt')
    save(dict(events=all_events,timing=timings),output/'diagnostics.pt')
    from .evaluation import score,audit_claims
    metrics=score(config,output/'predictions.txt');audit=audit_claims(config,all_events)
    report=dict(model=name,checkpoint_sha256=spec['sha256'],metrics=metrics,claim_audit=audit,timing=timings,
        timing_scope='Provided detections: input preparation, decode, crop/preprocess/embedding, local/global association, initialization and checkpoint writes reported separately. Final persisted checkpoint timing can lag one write. Excludes supplied detector generation, final review rendering and scoring. Not full live detector FPS.',
        smoke_check=bool(config.get('check_frames')),training=spec['training'])
    (output/'report.yaml').write_text(yaml.safe_dump(report));print(yaml.safe_dump(report),flush=True)


def prepare_rois(config,selection):
    for cameras in selection['scenes'].values():
        for camera,spec in cameras.items():
            destination=ROOT/config['state_root']/'ROIs'/'validation'/camera/'roi.jpg';destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():shutil.copy2(spec['roi'],destination)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/cityflow.yaml')
    parser.add_argument('--check',action='store_true',help='Integration only; separate outputs, first 50 frames per camera')
    parser.add_argument('--worker',help=argparse.SUPPRESS);args=parser.parse_args(argv)
    config=yaml.safe_load((ROOT/args.config).read_text())
    if config['device']!='auto':raise ValueError('Use automatic devices')
    if args.check:config=deepcopy(config);config['output']+='/check50';config['state_root']+='/check50';config['check_frames']=50
    if args.worker:return track(config,load(ROOT/config['output']/'selection.pt'),args.worker)
    root=ROOT/config['state_root'];root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('CityFlow benchmark already running') from None
        output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True);status=dict(status='preparing',completed=[],current=None)
        def progress():
            temporary=output/'queue.partial';temporary.write_text(yaml.safe_dump(status));temporary.replace(output/'queue.yaml')
        progress()
        try:
            selection=freeze(config);prepare_rois(config,selection)
            for name in config['models']:
                status.update(status='running',current=name);progress()
                command=[sys.executable,str(ROOT/'benchmark.py'),'cityflow','--config',args.config,'--worker',name]
                if args.check:command.append('--check')
                subprocess.run(command,cwd=ROOT,check=True);status['completed'].append(name);progress()
                from .report import summarize
                summarize(config)
            from .report import render
            status['current']='review video';progress();render(config,selection);status.update(status='complete',current=None);progress()
        except BaseException as error:
            status.update(status='failed',error=str(error));progress();raise
