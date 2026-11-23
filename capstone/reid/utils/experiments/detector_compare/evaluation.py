"""Causal full-sequence association with transaction-safe RLE/checkpoint resume."""
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import yaml
from utils.tracking.common import load, save
from utils.full_frame.tracker import VehicleTracker
from utils.full_frame.mots_benchmark import write_frame
from utils.vehicle_reid.recovery import Recovery
from .models import identity_predictions


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def infer_frame(path, detector, encoder, tracker, config, device, index):
    sync(device);start=time.perf_counter()
    image=cv2.imread(str(path))
    if image is None:raise ValueError(f'Cannot read {path}')
    raw=detector(image)
    pred=identity_predictions(image,raw,encoder,config)
    identities=tracker.update(pred,index)
    sync(device)
    return image,pred,identities,start


def sequence(root, config, name, seq, detector, encoder, tracking, signature, device):
    folder=root/config['dataset_root']/'training/image_02'/seq
    paths=sorted(folder.glob('*.png'))
    if not paths or [int(p.stem) for p in paths]!=list(range(len(paths))):raise ValueError(f'Incomplete sequence {seq}')
    base=root/config['output']/config['run']/'detectors'
    state_path=root/config['state_root']/config['run']/name/f'{seq}.pt'
    destination=base/'predictions'/name/'data'/f'{seq}.txt'
    state=load(state_path) if state_path.exists() else dict(signature=signature,next_frame=0,offset=0,
        complete=False,times=[],checkpoint_seconds=0.,tracker=VehicleTracker(tracking),
        recovery=Recovery(root/config['dataset_root']/'instances_txt'/f'{seq}.txt',10))
    if state['signature']!=signature:raise ValueError('Detector inputs changed; choose a new run name')
    if state['complete']:
        if not destination.exists() or destination.stat().st_size!=state['offset']:raise ValueError('Complete RLE output is missing or changed')
        print(f'{name}/{seq}: complete predictions reused',flush=True);return state
    destination.parent.mkdir(parents=True,exist_ok=True)
    begin=state['next_frame']
    if begin and (not destination.exists() or destination.stat().st_size<state['offset']):raise ValueError('Committed predictions missing')
    with destination.open('r+' if begin else 'w+') as stream:
        if begin:stream.seek(state['offset']);stream.truncate()
        for index in range(begin,len(paths)):
            _,pred,identities,start=infer_frame(paths[index],detector,encoder,state['tracker'],config,device,index)
            write_frame(stream,index,pred,identities)
            state['times'].append((time.perf_counter()-start)*1000)
            # Truth is evaluation-only and never reaches detector/encoder/tracker.
            decisions=[dict(identity=i,status='reference') for i in identities]
            state['recovery'].update(index,[m.numpy() for m in pred['masks']],decisions)
            state['next_frame']=index+1
            if (index+1)%config['checkpoint_frames']==0 or index+1==len(paths):
                began=time.perf_counter();stream.flush();state['offset']=stream.tell()
                state['complete']=index+1==len(paths)
                if state['complete']:state['recovery_summary']=state['recovery'].summary()
                save(state,state_path)
                state['checkpoint_seconds']+=time.perf_counter()-began
                print(f'{name}/{seq}: {index+1}/{len(paths)} frames committed',flush=True)
    return state


def annotated(image,pred,identities,label):
    result=image.copy()
    for mask,box,identity in zip(pred['masks'],pred['boxes'],identities):
        color=np.array([(identity*71)%200+40,(identity*113)%200+40,(identity*37)%200+40],dtype=np.uint8)
        m=mask.numpy();result[m]=(.7*result[m]+.3*color).astype(np.uint8)
        x,y,right,bottom=map(int,box)
        cv2.rectangle(result,(x,y),(right,bottom),color.tolist(),1)
        cv2.putText(result,str(identity),(x,max(15,y)),cv2.FONT_HERSHEY_SIMPLEX,.5,color.tolist(),1)
    cv2.putText(result,label,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    return result


def video_frame(image):
    """MP4 encoders require even dimensions; preserve every original image row."""
    h,w=image.shape[:2]
    return cv2.copyMakeBorder(image,0,h%2,0,w%2,cv2.BORDER_CONSTANT,value=(0,0,0))


def live_video(root, config, name, detector, encoder, tracking, signature, device):
    base=root/config['output']/config['run']/'detectors';spec=config['video']
    path=base/f'{name}.mp4';record=base/f'{name}_video.yaml'
    if path.exists() and record.exists():
        old=yaml.safe_load(record.read_text())
        if old['signature']!=signature:raise ValueError('Live video signature changed')
        return old
    paths=sorted((root/config['dataset_root']/'training/image_02'/spec['sequence']).glob('*.png'))[:spec['frames']]
    first=video_frame(cv2.imread(str(paths[0])));h,w=first.shape[:2]
    temporary=path.with_suffix('.partial.mp4')
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),spec['fps'],(w,h))
    if not writer.isOpened():raise RuntimeError('Cannot create annotated video')
    tracker=VehicleTracker(tracking);times=[]
    try:
        for index,source in enumerate(paths):
            image,pred,identities,start=infer_frame(source,detector,encoder,tracker,config,device,index)
            frame=video_frame(annotated(image,pred,identities,name+' + frozen TransReID'))
            if frame.shape[:2]!=(h,w):raise ValueError('Video frame dimensions changed within sequence')
            writer.write(frame)
            sync(device);times.append((time.perf_counter()-start)*1000)
        start=time.perf_counter();writer.release();times[-1]+=(time.perf_counter()-start)*1000
    finally:writer.release()
    temporary.replace(path)
    result=dict(signature=signature,frames=len(times),mean_ms=float(np.mean(times)),
        p95_ms=float(np.percentile(times,95)),hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        scope='Fresh decode + detector + mask filtering + specialist + association + annotation + video encoding/finalization; initialization and offline scoring excluded')
    record.write_text(yaml.safe_dump(result));return result
