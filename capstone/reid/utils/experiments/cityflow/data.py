"""Read-only external data, complete cameras, and a chronological event schedule."""
from copy import deepcopy
import heapq
import subprocess
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.ops import nms
import yaml
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256


def capture(path,start=1):
    params=[cv2.CAP_PROP_N_THREADS,1] if hasattr(cv2,'CAP_PROP_N_THREADS') else []
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,params)
    if not cap.isOpened():raise ValueError(f'Cannot open video {path}')
    if start>1 and not cap.set(cv2.CAP_PROP_POS_FRAMES,start-1):raise ValueError('Cannot seek resume frame')
    return cap


def schedule(cameras,next_frames):
    heap=[(spec['offset']+(next_frames[camera]-1)/spec['fps'],camera,next_frames[camera])
          for camera,spec in cameras.items() if next_frames[camera]<=spec['frames']]
    heapq.heapify(heap)
    while heap:
        timestamp,camera,frame=heapq.heappop(heap);yield timestamp,camera,frame
        spec=cameras[camera];frame+=1
        if frame<=spec['frames']:heapq.heappush(heap,(spec['offset']+(frame-1)/spec['fps'],camera,frame))


def freeze(config):
    if config['split']!='validation':raise ValueError('This entry point evaluates the complete labeled validation split')
    root=Path(config['dataset_root']);reference=load(ROOT/config['reference_selection'])
    settings=yaml.safe_load((ROOT/config['base_config']).read_text())
    ref=settings['references'];revision=subprocess.check_output(['git','-C',str(ROOT/ref['msinet_source']),'rev-parse','HEAD'],text=True).strip()
    if revision!=ref['msinet_revision']:raise ValueError('MSINet source revision changed')
    points=yaml.safe_load((ROOT/config['source_thresholds']).read_text());models={}
    for name in config['models']:
        spec=deepcopy(reference['models'][name])
        if sha256(ROOT/spec['checkpoint'])!=spec['sha256']:raise ValueError('Frozen model weights changed')
        spec['point']={key:points[name][key] for key in ['threshold','margin']};models[name]=spec
    scenes={};inputs={};videos=[];raw_detections=0
    for scene in config['scenes']:
        timestamp=root/'cam_timestamp'/f'{scene}.txt';offsets={cam:float(value) for cam,value in (line.split() for line in timestamp.read_text().splitlines())}
        inputs[str(timestamp)]=sha256(timestamp);cameras={}
        for cam in sorted((root/config['split']/scene).glob('c*')):
            video=cam/'vdo.avi';cap=capture(video)
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
            size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))];cap.release()
            for path in [cam/config['detections']['file'],cam/'roi.jpg',cam/'gt/gt.txt']:inputs[str(path)]=sha256(path)
            raw_detections+=sum(1 for _ in (cam/config['detections']['file']).open())
            videos.append((str(video),video.stat().st_size,video.stat().st_mtime_ns))
            cameras[cam.name]=dict(path=str(video),frames=min(frames,config['check_frames']) if config.get('check_frames') else frames,
                full_frames=frames,fps=fps,size=size,offset=offsets[cam.name],roi=str(cam/'roi.jpg'),detections=str(cam/config['detections']['file']))
        if not cameras or set(cameras)!=set(offsets):raise ValueError('Missing cameras in scene')
        scenes[scene]=cameras
    for path in [root/'eval/eval.py',root/'eval/ground_truth_validation.txt']:inputs[str(path)]=sha256(path)
    code=[*Path(__file__).parent.glob('*.py'),ROOT/'utils/full_frame/tracker.py',ROOT/'utils/experiments/vehicle_reference/inference.py',
          ROOT/'utils/reid_teacher/model.py',ROOT/'utils/reid_teacher/data.py',ROOT/'utils/benchmarking/msinet_baseline.py']
    selection=dict(config=config,models=models,scenes=scenes,inputs=inputs,videos=videos,
        reference_sha256=sha256(ROOT/config['reference_selection']),thresholds_sha256=sha256(ROOT/config['source_thresholds']),
        base_config_sha256=sha256(ROOT/config['base_config']),
        implementation={str(path.relative_to(ROOT)):sha256(path) for path in sorted(code)},raw_detection_rows=raw_detections,
        frames=sum(s['frames'] for scene in scenes.values() for s in scene.values()),
        scope='All official validation scenes/cameras/frames; supplied YOLOv3 detections; appearance/motion baseline; official MTMC ID metrics',
        limitations=['Ground-camera CityFlowV2, not aerial performance',
            'Provided detections, not the joint detector or a live detector speed measurement',
            'Source-fitted thresholds unchanged; no target training, threshold selection or offline future-frame clustering',
            'Local validation score, not an official test leaderboard submission',
            'Models use different external training histories; joint model excluded because it used AICity'])
    path=ROOT/config['output']/'selection.pt'
    if path.exists():
        if load(path)!=selection:raise ValueError('CityFlow frozen inputs changed; use new state/output paths')
    else:save(selection,path)
    (path.parent/'protocol.yaml').write_text(yaml.safe_dump(selection))
    return selection


def detections(spec,config):
    rows=np.loadtxt(spec['detections'],delimiter=',',ndmin=2);roi=cv2.imread(spec['roi'],cv2.IMREAD_GRAYSCALE)
    if roi is None or list(roi.shape[::-1])!=spec['size']:raise ValueError('ROI size differs from source video')
    if np.any(np.diff(rows[:,0])<0):raise ValueError('Detection rows must be sorted by frame')
    values,starts,counts=np.unique(rows[:,0].astype(int),return_index=True,return_counts=True)
    grouped={int(frame):rows[start:start+count] for frame,start,count in zip(values,starts,counts)}
    result={};width,height=spec['size']
    for frame in range(1,spec['frames']+1):
        selected=grouped.get(frame,np.empty((0,10)));selected=selected[selected[:,6]>=config['detections']['minimum_score']]
        boxes=selected[:,2:6].copy();boxes[:,2:]+=boxes[:,:2]
        boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,width-1);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,height-1)
        boxes=np.round(boxes).astype(np.float32);scores=selected[:,6].astype(np.float32)
        keep=(boxes[:,2:]-boxes[:,:2]).min(1)>=config['detections']['minimum_side']
        boxes,scores=boxes[keep],scores[keep]
        inside=[]
        for box in boxes:
            x1,y1,x2,y2=box.astype(int);inside.append(bool(min(roi[y1,x1],roi[y2,x1],roi[y1,x2],roi[y2,x2])==255))
        inside=np.asarray(inside,dtype=bool);boxes,scores=boxes[inside],scores[inside]
        if len(boxes):
            keep=nms(torch.from_numpy(boxes),torch.from_numpy(scores),config['detections']['nms_iou']).numpy();boxes,scores=boxes[keep],scores[keep]
        quality=[]
        for index,box in enumerate(boxes):
            lo=np.maximum(box[:2],boxes[:,:2]);hi=np.minimum(box[2:],boxes[:,2:]);inter=np.maximum(hi-lo,0).prod(1)
            inter[index]=0;overlap=float(inter.max())/float(np.prod(box[2:]-box[:2]))
            quality.append(bool(scores[index]>=config['quality']['minimum_score'] and min(box[2:]-box[:2])>=config['quality']['minimum_side'] and overlap<=config['quality']['maximum_overlap_fraction']))
        result[frame]=(boxes,scores,quality)
    return result
