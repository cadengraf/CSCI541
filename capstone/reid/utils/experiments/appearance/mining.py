"""Frozen train-only cross-camera prototypes and fixed confusing-ID neighbors."""
from collections import defaultdict
import random
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from utils.tracking.common import ROOT,load,save
from utils.reid_teacher.model import build
from utils.reid_teacher.data import transform


def records_for_mining(records,settings):
    grouped=defaultdict(lambda:defaultdict(list))
    for row in records:grouped[row['id']][row['camera']].append(row)
    rng=random.Random(settings['seed']);chosen=[]
    for identity in sorted(grouped):
        cameras=grouped[identity]
        if len(cameras)!=2:raise ValueError('Mining currently requires exactly two training cameras')
        for camera in sorted(cameras):
            pool=sorted(cameras[camera],key=lambda row:row['path'])
            chosen.extend(rng.sample(pool,min(len(pool),settings['mining']['images_per_camera'])))
    return chosen


def review(neighbors,records,output):
    # A short train-only diagnostic video, not a new training-label source.
    paths={}
    for row in records:paths.setdefault(row['id'],row['path'])
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),3,(960,380))
    if not writer.isOpened():raise RuntimeError('Cannot create hard-negative review video')
    try:
        for identity in list(neighbors)[:30]:
            canvas=np.full((380,960,3),25,dtype=np.uint8)
            for column,other in enumerate([identity,*neighbors[identity][:2]]):
                frame=cv2.imread(paths[other])
                frame=cv2.resize(frame,(300,300))
                canvas[60:360,column*320+10:column*320+310]=frame
                label=('anchor ' if column==0 else 'different labeled ID ')+str(other)
                cv2.putText(canvas,label,(column*320+10,35),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1)
            for _ in range(6):writer.write(canvas)
    finally:writer.release()


def run(experiment,settings,data,signature):
    destination=ROOT/experiment['state_root']/'mining.pt'
    previous=load(destination) if destination.exists() else None
    if previous is not None and previous['signature']!=signature:
        raise ValueError('Mining inputs/code changed; use a new state root')
    if previous is not None and previous['complete']:
        print('Reusing completed train-only neighbor mining',flush=True)
        return previous['neighbors']
    records=records_for_mining(data['train'],experiment)
    ids=sorted({row['id'] for row in data['train']})
    model=build(settings,len(ids),pretrained=False).cuda().eval()
    saved=load(ROOT/experiment['start_checkpoint'])
    if saved['identity_mapping']!={identity:i for i,identity in enumerate(ids)}:
        raise ValueError('Warm-start classifier identity mapping differs')
    model.load_state_dict(saved['model']);del saved
    conversion=transform(settings['model']['image_size'])
    chunks=[] if previous is None else previous['chunks']
    begin=sum(len(chunk) for chunk in chunks)
    batch=settings['evaluation']['batch_size']
    with torch.inference_mode():
        for start in range(begin,len(records),batch):
            images=[]
            for row in records[start:start+batch]:
                with Image.open(row['path']) as image:images.append(conversion(image.convert('RGB')))
            vectors=F.normalize(model(torch.stack(images).cuda()).float(),dim=1)
            chunks.append(vectors.cpu().half())
            if (start//batch+1)%experiment['mining']['checkpoint_batches']==0 or start+batch>=len(records):
                save(dict(signature=signature,chunks=chunks,complete=False),destination)
                print(f'Train-only mining: {min(start+batch,len(records))}/{len(records)} crops',flush=True)
    vectors=torch.cat(chunks).float()
    grouped=defaultdict(lambda:defaultdict(list))
    for index,row in enumerate(records):grouped[row['id']][row['camera']].append(index)
    prototypes=[]
    for identity in ids:
        prototypes.append(torch.stack([F.normalize(vectors[indices].mean(0),dim=0) for camera,indices in sorted(grouped[identity].items())]))
    prototypes=torch.stack(prototypes).cuda()
    # Symmetric cross-camera similarity; no calibration/final identity is mined.
    similarity=(prototypes[:,0]@prototypes[:,1].T+prototypes[:,1]@prototypes[:,0].T)/2
    similarity.fill_diagonal_(-float('inf'))
    count=min(experiment['mining']['top_k'],len(ids)-1)
    values,columns=similarity.topk(count,dim=1)
    neighbors={identity:[ids[column] for column in columns[i].cpu().tolist()] for i,identity in enumerate(ids)}
    save(dict(signature=signature,complete=True,neighbors=neighbors,
              similarity=values.cpu(),crops=len(records),identities=len(ids)),destination)
    review(neighbors,records,ROOT/experiment['output']/'hard_negative_review.mp4')
    print(f'Mined {count} fixed neighbors for {len(ids)} training identities',flush=True)
    return neighbors
