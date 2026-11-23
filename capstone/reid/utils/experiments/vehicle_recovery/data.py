"""Freeze GT-box inputs and explicit synthetic blackouts before inference."""
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256


def reliable(boxes,size,spec):
    width,height=size;result=[]
    for i,box in enumerate(boxes):
        x1,y1,x2,y2=box;area=(x2-x1)*(y2-y1);overlap=0.
        for j,other in enumerate(boxes):
            if i==j:continue
            intersection=np.maximum(0,np.minimum(box[2:],other[2:])-np.maximum(box[:2],other[:2])).prod()
            overlap=max(overlap,float(intersection/area))
        result.append(bool(min(x2-x1,y2-y1)>=spec['trusted_minimum_side'] and
            min(x1,y1,width-x2,height-y2)>=spec['border'] and overlap<=spec['maximum_overlap_fraction']))
    return result


def freeze(config):
    root=ROOT/config['dataset_root'];selection=yaml.safe_load((ROOT/config['selection']).read_text())
    candidate=selection['candidate']
    if sha256(ROOT/candidate['checkpoint'])!=candidate['checkpoint_sha256']:raise ValueError('Reference encoder changed')
    cases={};sources={};frame_metadata=[]
    for entry in config['cases']:
        sequence=entry['sequence'];annotation=root/'annotations'/f'{sequence}.txt'
        sources[str(annotation.relative_to(ROOT))]=sha256(annotation)
        paths=sorted((root/'sequences'/sequence).glob('*.jpg'))
        if not paths or [int(p.stem) for p in paths]!=list(range(1,len(paths)+1)):raise ValueError('Noncontiguous sequence')
        with Image.open(paths[0]) as image:size=image.size
        rows={i:dict(frame=i,path=str(path.resolve()),truth=[],observations=[]) for i,path in enumerate(paths,1)}
        for fields in csv.reader(annotation.open()):
            frame,identity,x,y,w,h,score,category,truncation,occlusion=[int(float(v)) for v in fields[:10]]
            if score!=1 or identity<0 or category not in config['observations']['classes']:continue
            rows[frame]['truth'].append(identity)
            box=[max(0,x),max(0,y),min(size[0],x+w),min(size[1],y+h)]
            if min(box[2]-box[0],box[3]-box[1])<config['observations']['minimum_side']:continue
            rows[frame]['observations'].append(dict(truth=identity,box=box,category=category,occlusion=occlusion,truncation=truncation))
        target_frames=[frame for frame,row in rows.items() if entry['target'] in [o['truth'] for o in row['observations']]]
        if not target_frames:raise ValueError('Target has no valid observations')
        begin=target_frames[0]+config['blackout']['enrollment_frames'];end=begin+config['blackout']['duration_frames']
        if not any(frame>=end for frame in target_frames):raise ValueError('Target never returns after fixed blackout')
        for row in rows.values():
            row['observations'].sort(key=lambda o:tuple(o['box'])) # No identity labels determine association order.
            boxes=np.asarray([o['box'] for o in row['observations']],dtype=np.float32).reshape(-1,4)
            row['reliable']=reliable(boxes,size,config['observations'])
            if len(set(row['truth']))!=len(row['truth']):raise ValueError('Duplicate GT identity in frame')
        if sum(o['truth']==entry['target'] and good for frame,row in rows.items() if frame<begin
               for o,good in zip(row['observations'],row['reliable']))<8:raise ValueError('Insufficient target enrollment evidence')
        cases[sequence]=dict(target=entry['target'],size=list(size),blackout=[begin,end],frames=list(rows.values()))
        frame_metadata.extend((str(path.relative_to(ROOT)),path.stat().st_size,path.stat().st_mtime_ns) for path in paths)
    files=[*Path(__file__).parent.glob('*.py')]
    files+=[ROOT/'utils/vehicle_reid'/name for name in ['models.py','association.py','tracklets.py','reference.py','aggregation.py']]
    files+=[ROOT/'utils/full_frame/tracker.py',ROOT/'utils/reid_teacher/model.py',ROOT/'utils/reid_teacher/data.py',ROOT/'utils/reid_teacher/evaluation.py']
    signed=dict(config=config,selection=selection,cases=cases,annotations=sources,frames=frame_metadata,
        implementation={str(path.relative_to(ROOT)):sha256(path) for path in sorted(files)},
        configs={name:sha256(ROOT/config[name]) for name in ['reference_config','candidate_config']},
        protocol='Frozen natural visibility and injected target-observation blackout, current/past frames only',
        limitations=['GT vehicle boxes isolate appearance/association; no detector or masks run',
            'Box-only reliability heuristics; GT occlusion labels are evaluation-only',
            'Controlled blackout removes observations, not pixels; not natural disappearance or a camera change',
            'VisDrone already observed; no target threshold fitting, training or automatic deployment promotion',
            'Three videos / three synthetic target returns are diagnostic, not proof of population reliability'])
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True);path=output/'selection.pt'
    if path.exists():
        if load(path)!=signed:raise ValueError('Frozen video recovery inputs changed; use new output/state paths')
    else:save(signed,path)
    protocol={key:value for key,value in signed.items() if key not in ['cases','frames']}
    protocol['cases']={name:dict(target=case['target'],frames=len(case['frames']),blackout=case['blackout']) for name,case in cases.items()}
    (output/'protocol.yaml').write_text(yaml.safe_dump(protocol));return signed


def visible_indices(case,row,scenario):
    hidden=scenario=='target_blackout' and case['blackout'][0]<=row['frame']<case['blackout'][1]
    return [i for i,o in enumerate(row['observations']) if not (hidden and o['truth']==case['target'])],hidden
