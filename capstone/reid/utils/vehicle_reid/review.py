"""Causal replay with observational evidence capture and video-only case review."""
from pathlib import Path
import hashlib
import cv2
import numpy as np
import yaml
from utils.tracking.common import ROOT,load,save,decode
from utils.teacher.assets import sha256
from .tracklets import TrackletAssociation
from .recovery import Recovery
from .models import observations as quality_observations
from .budget import common_masks,metric_decisions,identity_digest


class EvidenceRecorder(TrackletAssociation):
    def __init__(self,spec):
        super().__init__(spec)
        self.reference_sources={}
        self.recent_sources={}

    def _enroll(self,local,box,feature,frame):
        identity=super()._enroll(local,box,feature,frame)
        self.reference_sources[identity]=dict(frame=frame,box=box.tolist())
        return identity

    def _touch(self,identity,box,feature,reliable,frame,recovery=False):
        result=super()._touch(identity,box,feature,reliable,frame,recovery)
        if result['memory_update']:
            items=self.recent_sources.setdefault(identity,[])
            items.append(dict(frame=frame,box=box.tolist()))
            self.recent_sources[identity]=items[-self.spec['recent_size']:]
        return result


def collect(config):
    pipeline=yaml.safe_load((ROOT/config['pipeline_config']).read_text())
    all_events=[]; checks={}
    for value in config['sequences']:
        seq=f'{value:04d}'
        bank=load(ROOT/config['features']/seq/'state.pt')
        truth=ROOT/config['data_root']/'instances_txt'/f'{seq}.txt'
        signature=dict(features=bank['signature'],settings=pipeline['association'],quality=pipeline['quality'],
            truth=sha256(truth),review_source=sha256(Path(__file__)),tracker_source=sha256(Path(__file__).with_name('tracklets.py')))
        path=ROOT/'data/vehicle_recovery_review/development'/f'{seq}.pt'
        actor=EvidenceRecorder(pipeline['association']); metric=Recovery(truth)
        origin_truth={}; histories={}; events=[]; rows=[]; begin=0
        if path.exists():
            old=load(path)
            if old['signature']!=signature: raise ValueError('Review input changed; use a new review state directory')
            if old['complete']:
                all_events.extend(old['events']);checks[seq]=old['matches_original'];continue
            actor,metric,origin_truth,histories,events,rows,begin=(old[k] for k in
                ['actor','metric','origin_truth','histories','events','rows','next_frame'])
        for frame in range(begin,len(bank['observations'])):
            obs=bank['observations'][frame]
            raw=decode(load(ROOT/config['cache']/seq/f'{frame:06d}.pt'))
            indices=obs['indices'].tolist(); height,width=raw['masks'].shape[-2:]
            _,_,_,reliable=quality_observations(np.empty((height,width,0),dtype=np.uint8),
                raw['boxes'][indices].numpy(),raw['scores'][indices].numpy(),raw['masks'][indices,0].bool().numpy(),pipeline['quality'])
            decisions=actor.update(obs['boxes'].numpy(),obs['embeddings'].numpy(),reliable,frame)
            masks,positions=common_masks(raw,indices,config['min_mask_pixels'])
            prepared=metric.prepare_frame(frame,masks)
            actual={positions[row]:identity for row,identity in prepared['matched'].items()}
            for column,decision in enumerate(decisions):
                local=decision['tracklet_id']; identity=decision['identity']
                history=histories.setdefault(local,[])
                history.append(dict(frame=frame,box=obs['boxes'][column].tolist(),reliable=bool(reliable[column])))
                histories[local]=history[-12:]
                if decision['status']=='new': origin_truth[identity]=actual.get(column)
                if decision['status']!='recovered': continue
                previous=metric.identity_truth.get(identity); current=actual.get(column)
                status='unscored' if previous is None or current is None else 'correct' if previous==current else 'false'
                source=actor.reference_sources[identity]
                item=actor.identities[identity]
                query=obs['embeddings'][column].numpy()
                recent_index=int(np.argmax([vector@query for vector in item.recent])) if item.recent else None
                recent=(actor.recent_sources.get(identity,[])[recent_index] if recent_index is not None else None)
                prototype_truth=origin_truth.get(identity)
                diagnosis=('local_identity_or_enrollment_contamination' if prototype_truth is not None and previous!=prototype_truth
                           else 'appearance_false_accept' if status=='false' else status)
                events.append(dict(sequence=seq,frame=frame,identity=int(identity),local_id=int(local),status=status,
                    expected_truth=previous,current_truth=current,reference_truth=prototype_truth,diagnosis=diagnosis,
                    reference=source,recent=recent,query_box=obs['boxes'][column].tolist(),
                    query_history=histories[local].copy(),score=float(decision['similarity']),margin=float(decision['margin']),
                    reference_cosine=float(item.references[0]@query),
                    recent_cosine=float(item.recent[recent_index]@query) if recent_index is not None else None))
            emitted=metric_decisions(decisions,positions)
            rows.append([int(item['identity']) for item in emitted]);metric.update(frame,masks,emitted,prepared=prepared)
            if (frame+1)%config['checkpoint_every']==0 or frame+1==len(bank['observations']):
                prior=load(ROOT/config['state_root']/'development'/f'{seq}.pt')
                expected=prior['summary']['s082_m008_h3']['identity_digest']
                complete=frame+1==len(bank['observations'])
                matched=identity_digest(rows)==expected if complete else None
                save(dict(signature=signature,actor=actor,metric=metric,origin_truth=origin_truth,histories=histories,
                    events=events,rows=rows,next_frame=frame+1,complete=complete,matches_original=matched),path)
        all_events.extend(events);checks[seq]=matched
    return all_events,checks


def fit(image,width,height):
    result=np.zeros((height,width,3),dtype=np.uint8)
    if image is None or not image.size:return result
    scale=min(width/image.shape[1],height/image.shape[0])
    resized=cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))))
    y=(height-resized.shape[0])//2;x=(width-resized.shape[1])//2
    result[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return result


def image_at(config,sequence,frame):
    return cv2.imread(str(ROOT/config['data_root']/'training/image_02'/sequence/f'{frame:06d}.png'))


def crop(image,box):
    x1,y1,x2,y2=map(int,box)
    return image[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)]


def text(image,line,x,y,color=(235,235,235),scale=.55):
    cv2.putText(image,line,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def render(config,events,output):
    video=output/'development_global_review.mp4'; temporary=output/'development_global_review.partial.mp4'
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),8,(1600,720))
    if not writer.isOpened():raise RuntimeError('Cannot create review video')
    try:
        for index,event in enumerate(events):
            reference=image_at(config,event['sequence'],event['reference']['frame'])
            recent=(image_at(config,event['sequence'],event['recent']['frame']) if event['recent'] else None)
            history=event['query_history']
            for step in range(len(history)+8):
                observation=history[min(step,len(history)-1)]
                current=image_at(config,event['sequence'],observation['frame'])
                canvas=np.full((720,1600,3),20,dtype=np.uint8)
                color=(50,70,245) if event['status']=='false' else (50,220,80)
                text(canvas,f"CASE {index+1}: {event['status'].upper()} | sequence {event['sequence']} | decision frame {event['frame']}",12,25,color,.7)
                text(canvas,f"ID {event['identity']} expected GT {event['expected_truth']} -> observed GT {event['current_truth']} | score {event['score']:.3f} | margin {event['margin']:.3f}",12,54)
                canvas[95:375,0:400]=fit(crop(reference,event['reference']['box']),400,280)
                text(canvas,f"Frozen enrollment | frame {event['reference']['frame']} | GT {event['reference_truth']}",10,88)
                canvas[420:700,0:400]=fit(reference,400,280)
                text(canvas,'Enrollment frame context',10,409)
                annotated=current.copy()
                x1,y1,x2,y2=map(int,observation['box']);cv2.rectangle(annotated,(x1,y1),(x2,y2),color,3)
                canvas[80:430,410:1600]=fit(annotated,1190,350)
                canvas[450:700,410:800]=fit(crop(current,observation['box']),390,250)
                text(canvas,'Incoming tracklet crop',420,446)
                if recent is not None:
                    canvas[450:700,810:1200]=fit(crop(recent,event['recent']['box']),390,250)
                    text(canvas,f"Best recent evidence | frame {event['recent']['frame']}",820,446)
                text(canvas,'Offline review; tracker used current/past frames only',1210,490,scale=.43)
                text(canvas,f"reference cosine: {event['reference_cosine']:.3f}",1210,520,scale=.43)
                text(canvas,event['diagnosis'].replace('_',' ')[:43],1210,550,scale=.38)
                writer.write(canvas)
                if step==len(history)-1:
                    cv2.imwrite(f'/tmp/vehicle_reid_review_case_{index+1}.png',canvas)
    finally:writer.release()
    temporary.replace(video)
    return video


def run():
    config=yaml.safe_load((ROOT/'configs/recovery_budget.yaml').read_text())
    output=ROOT/'metrics/vehicle_recovery_review';output.mkdir(parents=True,exist_ok=True)
    events,checks=collect(config)
    if not all(checks.values()):raise RuntimeError('Evidence recorder changed the original decision stream')
    video=render(config,events,output)
    report=dict(events=events,identical_development_streams=checks,video=str(video.relative_to(ROOT)),
        scope='Development-only visual review; no validation cases used for designing aggregation')
    (output/'cases.yaml').write_text(yaml.safe_dump(report))
    print(f'{len(events)} original global claims reviewed; video {video}',flush=True)


if __name__=='__main__':run()
