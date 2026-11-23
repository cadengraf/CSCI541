"""One review video using actual query and retrieved reference crops."""
import cv2
import numpy as np
from utils.tracking.common import ROOT,load
from .data import crop


def render(config):
    selection=load(ROOT/config['output']/'selection.pt');data=selection['data'];names=config['review']['models']
    states={name:load(ROOT/config['state_root']/name/'state.pt') for name in names}
    decisions={name:{r['query']:r for r in state['decisions']} for name,state in states.items()}
    # Review a bounded mix of agreement, disagreement, natural unknown and occlusion cases.
    buckets={name:[] for name in ['disagreement','both_wrong','both_correct','natural_unknown','occluded']}
    for i,row in enumerate(data['query']):
        a,b=[decisions[name][i] for name in names]
        ca=a['closed_known'] and a['rank']==1;cb=b['closed_known'] and b['rank']==1
        bucket='natural_unknown' if a['natural_unknown'] else 'disagreement' if ca!=cb else 'both_correct' if ca else 'both_wrong'
        buckets[bucket].append(i)
        if row['occlusion']>0:buckets['occluded'].append(i)
    selected=[]
    for offset in range(config['review']['maximum_examples']):
        for group in buckets.values():
            if offset<len(group) and group[offset] not in selected:selected.append(group[offset])
            if len(selected)>=config['review']['maximum_examples']:break
        if len(selected)>=config['review']['maximum_examples']:break
    path=ROOT/config['output']/'review.mp4';temporary=path.with_name('review.partial.mp4')
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),2,(1200,600))
    if not writer.isOpened():raise RuntimeError('Cannot create transfer review video')
    def text(canvas,value,position,color=(235,235,235),size=.55):
        cv2.putText(canvas,value,position,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
    def tile(canvas,row,column):
        image=cv2.cvtColor(np.asarray(crop(row)),cv2.COLOR_RGB2BGR);h,w=image.shape[:2]
        scale=min(350/w,300/h);image=cv2.resize(image,(max(1,int(w*scale)),max(1,int(h*scale))))
        h,w=image.shape[:2];x=column*400+(400-w)//2;y=160+(300-h)//2;canvas[y:y+h,x:x+w]=image
    try:
        for i in selected:
            canvas=np.full((600,1200,3),24,np.uint8);q=data['query'][i]
            text(canvas,'UNSEEN VISDRONE | GT crops | same-video appearance review',(20,30),size=.7)
            text(canvas,q['sequence']+f' | query frame {q["frame"]} | occlusion {q["occlusion"]}',(20,58))
            text(canvas,'Query (ground truth)',(20,110));text(canvas,q['id'].split(':')[-1],(20,135));tile(canvas,q,0)
            for column,name in enumerate(names,1):
                d=decisions[name][i];reference=data['gallery'][d['open_prediction']]
                # Source VRU operating point for mixed model, source VRAI for original; fixed in config/protocol.
                source='vru_fit' if 'vru_fit' in selection['models'][name]['thresholds'] else 'vrai_fit'
                p=selection['models'][name]['thresholds'][source]
                accepted=d['similarity']>=p['threshold'] and d['margin']>=p['margin']
                label='CORRECT MATCH' if accepted and d['known'] and d['correct'] else 'FALSE MATCH' if accepted else 'MISSED KNOWN' if d['known'] else 'UNKNOWN REJECTED'
                color=(70,210,70) if label in ['CORRECT MATCH','UNKNOWN REJECTED'] else (60,60,240) if label=='FALSE MATCH' else (30,210,240)
                x=column*400+20;text(canvas,name,(x,110));text(canvas,label,(x,135),color);tile(canvas,reference,column)
                text(canvas,f'reference ID {reference["id"].split(":")[-1]} frame {reference["frame"]}',(x,490))
                text(canvas,f'score {d["similarity"]:.3f} margin {d["margin"]:.3f}',(x,515))
                text(canvas,f'{source}: {p["threshold"]:.2f} / {p["margin"]:.2f}',(x,540))
            text(canvas,'Curated error review, not a representative accuracy sample. Metrics use all selected queries.',(20,580))
            for _ in range(4):writer.write(canvas)
    finally:writer.release()
    temporary.replace(path)
    capture=cv2.VideoCapture(str(path));ok,frame=capture.read();capture.release()
    if not ok or frame.shape[:2]!=(600,1200):raise ValueError('Review video verification failed')
