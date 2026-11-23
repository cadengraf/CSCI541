"""One gate scorecard and a bounded real-crop video review."""
import csv
import cv2
import numpy as np
import yaml
from utils.tracking.common import ROOT,load
from .data import image


def summarize(config):
    output=ROOT/config['output'];selection=load(output/'selection.pt');rows=[]
    lines=['# Milestone 1: fixed vehicle reference comparison','',
        'Same 340 known queries / 104 identities, 146 references and 622 total queries across six VisDrone videos. '
        'No target fitting; all models use the same VRAI fit identities and the same empirical 1% calibration budget. '
        'Different training data/recipes and native normalization. These are same-video GT-crop results, not tracking or cross-camera metrics.','',
        '| Model | Rank-1 | mAP | Correct known acceptance | Wrong known acceptance | Unknown FAR | Batch-1 median ms | Batch-32 median ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in config['models']:
        r=yaml.safe_load((output/f'{name}.yaml').read_text());p=r['open_set']['common_vrai_fit'];c={b['batch_size']:b for b in r['cost']['batches']}
        row=dict(model=name,rank1=r['retrieval']['rank1'],map=r['retrieval']['map'],known_acceptance=p['known_recovery'],
                 wrong_known=p['wrong_known_rate'],unknown_far=p['unknown_far'],batch1_ms=c[1]['median_ms'],batch32_ms=c[32]['median_ms'])
        rows.append(row);lines.append(f'| {name} | {row["rank1"]:.2%} | {row["map"]:.2%} | {row["known_acceptance"]:.2%} | {row["wrong_known"]:.2%} | {row["unknown_far"]:.2%} | {row["batch1_ms"]:.2f} | {row["batch32_ms"]:.2f} |')
    lines+=['','Encoder timings use FP32 on preloaded 256-square inputs, including L2 normalization. '
        'They exclude detection, masks, crop I/O, association and rendering. Full online FPS is not measured here. '
        'Hardware, p95 latency, memory, loaded parameters and parameters in invoked modules are in per-model YAML.','',
        '## Training exposure','']
    for name,spec in selection['models'].items():lines.append(f'- **{name}:** {spec["training"]}. Preprocessing: {spec["preprocessing"]}.')
    lines+=['','The joint model is excluded because historical joint training used VisDrone. '
        'This comparison does not establish a fair unseen-data result for it.','',
        '## Gate decision','',
        'The bounded reference comparison is complete. Do not add further baseline architectures or start another training sweep at this gate. '
        'Retain the original TransReID as the existing video integration reference; treat MSINet and mixed AdaSP as measured alternatives. '
        'Choose any replacement through a controlled video-recovery comparison at comparable false-reconnection rates, rather than crop Rank-1 alone. '
        'No live tracker defaults or model selections were changed.','',
        'Next: freeze a small video recovery suite and evaluate current reference recovery, missed recoveries, false reconnections and confirmation delay. '
        'Address the largest diagnosed failure before changing the appearance recipe.','',
        'The target protocol was previously observed. Further tuning on it makes it development data. '
        'Small correlated track samples and different training exposure limit architecture conclusions. '
        'Per-video and natural-versus-withheld unknown results are in per-model YAML.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    with (output/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'gate.yaml').write_text(yaml.safe_dump(dict(milestone=1,status='complete',
        scope='Three frozen appearance references on an identical target protocol and common source calibration',
        next='Controlled video recovery evaluation',integration_reference='original',deployment_changed=False,
        stop_rule='No more baseline architectures or appearance training sweeps before video recovery diagnosis')))


def render(config):
    output=ROOT/config['output'];selection=load(output/'selection.pt');names=config['models']
    states={name:load(ROOT/config['state_root']/name/'state.pt') for name in names}
    decisions={name:{r['query']:r for r in state['decisions']} for name,state in states.items()}
    data={group:selection['data']['target_'+group] for group in ['query','gallery']}
    buckets={name:[] for name in ['disagreement','false_match','missed_known','unknown','correct']}
    for index,row in enumerate(data['query']):
        labels=[]
        for name in names:
            r=decisions[name][index];p=states[name]['operating_point'];accepted=r['similarity']>=p['threshold'] and r['margin']>=p['margin']
            labels.append('CORRECT' if accepted and r['known'] and r['correct'] else 'FALSE MATCH' if accepted else 'MISSED KNOWN' if r['known'] else 'UNKNOWN REJECTED')
        category='disagreement' if len(set(labels))>1 else 'false_match' if labels[0]=='FALSE MATCH' else 'missed_known' if labels[0]=='MISSED KNOWN' else 'unknown' if labels[0]=='UNKNOWN REJECTED' else 'correct'
        buckets[category].append((index,labels))
    selected=[]
    for offset in range(config['review']['maximum_examples']):
        for bucket in buckets.values():
            if offset<len(bucket):selected.append(bucket[offset])
            if len(selected)>=config['review']['maximum_examples']:break
        if len(selected)>=config['review']['maximum_examples']:break
    path=output/'review.mp4';temporary=output/'review.partial.mp4'
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),2,(1440,600))
    if not writer.isOpened():raise RuntimeError('Cannot open video writer')
    def text(canvas,value,x,y,color=(235,235,235),scale=.5):cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    def tile(canvas,row,column):
        value=cv2.cvtColor(np.asarray(image(row)),cv2.COLOR_RGB2BGR);h,w=value.shape[:2];scale=min(320/w,300/h)
        value=cv2.resize(value,(max(1,int(w*scale)),max(1,int(h*scale))));h,w=value.shape[:2]
        x=column*360+(360-w)//2;y=170+(300-h)//2;canvas[y:y+h,x:x+w]=value
    try:
        for index,labels in selected:
            canvas=np.full((600,1440,3),24,np.uint8);q=data['query'][index]
            text(canvas,'FROZEN VEHICLE REFERENCES | Same crops and source calibration | GT-box appearance review',20,30,scale=.7)
            text(canvas,q['sequence']+f' | later query frame {q["frame"]}',20,60,scale=.6)
            text(canvas,'QUERY ID '+q['id'].split(':')[-1],20,115,scale=.6);tile(canvas,q,0)
            for column,(name,label) in enumerate(zip(names,labels),1):
                r=decisions[name][index];reference=data['gallery'][r['open_prediction']];x=column*360+20
                color=(60,210,60) if label in ['CORRECT','UNKNOWN REJECTED'] else (60,60,240) if label=='FALSE MATCH' else (30,210,240)
                text(canvas,name,x,115,scale=.6);text(canvas,label,x,140,color,scale=.6);tile(canvas,reference,column)
                text(canvas,f'earlier ID {reference["id"].split(":")[-1]} frame {reference["frame"]}',x,500)
                text(canvas,f'score {r["similarity"]:.3f} margin {r["margin"]:.3f}',x,525)
            text(canvas,'Curated error examples, not representative accuracy. Full metrics use all selected queries.',20,580,scale=.6)
            for _ in range(4):writer.write(canvas)
    finally:writer.release()
    temporary.replace(path)
    capture=cv2.VideoCapture(str(path));okay,frame=capture.read();capture.release()
    if not okay or frame.shape[:2]!=(600,1440):raise ValueError('Reference video cannot be decoded')
