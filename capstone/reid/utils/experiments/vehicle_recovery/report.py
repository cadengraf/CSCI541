"""Recovery counts, explicit gate decision, and one annotated comparison video."""
from collections import Counter
import cv2
import numpy as np
import yaml
from utils.tracking.common import ROOT,load


def report(config,selection):
    output=ROOT/config['output'];per_case={};aggregate={};lines=['# Controlled aerial video recovery','',
        'Original TransReID; identical GT boxes and embeddings. Current local reference versus frozen query pooling (.82 score / .08 margin). '
        'No detector, masks, training, threshold search or automatic promotion. Natural and injected returns are separate.','',
        '| Scenario | Method | Return events | Recovered | Missed | Censored | No prior trusted ID | Mean confirmation delay (frames) | Correct global claims | False global claims |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scenario in config['scenarios']:
        aggregate[scenario]={}
        for name in ['reference','query_pool']:
            counts=Counter();events=[];seconds=0.
            for sequence in selection['cases']:
                state=load(ROOT/config['state_root']/'replay'/sequence/f'{scenario}.pt')
                if not state['complete']:raise ValueError('Cannot summarize incomplete recovery')
                summary=state['summary'][name];per_case.setdefault(sequence,{}).setdefault(scenario,{})[name]=summary
                counts.update(summary['counts']);seconds+=state['seconds'][name]
                kind='natural_annotation' if scenario=='natural' else 'controlled_blackout'
                events.extend(event for event in summary['events'] if event['kind']==kind)
            statuses=Counter(e['result'] for e in events);delays=[e['delay_frames'] for e in events if e['result']=='recovered']
            long=[e for e in events if e['long_gap']]
            r=dict(counts=dict(counts),events=len(events),recovered=statuses['recovered'],missed=statuses['missed'],censored=statuses['censored'],
                unenrolled=statuses['unenrolled'],mean_confirmation_delay_frames=float(np.mean(delays)) if delays else None,
                long_returns=dict(events=len(long),recovered=sum(e['result']=='recovered' for e in long),
                    missed=sum(e['result']=='missed' for e in long),censored=sum(e['result']=='censored' for e in long),unenrolled=sum(e['result']=='unenrolled' for e in long)),
                missed_reasons=dict(Counter(e['reason'] for e in events if e['result']=='missed')),association_seconds=seconds)
            aggregate[scenario][name]=r;delay=f'{r["mean_confirmation_delay_frames"]:.1f}' if delays else 'n/a'
            lines.append(f'| {scenario} | {name} | {r["events"]} | {r["recovered"]} | {r["missed"]} | {r["censored"]} | {r["unenrolled"]} | {delay} | {counts["global_correct"]} | {counts["global_false"]} |')
    reference_recovered=sum(aggregate[s]['reference']['long_returns']['recovered'] for s in aggregate)
    candidate_recovered=sum(aggregate[s]['query_pool']['long_returns']['recovered'] for s in aggregate)
    cand_counts=Counter();ref_counts=Counter()
    for scenario in aggregate:
        cand_counts.update(aggregate[scenario]['query_pool']['counts']);ref_counts.update(aggregate[scenario]['reference']['counts'])
    false=cand_counts['global_false']+cand_counts['local_false'];correct=cand_counts['global_correct']+cand_counts['local_correct']
    baseline_false=ref_counts['global_false']+ref_counts['local_false'];scored=false+correct
    checks=dict(improves_long_gap_recovery=candidate_recovered>reference_recovered,
        false_claim_count_not_worse=false<=baseline_false,
        enough_scored_reconnections=scored>=config['evaluation']['minimum_scored_reconnections'],
        empirical_false_rate_within_budget=bool(scored and false/scored<=config['evaluation']['max_false_reconnection_rate']),
        no_unscored_reconnections=cand_counts['global_unscored']+cand_counts['local_unscored']==0)
    gate=dict(milestone=3,status='diagnostic_pass' if all(checks.values()) else 'not_passed',checks=checks,
        reference_long_returns_recovered=reference_recovered,candidate_long_returns_recovered=candidate_recovered,
        candidate_false_reconnections=false,reference_false_reconnections=baseline_false,scored_reconnections=scored,
        automatic_promotion=False,deployment_changed=False,
        caution='Natural and controlled cases share frames and identities; repeated observations are correlated. Empirical budget is not a reliability guarantee.')
    (output/'gate.yaml').write_text(yaml.safe_dump(gate));(output/'report.yaml').write_text(yaml.safe_dump(dict(aggregate=aggregate,per_case=per_case,gate=gate)))
    lines+=['','Long gaps contain at least 16 absent frames, beyond the 15-frame local lifetime. Natural events in the injected replay are not pooled into the natural row. '
        'Unenrolled returns and censored windows are reported rather than silently counted as recoverable misses. Confirmation delay is measured from first return frame to first accepted original-ID claim.','',
        '## Gate','',f'Status: **{gate["status"]}**. No live configuration is changed.','']
    for key,value in checks.items():lines.append(f'- {key}: {value}')
    lines+=['','## Failure diagnosis','']
    for scenario,methods in aggregate.items():
        for name,r in methods.items():
            lines.append(f'- {scenario}/{name}: long-gap recovery {r["long_returns"]["recovered"]}/{r["long_returns"]["events"]}; missed reasons {r["missed_reasons"]}; selected-target false-match frames {r["counts"].get("target_false_match_frames",0)}.')
    banks=[load(ROOT/config['state_root']/'features'/sequence/'state.pt') for sequence in selection['cases']]
    lines+=['','## Cost and scope','',f'Shared feature extraction: {sum(s["seconds"] for s in banks):.2f} seconds over {sum(s["frames"] for s in banks)} frames; devices: {sorted({s["device"] for s in banks})}. '
        'Embedding features are computed once and reused for both causal methods/scenarios. Per-method association times are in report.yaml. '
        'Neither number is full online FPS: ground-truth discovery substitutes for detection, masks are absent, and rendering/checkpoint time is separate.','',
        'Video: `comparison.mp4`, original frames with GT-box observation overlays, selected-target correctness and explicit blackout labels. '
        'During an injected blackout the vehicle may remain visible in the pixels; only its observations are withheld from the tracker.','',*selection['limitations']]
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


def render(config,selection):
    output=ROOT/config['output'];width=config['video']['panel_width'];height=450;banner=100
    temporary=output/'comparison.partial.mp4';path=output/'comparison.mp4'
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),config['video']['fps'],(width*2,height+banner))
    if not writer.isOpened():raise RuntimeError('Cannot create recovery video')
    def text(canvas,value,x,y,color=(235,235,235),scale=.5):cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    frames=0
    try:
        for sequence,case in selection['cases'].items():
            for scenario in config['scenarios']:
                state=load(ROOT/config['state_root']/'replay'/sequence/f'{scenario}.pt')
                for index,row in enumerate(case['frames']):
                    image=cv2.imread(row['path'])
                    if image is None:raise ValueError('Cannot decode recovery source video frame')
                    record=state['rows'][index];canvas=np.full((height+banner,width*2,3),24,np.uint8)
                    for panel,name in enumerate(['reference','query_pool']):
                        view=image.copy();target_identity=state['summary'][name]['target_identity'];decisions=record['methods'][name]
                        claim=None
                        for position,decision in zip(record['indices'],decisions):
                            observation=row['observations'][position];box=observation['box'];actual=observation['truth'];identity=decision['identity']
                            claimed=target_identity is not None and identity==target_identity
                            color=(60,60,240) if claimed and actual!=case['target'] else (60,220,60) if claimed else (20,210,240) if actual==case['target'] else (160,160,160)
                            if claimed:claim=actual
                            x1,y1,x2,y2=box;cv2.rectangle(view,(x1,y1),(x2,y2),color,2)
                            label=f'{identity if identity is not None else "?"} {decision["status"]}'
                            cv2.putText(view,label,(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
                        if record['hidden']:
                            for observation in row['observations']:
                                if observation['truth']==case['target']:
                                    x1,y1,x2,y2=observation['box'];cv2.rectangle(view,(x1,y1),(x2,y2),(220,100,220),2)
                                    cv2.putText(view,'WITHHELD OBSERVATION',(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,100,220),1,cv2.LINE_AA)
                        h,w=view.shape[:2];scale=min(width/w,height/h);resized=cv2.resize(view,(int(w*scale),int(h*scale)))
                        h,w=resized.shape[:2];x=panel*width+(width-w)//2;y=banner+(height-h)//2;canvas[y:y+h,x:x+w]=resized
                        status='CORRECT TARGET' if claim==case['target'] else 'FALSE TARGET MATCH' if claim is not None else 'NO CONFIDENT TARGET'
                        text(canvas,f'{name} | {status}',panel*width+12,62,(60,220,60) if claim==case['target'] else (60,60,240) if claim is not None else (20,210,240))
                    text(canvas,f'{sequence} | {scenario} | frame {row["frame"]} | selected GT target {case["target"]}',12,25,scale=.6)
                    text(canvas,'GT BOXES / NO DETECTOR OR MASKS | '+('TARGET OBSERVATIONS WITHHELD; pixels unchanged' if record['hidden'] else 'Earlier evidence only; natural and injected cases are separate'),12,88)
                    writer.write(canvas);frames+=1
    finally:writer.release()
    temporary.replace(path);capture=cv2.VideoCapture(str(path));okay,frame=capture.read();count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release()
    if not okay or count!=frames:raise ValueError('Recovery video verification failed')
