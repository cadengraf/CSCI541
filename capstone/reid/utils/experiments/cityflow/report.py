"""Official full-split scorecard and a bounded multi-model video review."""
from pathlib import Path
import cv2
import numpy as np
import yaml
from utils.tracking.common import ROOT,load
from .data import capture


def summarize(config):
    output=ROOT/config['output'];selection=load(output/'selection.pt');reports=[]
    lines=['# CityFlowV2 complete validation','',
        ('INTEGRATION CHECK ONLY: first 50 frames per camera. Not benchmark accuracy. ' if config.get('check_frames') else '')+
        f'{len(selection["scenes"])} scenes, {sum(len(c) for c in selection["scenes"].values())} cameras, {selection["frames"]:,} frames. '
        'Official supplied validation labels and evaluator; provided YOLOv3 detections, identical causal association policy, source-fitted model-specific thresholds.','',
        '| Encoder | Official IDF1 | ID precision | ID recall | Correct sparse claims | False sparse claims | Unscored sparse claims |','|---|---:|---:|---:|---:|---:|---:|']
    for name in config['models']:
        path=output/name/'report.yaml'
        if not path.exists():continue
        r=yaml.safe_load(path.read_text());reports.append(r);m=r['metrics'];a=r['claim_audit']['counts']
        lines.append(f'| {name} | {100*m["idf1"]:.2f} | {100*m["idp"]:.2f} | {100*m["idr"]:.2f} | {a["correct"]} | {a["false"]} | {a["unscored"]} |')
    lines+=['','Official evaluation removes predictions outside ROIs, identities appearing in only one camera, and duplicate camera/identity/frame rows. '
        'Sparse claim diagnostics have a different denominator and retain unscored claims. They do not replace official IDF1.','',
        'No training, target threshold fitting, ground-truth boxes in inference, homography, camera-link topology, offline clustering or retrospective identity rewriting. '
        'The association implementation is a simple causal appearance baseline, not a reproduction of challenge-winning tracking systems.','',
        'This is a local score on the complete official validation split, not an official test leaderboard score. '
        'Training histories differ and include external data. The joint model is excluded because its historical training used AICity.','',
        'The evaluator runs the supplied eval.py formulas and filtering, with its unused pytrec_eval import removed at AST load time and NumPy asfarray compatibility restored. '
        'The distributed source file stays unchanged. A zero-surviving-prediction edge case is explicitly reported as all GT missed.','',
        'All source videos/detections stay read-only in Downloads. Required resumable states and official prediction TXT files live under project data/metrics. '
        'Component timing in per-model reports includes input preparation, decode, embedding, association, initialization and checkpoint writes separately. '
        'Detector-generation cost is unavailable for the supplied detections; no full live FPS claim is made.','',
        'Final review video uses bounded windows from every camera; quantitative scoring uses all frames. '
        'CityFlow is ground-camera footage and does not establish aerial performance.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    (output/'report.yaml').write_text(yaml.safe_dump(dict(results=reports,complete_split=not bool(config.get('check_frames')),frames=selection['frames'],limitations=selection['limitations'])))


def render(config,selection):
    output=ROOT/config['output'];models=config['models'];predictions={};events={}
    for name in models:
        rows=np.loadtxt(output/name/'predictions.txt',ndmin=2)
        by_camera={}
        for camera in np.unique(rows[:,0]).astype(int):
            values=rows[rows[:,0]==camera];order=values[:,2].argsort(kind='stable');values=values[order]
            frames,starts,counts=np.unique(values[:,2].astype(int),return_index=True,return_counts=True)
            by_camera[camera]={int(f):values[s:s+n] for f,s,n in zip(frames,starts,counts)}
        predictions[name]=by_camera;events[name]=load(output/name/'diagnostics.pt')['events']
    width=640;height=360;banner=100;temporary=output/'review.partial.mp4';destination=output/'review.mp4'
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),config['review']['output_fps'],(width*len(models),height+banner))
    if not writer.isOpened():raise ValueError('Cannot create CityFlow review video')
    def text(canvas,value,x,y,scale=.5):cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(235,235,235),1,cv2.LINE_AA)
    frames_written=0
    try:
        for scene,cameras in selection['scenes'].items():
            for camera,spec in cameras.items():
                claims=[e for e in events[models[0]] if e['camera']==camera and e['action']=='cross_camera']
                anchor=claims[0]['frame'] if claims else 1;length=min(spec['frames'],round(config['review']['seconds_per_camera']*spec['fps']))
                start=max(1,min(anchor-length//2,spec['frames']-length+1));cap=capture(spec['path'],start)
                try:
                    for frame in range(start,start+length):
                        okay,image=cap.read()
                        if not okay:raise ValueError('Review source decode failed')
                        canvas=np.full((height+banner,width*len(models),3),24,np.uint8)
                        text(canvas,f'CityFlowV2 validation | {scene}/{camera} frame {frame} | supplied YOLOv3 boxes | review windows only',12,24,.65)
                        for panel,name in enumerate(models):
                            view=image.copy();rows=predictions[name].get(int(camera[1:]),{}).get(frame,[])
                            for row in rows:
                                identity=int(row[1]);x,y,w,h=map(int,row[3:7]);known=identity<1_000_000_000
                                color=((identity*67)%180+60,(identity*97)%180+60,(identity*31)%180+60) if known else (160,160,160)
                                cv2.rectangle(view,(x,y),(x+w,y+h),color,2);cv2.putText(view,str(identity) if known else '?',(x,max(16,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
                            h,w=view.shape[:2];scale=min(width/w,height/h);view=cv2.resize(view,(int(w*scale),int(h*scale)));h,w=view.shape[:2]
                            x=panel*width+(width-w)//2;y=banner+(height-h)//2;canvas[y:y+h,x:x+w]=view
                            text(canvas,name,panel*width+12,57,.65)
                        text(canvas,'All metrics use complete split; displayed windows are not an accuracy sample. Ground-camera data; no masks.',12,85,.55)
                        writer.write(canvas);frames_written+=1
                finally:cap.release()
    finally:writer.release()
    temporary.replace(destination);cap=cv2.VideoCapture(str(destination));okay,_=cap.read();count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if not okay or count!=frames_written:raise ValueError('CityFlow review verification failed')
