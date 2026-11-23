"""Score committed causal evidence streams with the existing TrackEval adapter."""
import csv
import numpy as np
import yaml
from pycocotools import mask as rle
from utils.tracking.common import ROOT,load,decode
from utils.full_frame.protocol import VALIDATION
from utils.full_frame.mots_evaluation import evaluate
from .budget import common_masks
from .benchmark import paired_intervals


def run(config_path='configs/tracklet_evidence.yaml'):
    config=yaml.safe_load((ROOT/config_path).read_text())
    output=ROOT/config['output']
    report=yaml.safe_load((output/'report.yaml').read_text())
    names=list(report['evaluation'])
    predictions=ROOT/config['state_root']/'predictions'
    lengths={}
    for value in VALIDATION:
        seq=f'{value:04d}'
        state=load(ROOT/config['state_root']/'evaluation'/f'{seq}.pt')
        if not state['complete']:raise ValueError(f'Incomplete evaluation {seq}')
        bank=load(ROOT/'data/vehicle_benchmark/specialist'/seq/'state.pt')
        lengths[seq]=len(bank['observations'])
        handles={}
        try:
            for name in names:
                path=predictions/name/'data'/f'{seq}.txt'
                path.parent.mkdir(parents=True,exist_ok=True)
                handles[name]=path.with_suffix('.partial').open('w')
            for frame,observation in enumerate(bank['observations']):
                raw=decode(load(ROOT/config['cache']/seq/f'{frame:06d}.pt'))
                masks,_=common_masks(raw,observation['indices'].tolist(),config['min_mask_pixels'])
                for name in names:
                    identities=state['rows'][name][frame]
                    if len(identities)!=len(masks):raise ValueError('Committed mask/identity mismatch')
                    if len(set(identities))!=len(identities):raise ValueError('Duplicate emitted identity')
                    for identity,mask in zip(identities,masks):
                        encoded=rle.encode(np.asfortranarray(mask.astype(np.uint8)))
                        h,w=mask.shape
                        handles[name].write(f"{frame} {identity} 1 {h} {w} {encoded['counts'].decode()}\n")
        finally:
            for handle in handles.values():handle.close()
        for name in names:
            path=predictions/name/'data'/f'{seq}.txt'
            path.with_suffix('.partial').replace(path)
        print(f'TrackEval export {seq}: {lengths[seq]} committed frames',flush=True)
    scores=evaluate(ROOT/config['data_root'],predictions,names,lengths,output/'tracking',print_results=False)
    intervals={}
    for candidate in names:
        if candidate=='reference':continue
        intervals[candidate]=paired_intervals(dict(methods=['reference',candidate],candidate=candidate,
                                                  output=str(output/'tracking')))[0]
    (output/'tracking'/'paired_intervals.yaml').write_text(yaml.safe_dump(intervals))
    with (output/'tracking'/'paired_intervals.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(next(iter(intervals.values()))))
        writer.writeheader();writer.writerows(intervals.values())
    lines=['','## Full within-video tracking metrics','',
           'Same frozen Co-DINO masks, nine validation sequences, causal IDs; higher HOTA/IDF1 is better.',
           '', '| Variant | HOTA | IDF1 | ID switches |','|---|---:|---:|---:|']
    for row in scores:lines.append(f"| {row['model']} | {row['HOTA']:.3f} | {row['IDF1']:.3f} | {row['IDSW']} |")
    lines+=['','Paired sequence HOTA intervals (mean sequence difference, not pooled HOTA):','']
    for name,item in intervals.items():
        lines.append(f"- {name}: {item['mean_sequence_HOTA_delta']:+.3f} points, 95% interval [{item['lower95']:+.3f}, {item['upper95']:+.3f}].")
    live_path=output/'live_integration.yaml'
    if live_path.exists():
        live=yaml.safe_load(live_path.read_text())
        lines+=['','## Live integration','',
            f"{live['frames_this_invocation']} real frames on {live['hardware']}: {live['wall_fps']:.2f} FPS. "
            f"Initialization: {live['initialization_seconds']:.2f} s separately. "
            'Includes all online components, checkpoint I/O, and video finalization. This is a short integration sample.',
            'Video: `data/videos/vehicle_query_pool/annotated.mp4`.']
    path=output/'summary.md'
    original=path.read_text().split('\n## Full within-video tracking metrics')[0]
    path.write_text(original+'\n'.join(lines)+'\n')
    print(yaml.safe_dump(dict(scores=scores,paired_intervals=intervals)),flush=True)


if __name__=='__main__':run()
