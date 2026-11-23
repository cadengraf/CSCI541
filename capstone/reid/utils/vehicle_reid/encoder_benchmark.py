"""Fixed-threshold paired benchmark of original and degradation-trained encoders."""
import argparse
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from pycocotools import mask as rle
import torch
import yaml
from utils.tracking.common import ROOT,load,save,decode
from utils.teacher.assets import sha256
from utils.full_frame.protocol import VALIDATION
from utils.full_frame.mots_evaluation import evaluate,TRACKEVAL_REVISION
from .appearance_encoder import export
from .extraction import extract
from .models import observations as quality_observations
from .tracklets import build
from .reference import ReferenceAssociation
from .recovery import Recovery
from .size_audit import SizeAudit,return_sizes
from .budget import common_masks,metric_decisions,identity_digest,aggregate
from .benchmark import paired_intervals

METHODS=['original_reference','degradation_reference','original_query_pool','degradation_query_pool']


def prepare(config):
    if config['protocol']!='kitti_mots_standard_val_v1' or config['sequences']!=list(VALIDATION):
        raise ValueError('Encoder comparison requires all nine validation sequences')
    pipeline=yaml.safe_load((ROOT/config['pipeline_config']).read_text())
    if pipeline['association']['engine']!='aggregated':raise ValueError('Expected frozen query-pooling settings')
    candidate=export(config['candidate_checkpoint'],config['candidate_selection'])
    changed=deepcopy(pipeline);changed['reid']['selection']=config['candidate_selection']
    changed['output']=config['live']['output']+'/degradation'
    path=ROOT/config['candidate_pipeline']
    if path.exists() and yaml.safe_load(path.read_text())!=changed:raise ValueError('Candidate pipeline changed')
    path.write_text(yaml.safe_dump(changed,sort_keys=False))
    extraction=deepcopy(config);extraction['pipeline_config']=config['candidate_pipeline']
    extract(extraction)
    return pipeline,candidate


def implementation():
    names=['encoder_benchmark.py','appearance_encoder.py','models.py','extraction.py','tracklets.py','aggregation.py',
           'reference.py','association.py','recovery.py','size_audit.py','budget.py']
    paths=[Path(__file__).parent/name for name in names]+[ROOT/'utils/full_frame/tracker.py',ROOT/'utils/full_frame/mots_evaluation.py']
    return {str(path.relative_to(ROOT)):sha256(path) for path in paths}


def sequence(config,pipeline,seq,provenance):
    original=load(ROOT/config['original_features']/seq/'state.pt')
    degradation=load(ROOT/config['features']/seq/'state.pt')
    if not original['complete'] or not degradation['complete']:raise ValueError('Incomplete feature banks')
    if len(original['observations'])!=len(degradation['observations']):raise ValueError('Feature bank lengths differ')
    truth=ROOT/config['data_root']/'instances_txt'/f'{seq}.txt'
    signed=dict(config=config,pipeline=pipeline,models=provenance,
        features=dict(original=original['signature'],degradation=degradation['signature']),
        truth_sha256=sha256(truth),implementation=implementation(),evaluator=TRACKEVAL_REVISION)
    destination=ROOT/config['state_root']/'association'/f'{seq}.pt'
    algorithms={name:(ReferenceAssociation(pipeline['association']) if name.endswith('_reference') else build(pipeline['association'])) for name in METHODS}
    metrics={name:Recovery(truth,config['recovery_window']) for name in METHODS}
    shared=metrics[METHODS[0]].truth
    for metric in metrics.values():metric.truth=shared
    sizes={name:SizeAudit(config['small_short_side']) for name in METHODS}
    seconds={name:0. for name in METHODS};rows={name:[] for name in METHODS}
    begin=0;offsets={name:0 for name in METHODS}
    if destination.exists():
        previous=load(destination)
        if previous['signature']!=signed:raise ValueError(f'Encoder benchmark inputs changed: {seq}; use new state/output paths')
        if previous['complete']:
            print(f'{seq}: reusing completed paired encoder audit',flush=True);return previous['summary']
        algorithms,metrics,sizes,seconds,rows,begin,offsets=(previous[key] for key in ['algorithms','metrics','sizes','seconds','rows','next_frame','offsets'])
    handles={}
    try:
        for name in METHODS:
            path=ROOT/config['state_root']/'predictions'/name/'data'/f'{seq}.txt';path.parent.mkdir(parents=True,exist_ok=True)
            if begin and (not path.exists() or path.stat().st_size<offsets[name]):raise ValueError('Committed MOTS output missing')
            handle=path.open('r+b' if begin else 'wb')
            if begin:handle.truncate(offsets[name]);handle.seek(offsets[name])
            handles[name]=handle
        for frame in range(begin,len(original['observations'])):
            left=original['observations'][frame];right=degradation['observations'][frame]
            if not torch.equal(left['indices'],right['indices']) or not torch.equal(left['boxes'],right['boxes']):
                raise ValueError(f'Common detection geometry differs at {seq}/{frame}')
            raw=decode(load(ROOT/config['cache']/seq/f'{frame:06d}.pt'))
            indices=left['indices'].tolist();height,width=raw['masks'].shape[-2:]
            _,_,_,reliable=quality_observations(np.empty((height,width,0),dtype=np.uint8),
                raw['boxes'][indices].numpy(),raw['scores'][indices].numpy(),raw['masks'][indices,0].bool().numpy(),pipeline['quality'])
            boxes=left['boxes'].numpy();decisions={}
            for name,algorithm in algorithms.items():
                bank=left if name.startswith('original_') else right
                start=time.perf_counter();decisions[name]=algorithm.update(boxes,bank['embeddings'].numpy(),reliable,frame)
                seconds[name]+=time.perf_counter()-start
            # All identity decisions precede GT matching and size diagnostics.
            masks,positions=common_masks(raw,indices,config['min_mask_pixels'])
            prepared=metrics[METHODS[0]].prepare_frame(frame,masks)
            encoded=[rle.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode() for mask in masks]
            for name,metric in metrics.items():
                emitted=metric_decisions(decisions[name],positions)
                if len({row['identity'] for row in emitted})!=len(emitted):raise ValueError('Duplicate output identity')
                sizes[name].update(boxes[positions],emitted,prepared,metric.identity_truth)
                metric.update(frame,masks,emitted,prepared=prepared)
                ids=[int(row['identity']) for row in emitted];rows[name].append(ids)
                for identity,counts in zip(ids,encoded):handles[name].write(f'{frame} {identity} 1 {height} {width} {counts}\n'.encode('ascii'))
            if (frame+1)%config['checkpoint_every']==0 or frame+1==len(original['observations']):
                for handle in handles.values():handle.flush()
                summary={name:dict(actions=metric.action_summary(),returns=metric.summary(),events=metric.events,
                    identity_digest=identity_digest(rows[name]),size_observations=sizes[name].summary(),
                    size_returns=return_sizes(metric,config['small_short_side']),association_seconds=seconds[name]) for name,metric in metrics.items()}
                save(dict(signature=signed,algorithms=algorithms,metrics=metrics,sizes=sizes,seconds=seconds,rows=rows,
                    next_frame=frame+1,offsets={name:handle.tell() for name,handle in handles.items()},
                    complete=frame+1==len(original['observations']),summary=summary),destination)
                print(f'Paired encoder association {seq}: {frame+1}/{len(original["observations"])}',flush=True)
    finally:
        for handle in handles.values():handle.close()
    return summary


def summarize_sizes(reports):
    result={name:dict(observations={group:Counter() for group in ['small','larger']},
                      returns={group:Counter() for group in ['small','larger']}) for name in METHODS}
    for entries in reports.values():
        for name,entry in entries.items():
            for group in ['small','larger']:
                result[name]['observations'][group].update(entry['size_observations'][group])
                result[name]['returns'][group].update(entry['size_returns'][group])
    return {name:{kind:{group:dict(counts) for group,counts in groups.items()} for kind,groups in entry.items()} for name,entry in result.items()}


def gpu_workload():
    try:
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
        processes=[]
        for line in output.splitlines():
            fields=[value.strip() for value in line.split(',')]
            if len(fields)>=3 and fields[0].isdigit() and int(fields[0])!=os.getpid():
                processes.append(dict(pid=int(fields[0]),name=fields[1],memory_mib=fields[2]))
        return dict(available=True,other_compute_processes=processes)
    except (FileNotFoundError,subprocess.CalledProcessError):return dict(available=False)


def live(config,pipeline):
    if not config['live']['enabled']:return {}
    output=ROOT/config['output'];result={}
    for name,selection in [('original',pipeline['reid']['selection']),('degradation',config['candidate_selection'])]:
        spec=deepcopy(pipeline);spec.update(source=config['live']['source'],frames=config['live']['frames'],
            output=config['live']['output']+'/'+name,device='auto')
        spec['reid']['selection']=selection
        path=output/f'live_{name}.yaml';path.write_text(yaml.safe_dump(spec,sort_keys=False))
        command=[sys.executable,str(ROOT/'demo.py'),'vehicle','--config',str(path)]
        if (ROOT/spec['output']/'state.pt').exists():command.append('--resume')
        if torch.cuda.is_available():torch.cuda.empty_cache()
        before=gpu_workload()
        print(f'Live end-to-end measurement: {name}; GPU workload {before}',flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
        result[name]=yaml.safe_load((ROOT/spec['output']/'summary.yaml').read_text())
        result[name]['gpu_workload_before']=before
        result[name]['gpu_workload_after']=gpu_workload()
        result[name]['isolated_gpu']=bool(before.get('available') and not before.get('other_compute_processes') and
            result[name]['gpu_workload_after'].get('available') and not result[name]['gpu_workload_after'].get('other_compute_processes'))
    videos=[str(ROOT/config['live']['output']/name/'annotated.mp4') for name in ['original','degradation']]
    subprocess.run(['ffmpeg','-loglevel','error','-y','-i',videos[0],'-i',videos[1],'-filter_complex','hstack=inputs=2',
        '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'comparison.mp4')],check=True)
    (output/'online_cost.yaml').write_text(yaml.safe_dump(result))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/vehicle_encoder_benchmark.yaml')
    args=parser.parse_args(argv);config=yaml.safe_load((ROOT/args.config).read_text())
    torch.set_num_threads(config['cpu_threads'])
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    pipeline,candidate=prepare(config)
    original=yaml.safe_load((ROOT/pipeline['reid']['selection']).read_text())
    provenance=dict(original=original['candidate'],degradation=candidate['candidate'])
    reports={};lengths={};equivalence={}
    for value in config['sequences']:
        seq=f'{value:04d}';reports[seq]=sequence(config,pipeline,seq,provenance)
        lengths[seq]=len(load(ROOT/config['original_features']/seq/'state.pt')['observations'])
        archived=load(ROOT/'data/vehicle_tracklet_evidence/evaluation'/f'{seq}.pt')['summary']
        equivalence[seq]={name:reports[seq][name]['identity_digest']==archived[old]['identity_digest']
            for name,old in [('original_reference','reference'),('original_query_pool','query_mean')]}
        if not all(equivalence[seq].values()):raise RuntimeError(f'Original encoder identity stream regression: {seq}')
    measured=aggregate(reports);sizes=summarize_sizes(reports)
    scores=evaluate(ROOT/config['data_root'],ROOT/config['state_root']/'predictions',METHODS,lengths,output/'tracking')
    intervals={}
    for original_name,candidate_name in [('original_reference','degradation_reference'),('original_query_pool','degradation_query_pool')]:
        intervals[candidate_name]=paired_intervals(dict(methods=[original_name,candidate_name],candidate=candidate_name,
            output=str(output/'tracking')))[0]
    comparison=dict(protocol=config['protocol'],frames=sum(lengths.values()),sequences=lengths,models=provenance,
        tracking=scores,recovery=measured,size=sizes,per_sequence=reports,original_identity_equivalence=equivalence,
        paired_intervals=intervals,thresholds_frozen=True,calibration_performed=False,default_changed=False,
        definitions=dict(small_observation=f'predicted crop short side < {config["small_short_side"]} original pixels',
            small_return=f'GT mask bounding-box short side < {config["small_short_side"]} original pixels at GT visibility return',
            missed_return='all visibility returns minus recovered minus censored; includes missing prior-reference cases'),
        limitations=['Within-video validation; not cross-camera retrieval or a hidden-test leaderboard',
            'Prior architecture work inspected these sequences; this is a development comparison',
            'Full tracking scores use common masks and causal IDs; size strata are separate action/return diagnostics',
            'No threshold or model selection occurs on these validation sequences'])
    (output/'report.yaml').write_text(yaml.safe_dump(comparison))
    costs=live(config,pipeline)
    lines=['# Fixed-threshold encoder benchmark','',
        'Same frozen Co-DINO masks, nine KITTI MOTS validation sequences, 2,981 frames. Only the ReID weights change within each pair. No thresholds were tuned.','',
        '| Encoder / association | HOTA | IDF1 | ID switches | Natural returns | Correct / false / unscored global claims |',
        '|---|---:|---:|---:|---:|---|']
    for row in scores:
        name=row['model'];item=measured[name];returns=item['natural_returns']
        lines.append(f"| {name} | {row['HOTA']:.3f} | {row['IDF1']:.3f} | {row['IDSW']} | {returns['recovered']}/{returns['visibility_returns']} | {item['global_correct']} / {item['global_false']} / {item['global_unscored']} |")
    lines+=['','## Small-object diagnostics','',
            '| Encoder / association | Small returns recovered | Small correct / false / unscored global claims |',
            '|---|---:|---|']
    for name,item in sizes.items():
        r=item['returns']['small'];o=item['observations']['small']
        lines.append(f"| {name} | {r.get('recovered',0)}/{r.get('visibility_returns',0)} | {o.get('global_correct',0)} / {o.get('global_false',0)} / {o.get('global_unscored',0)} |")
    lines+=['','## Live cost','']
    for name,item in costs.items():
        lines.append(f"- {name}: {item['wall_fps']:.2f} FPS on {item['hardware']}, {item['frames_this_invocation']} frames; initialization {item['initialization_seconds']:.2f} s separately. Includes all online components and video/checkpoint I/O. Isolated GPU: {item['isolated_gpu']}.")
    lines+=['','A short live sample is not dataset-wide throughput. Side-by-side video: `comparison.mp4` (original left, degradation right).',
            'All original identity streams reproduce their archived results exactly. The default encoder remains unchanged.','',
            'Small observations use predicted crop short side <48 px; small returns use GT mask box short side <48 px at return. These are not size-filtered HOTA scores.','',
            'Paired whole-sequence HOTA intervals (mean sequence differences, not pooled HOTA):']
    for name,item in intervals.items():lines.append(f"- {name}: {item['mean_sequence_HOTA_delta']:+.3f} points [{item['lower95']:+.3f}, {item['upper95']:+.3f}].")
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(yaml.safe_dump(dict(tracking=scores,recovery=measured,paired_intervals=intervals)),flush=True)


if __name__=='__main__':main()
