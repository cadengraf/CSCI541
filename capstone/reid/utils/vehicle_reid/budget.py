"""Development-only global recovery calibration, followed by frozen assessment."""
import argparse
from copy import deepcopy
import csv
import hashlib
from itertools import product
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import yaml
from scipy.stats import beta
from utils.tracking.common import ROOT, load, save, decode
from utils.teacher.assets import sha256
from utils.full_frame.protocol import TRAIN, VALIDATION
from .tracklets import TrackletAssociation, tracking_identity
from .reference import ReferenceAssociation
from .models import observations as quality_observations
from .recovery import Recovery
from .extraction import extract


def validate_development(config):
    sequences = config['sequences']
    if not sequences or len(set(sequences))!=len(sequences) or not set(sequences)<=set(TRAIN):
        raise ValueError('Recovery calibration requires distinct training-partition sequences only')
    if set(sequences)&set(VALIDATION):
        raise ValueError('Validation sequences cannot select recovery thresholds')
    budget = config['budget']
    if not 0<=budget['max_false_reconnection_rate']<1 or budget['minimum_scored_accepts']<1:
        raise ValueError('Invalid false-reconnection budget/support requirement')


def prepare_detector(config_path, config):
    from utils.teacher.settings import read,resolve
    settings,root = read()
    # Completed cache reuse needs provenance verification, not model loading.
    from utils.tracking.common import read as read_cache_settings
    manifest=ROOT/config['cache']/'source.yaml'
    complete=all((ROOT/config['cache']/f'{seq:04d}'/f'{path.stem}.pt').is_file()
        for seq in config['sequences'] for path in (ROOT/config['data_root']/'training/image_02'/f'{seq:04d}').glob('*.png'))
    if complete and manifest.exists():
        provenance=yaml.safe_load(manifest.read_text())
        if (provenance['checkpoint_sha256']!=sha256(ROOT/settings['training']['best_checkpoint']) or
            provenance['model']!=settings['model'] or provenance['source']!=settings['source'] or
            provenance['min_score']!=read_cache_settings()['cache_min_score']):
            raise ValueError('Frozen development detector cache provenance changed')
        return
    environment = resolve(root,settings['environment'])
    if Path(sys.prefix).resolve()!=environment.resolve():
        python = environment/'bin/python'
        if not python.is_file():
            raise FileNotFoundError('The existing Co-DINO environment is missing')
        subprocess.run([str(python),str(ROOT/'benchmark.py'),'vehicle-budget',
                        '--config',str(config_path),'--prepare-only'],cwd=ROOT,check=True)
        return
    from utils.tracking.common import read as cache_settings
    if config['cache']!=cache_settings()['cache']:
        raise ValueError('Development detector cache must match the configured frozen teacher cache')
    from utils.tracking.cache import export
    export(development_sequences=config['sequences'])


def variants(config):
    result = dict(reference=None)
    for score,margin,hits in product(config['grid']['similarity'],config['grid']['margin'],config['grid']['hits']):
        name = f's{round(score*100):03d}_m{round(margin*100):03d}_h{hits}'
        result[name] = dict(recovery_similarity=float(score),recovery_margin=float(margin),recovery_hits=int(hits))
    return result


def common_masks(raw, indices, minimum):
    occupied = torch.zeros(raw['masks'].shape[-2:],dtype=torch.bool)
    masks,positions = [],[]
    for position in sorted(range(len(indices)),key=lambda column:-float(raw['scores'][indices[column]])):
        mask = raw['masks'][indices[position],0].bool() & ~occupied
        if int(mask.sum())<minimum:
            continue
        occupied |= mask
        masks.append(mask.numpy()); positions.append(position)
    return masks,positions


def metric_decisions(decisions,positions):
    result=[]
    for position in positions:
        decision=decisions[position]
        if decision['identity'] is None:
            decision={**decision,'identity':tracking_identity(decision),'global_identity':None}
        result.append(decision)
    return result


def identity_digest(rows):
    digest=hashlib.sha256()
    for frame,identities in enumerate(rows):
        digest.update(f'{frame}:{identities}\n'.encode())
    return digest.hexdigest()


def prior_digest(path,frames):
    if not path.exists():
        return None
    rows=[[] for _ in range(frames)]
    for line in path.read_text().splitlines():
        parts=line.split()
        rows[int(parts[0])].append(int(parts[1]))
    return identity_digest(rows)


def replay(config,pipeline,choices,phase,selection_signature=None):
    output={}
    source_hash={name:sha256(Path(__file__).parent/name) for name in
                 ['tracklets.py','reference.py','recovery.py','budget.py','aggregation.py','evidence_experiment.py','association.py','models.py']}
    source_hash['local_tracker']=sha256(ROOT/'utils/full_frame/tracker.py')
    for value in config['sequences']:
        sequence=f'{value:04d}'
        bank=load(ROOT/config['features']/sequence/'state.pt')
        if not bank['complete']:
            raise ValueError(f'Incomplete observations for {sequence}')
        truth=ROOT/config['data_root']/'instances_txt'/f'{sequence}.txt'
        signature=dict(config=config,pipeline={key:pipeline[key] for key in ['association','quality','reid']},
            choices=choices,features=bank['signature'],truth_sha256=sha256(truth),implementation=source_hash,
            selection=selection_signature)
        destination=ROOT/config['state_root']/phase/f'{sequence}.pt'
        algorithms={}
        metrics={}
        for name,override in choices.items():
            spec=deepcopy(pipeline['association'])
            if override is None:
                algorithms[name]=ReferenceAssociation(spec)
            else:
                spec.update(override)
                from .tracklets import build
                algorithms[name]=build(spec)
            metrics[name]=Recovery(truth,config['recovery_window'])
        # GT is immutable evaluation input; share storage, never pass it to a tracker.
        shared_truth=next(iter(metrics.values())).truth
        for metric in metrics.values(): metric.truth=shared_truth
        rows={name:[] for name in choices}
        begin=0
        if destination.exists():
            previous=load(destination)
            if previous['signature']!=signature:
                raise ValueError(f'{phase} inputs/code changed; choose a new state_root: {sequence}')
            if previous['complete']:
                output[sequence]=previous['summary']
                print(f'{phase} {sequence}: reusing complete action audit',flush=True)
                continue
            algorithms,metrics,rows,begin=(previous[key] for key in ['algorithms','metrics','rows','next_frame'])
        for frame in range(begin,len(bank['observations'])):
            observation=bank['observations'][frame]
            raw=decode(load(ROOT/config['cache']/sequence/f'{frame:06d}.pt'))
            indices=observation['indices'].tolist()
            height,width=raw['masks'].shape[-2:]
            _,_,_,reliable=quality_observations(np.empty((height,width,0),dtype=np.uint8),
                raw['boxes'][indices].numpy(),raw['scores'][indices].numpy(),
                raw['masks'][indices,0].bool().numpy(),pipeline['quality'])
            boxes=observation['boxes'].numpy(); features=observation['embeddings'].numpy()
            decisions={name:algorithm.update(boxes,features,reliable,frame) for name,algorithm in algorithms.items()}
            masks,positions=common_masks(raw,indices,config['min_mask_pixels'])
            prepared=next(iter(metrics.values())).prepare_frame(frame,masks)
            for name,metric in metrics.items():
                emitted=metric_decisions(decisions[name],positions)
                metric.update(frame,masks,emitted,prepared=prepared)
                rows[name].append([int(decision['identity']) for decision in emitted])
            if (frame+1)%config['checkpoint_every']==0 or frame+1==len(bank['observations']):
                summary={name:dict(actions=metric.action_summary(),returns=metric.summary(),
                                  identity_digest=identity_digest(rows[name])) for name,metric in metrics.items()}
                save(dict(signature=signature,algorithms=algorithms,metrics=metrics,rows=rows,next_frame=frame+1,
                    complete=frame+1==len(bank['observations']),summary=summary),destination)
                print(f'{phase} {sequence}: {frame+1}/{len(bank["observations"])} frames across {len(choices)} settings',flush=True)
        output[sequence]=summary
    return output


def aggregate(sequence_reports):
    from collections import Counter
    result={}
    for sequence,report in sequence_reports.items():
        for name,entry in report.items():
            item=result.setdefault(name,dict(counts=Counter(),returns=Counter()))
            for key,value in entry['actions'].items():
                if key in ['global_correct','global_false','global_unscored','global_scored','global_scored_identities',
                           'global_false_identities','local_correct','local_false','local_unscored','global_accepted']:
                    item['counts'][key]+=value
            for key,value in entry['returns'].items():
                if key in ['visibility_returns','eligible_returns','recovered','failed','censored']:
                    item['returns'][key]+=value
    for name,item in result.items():
        counts=dict(item['counts']); scored=counts.get('global_scored',0)
        n=counts.get('global_scored_identities',0); k=counts.get('global_false_identities',0)
        result[name]=dict(**counts,global_false_rate=counts.get('global_false',0)/scored if scored else None,
            identity_any_false_upper95=1. if n==0 or n==k else float(beta.ppf(.95,k+1,n-k)),
            natural_returns=dict(item['returns']))
    return result


def select_point(measured,choices,budget):
    eligible=[]
    for name,stats in measured.items():
        if choices[name] is None:
            continue
        supported=stats.get('global_scored',0)>=budget['minimum_scored_accepts']
        scorable=not budget['require_no_unscored_claims'] or stats.get('global_unscored',0)==0
        passes=supported and scorable and stats['global_false_rate']<=budget['max_false_reconnection_rate']
        stats['empirical_budget_pass']=bool(passes)
        stats['identity_risk_bound_within_budget']=bool(passes and stats['identity_any_false_upper95']<=budget['max_false_reconnection_rate'])
        if passes:
            eligible.append(name)
    if not eligible:
        return None
    return max(eligible,key=lambda name:(measured[name].get('global_correct',0),
        -measured[name].get('global_false',0),choices[name]['recovery_similarity'],choices[name]['recovery_margin']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/recovery_budget.yaml')
    parser.add_argument('--prepare-only',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    path=(ROOT/args.config).resolve()
    config=yaml.safe_load(path.read_text())
    validate_development(config)
    if any(not list((ROOT/config['data_root']/'training/image_02'/f'{seq:04d}').glob('*.png')) for seq in config['sequences']):
        raise FileNotFoundError('Development images missing')
    torch.set_num_threads(config['cpu_threads'])
    prepare_detector(path,config)
    if args.prepare_only:
        return
    extract(config)
    pipeline=yaml.safe_load((ROOT/config['pipeline_config']).read_text())
    choices=variants(config)
    development=replay(config,pipeline,choices,'development')
    measured=aggregate(development)
    selected=select_point(measured,choices,config['budget'])
    selection=dict(name=selected,settings=choices[selected] if selected else None,
        budget=config['budget'],development_sequences=config['sequences'],
        empirical_support=selected is not None,
        identity_risk_bound_supported=bool(selected and measured[selected]['identity_risk_bound_within_budget']),
        default_engine='reference',automatically_enable_global_recovery=False)
    output=ROOT/config['output']; output.mkdir(parents=True,exist_ok=True)
    (output/'selection.yaml').write_text(yaml.safe_dump(selection))
    (output/'development.yaml').write_text(yaml.safe_dump(dict(per_sequence=development,aggregate=measured)))
    fields=list(dict.fromkeys(key for item in measured.values() for key in item if key!='natural_returns'))
    with (output/'development.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['setting']+fields)
        writer.writeheader()
        for name,item in measured.items(): writer.writerow(dict(setting=name,**{k:v for k,v in item.items() if k!='natural_returns'}))
    # Freeze the selected development point before observing any validation action audit.
    evaluation_config=deepcopy(config)
    evaluation_config['sequences']=list(VALIDATION)
    evaluation_config['features']='data/vehicle_benchmark/specialist'
    evaluation_choices=dict(reference=None,original_experimental={key:pipeline['association'][key] for key in
        ['recovery_similarity','recovery_margin','recovery_hits']})
    if selected:
        evaluation_choices['development_selected']=choices[selected]
    evaluation=replay(evaluation_config,pipeline,evaluation_choices,'evaluation',selection_signature=selection)
    assessed=aggregate(evaluation)
    prior=ROOT/'data/vehicle_benchmark/tracklet_stable_predictions'
    comparisons={}
    for seq,entries in evaluation.items():
        frames=len(list((ROOT/config['data_root']/'training/image_02'/seq).glob('*.png')))
        comparisons[seq]={name:entries[name]['identity_digest']==prior_digest(prior/method/'data'/f'{seq}.txt',frames)
                          for name,method in [('reference','legacy_specialist'),('original_experimental','tracklet_specialist')]}
    report=dict(selection=selection,evaluation=assessed,prediction_identity_equivalence=comparisons,
        definitions=dict(global_false_rate='wrong scored accepted GLOBAL recoveries / scored accepted GLOBAL recoveries',
            unscored='accepted recovery without prior/current sufficient mask-IoU identity label; never counted as correct',
            local='reported separately; not part of global recovery error denominator'),
        limitations=['Development frames were used to train the detector; association calibration only, not independent detector validation',
            'Empirical budget selection is not a guaranteed held-out error bound',
            'Repeated recovery events are correlated; identity-level any-error upper bound is a conservative diagnostic, not a distributional guarantee',
            'No operating point with insufficient support is automatically enabled',
            'The reference tracking engine remains the live default'])
    (output/'report.yaml').write_text(yaml.safe_dump(report))
    print(yaml.safe_dump(report),flush=True)
