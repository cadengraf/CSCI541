"""Development-only temporal-evidence ablation and frozen validation assessment."""
import argparse
from copy import deepcopy
from itertools import product
import torch
import yaml
from utils.tracking.common import ROOT, load
from utils.full_frame.protocol import VALIDATION
from . import budget


def variants(config):
    choices=dict(reference=None)
    for mode,settings in config['modes'].items():
        for score,margin in product(config['grid']['similarity'],config['grid']['margin']):
            spec=dict(engine='tracklets',recovery_similarity=score,recovery_margin=margin,
                      recovery_hits=settings['recovery_hits'])
            if settings['aggregation']:
                spec.update(engine='aggregated',aggregation={**config['evidence'],
                    **{key:settings[key] for key in ['pool_source','quarantine_recovered_memory']}})
            choices[f'{mode}_s{round(score*100):03d}_m{round(margin*100):03d}']=spec
    return choices


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/tracklet_evidence.yaml')
    args=parser.parse_args(argv)
    config=yaml.safe_load((ROOT/args.config).read_text())
    budget.validate_development(config)
    torch.set_num_threads(config['cpu_threads'])
    # Require complete frozen observations. Detection/extraction is a separate
    # signed stage; never silently substitute a different model or bank.
    for seq in [*config['sequences'],*VALIDATION]:
        base=config['features'] if seq in config['sequences'] else 'data/vehicle_benchmark/specialist'
        if not load(ROOT/base/f'{seq:04d}'/'state.pt')['complete']:
            raise ValueError(f'Incomplete specialist bank: {seq:04d}')
    pipeline=yaml.safe_load((ROOT/config['pipeline_config']).read_text())
    choices=variants(config)
    development=budget.replay(config,pipeline,choices,'development')
    measured=budget.aggregate(development)
    selected={}
    for mode in config['modes']:
        subset={name:value for name,value in choices.items() if name.startswith(mode+'_')}
        selected[mode]=budget.select_point({name:measured[name] for name in subset},subset,config['budget'])
    frozen=dict(selected=selected,budget=config['budget'],development_sequences=config['sequences'],
                automatic_promotion=False,default_engine='reference',
                fallback_assessment='pre-registered similarity .82 and margin .08; diagnostic only')
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    (output/'selection.yaml').write_text(yaml.safe_dump(frozen))
    (output/'development.yaml').write_text(yaml.safe_dump(dict(per_sequence=development,aggregate=measured)))
    # Verify hooks did not alter the archived control streams before validation.
    old=yaml.safe_load((ROOT/'metrics/vehicle_recovery_budget/development.yaml').read_text())
    equivalence={seq:entries['frame_control_s082_m008']['identity_digest']==
                 old['per_sequence'][seq]['s082_m008_h3']['identity_digest']
                 for seq,entries in development.items()}
    if not all(equivalence.values()):
        raise RuntimeError(f'Frame control regression: {equivalence}')
    evaluation_choices=dict(reference=None)
    for mode in config['modes']:
        point=selected[mode] or f'{mode}_s082_m008'
        evaluation_choices[mode]=choices[point]
    evaluation_config=deepcopy(config)
    evaluation_config.update(sequences=list(VALIDATION),features='data/vehicle_benchmark/specialist')
    evaluation=budget.replay(evaluation_config,pipeline,evaluation_choices,'evaluation',selection_signature=frozen)
    assessed=budget.aggregate(evaluation)
    report=dict(selection=frozen,development=measured,evaluation=assessed,
                control_identity_equivalence=equivalence,
                limitations=['No training or encoder changes; association-only ablation on frozen observations',
                  'Natural GT visibility returns and accepted global claims have different denominators',
                  'Development sequences overlap detector training; validation is separate',
                  'Pooled evidence requires at least 3 observations spanning 4 frames; control requires 3 consecutive claims',
                  'Zero accepted claims is not evidence of accurate recovery',
                  'These sequences do not evaluate cross-camera retrieval or end-to-end inference speed'])
    (output/'report.yaml').write_text(yaml.safe_dump(report))
    lines=['# Temporal evidence experiment','',
           'Default inference remains the reference tracker. No automatic promotion.','',
           '| Frozen validation variant | Correct global | False global | Unscored global | Natural returns recovered |',
           '|---|---:|---:|---:|---:|']
    for name,item in assessed.items():
        returns=item['natural_returns']
        lines.append(f"| {name} | {item.get('global_correct',0)} | {item.get('global_false',0)} | {item.get('global_unscored',0)} | {returns.get('recovered',0)}/{returns.get('visibility_returns',0)} |")
    lines+=['',f'Development selection: {selected}.',
            'All four original control identity streams reproduce the archived development results.',
            '',*report['limitations']]
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(yaml.safe_dump(dict(selected=selected,evaluation=assessed)),flush=True)


if __name__=='__main__':
    main()
