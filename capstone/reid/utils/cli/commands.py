"""Current commands stay discoverable; historical spellings remain compatible."""
import argparse
from importlib import import_module
import sys


COMMANDS = {
    'train': {
        'rfdetr': 'Fine-tune RF-DETR Seg Large on KITTI training masks; internal development selection',
        'joint-metric': 'Matched square-crop joint training: triplet control versus VRAI AdaSP',
        'teacher': 'Train the shared Co-DINO / RefineMask / ReID model',
        'joint-distill': 'Transfer specialist targets into the joint model',
        'joint-distill-control': 'Matched joint continuation without distillation',
        'compare-reid': 'Matched joint versus identity-only ablation',
        'compare-roi': 'Matched shared-ROI identity adapter ablation',
        'compare-crops': 'Matched crop preprocessing ablation',
        'vrai-adapt': 'Existing joint VRAI-only adaptation',
        'reid-teacher': 'Train the independent TransReID reference',
        'aerial': 'Run/resume the configured mixed VRAI/VRU training queue',
    },
    'benchmark': {
        'rfdetr': 'Repeat the frozen-specialist detector comparison with the completed fine-tuned RF-DETR',
        'detectors': 'Co-DINO versus RF-DETR Seg, frozen specialist/tracker; all KITTI validation',
        'joint-metric': 'Evaluate both completed metric-training arms and update the best-model scorecard',
        'suite': 'Run/resume the configured benchmarks and build consolidated HTML tables',
        'milestones': 'Refresh the HTML snapshot only; no model evaluation',
        'joint': 'Held-out VRAI + KITTI evaluation of the selected square-crop joint model',
        'cityflow': 'Full ground-camera CityFlowV2 validation',
        'aerial': 'Full existing mixed-encoder VRAI/VRU retrieval',
        'aerial-transfer': 'Existing frozen VisDrone transfer protocol',
        'vehicle-reference': 'Bounded three-encoder reference comparison',
        'vehicle-recovery': 'Controlled aerial video recovery diagnostic',
        'vehicle-vru': 'Frozen original/degradation/pretrained VRU references',
        'vehicle': 'Full fixed-detector vehicle benchmark',
        'vehicle-retrieval': 'Specialist retrieval and rejection evaluation',
    },
    'demo': {'vehicle': 'Video → masks → ReID → identity → annotated MP4'},
    'prepare': {name: description for name,description in [
        ('vrai','VRAI identity manifests'),('vru','VRU identity manifests'),
        ('aicity','Existing local AICity crop preparation'),('kitti','KITTI MOTS archives'),
        ('visdrone','VisDrone frames/manifests'),('isaid','iSAID instance masks'),
        ('vehicle-masks','Existing vehicle mask preparation')]},
}

DIRECT = {
    ('train','rfdetr'): ('utils.experiments.rfdetr_finetune.command','main'),
    ('benchmark','rfdetr'): ('utils.experiments.rfdetr_finetune.command','benchmark'),
    ('benchmark','detectors'): ('utils.experiments.detector_compare.command','main'),
    ('train','joint-metric'): ('utils.experiments.joint_metric.command','main'),
    ('benchmark','joint-metric'): ('utils.experiments.joint_metric.command','benchmark'),
    ('benchmark','suite'): ('utils.benchmarking.suite.command','main'),
    ('benchmark','joint-recovery'): ('utils.experiments.joint_evaluation.recovery','main'),
    ('benchmark','joint'): ('utils.experiments.joint_evaluation.command','main'),
    ('train','aerial'): ('utils.experiments.aerial_training.command','main'),
    ('benchmark','milestones'): ('utils.reporting.milestones','run'),
    ('benchmark','cityflow'): ('utils.experiments.cityflow.command','main'),
    ('benchmark','aerial'): ('utils.experiments.aerial_benchmark.command','main'),
    ('benchmark','aerial-transfer'): ('utils.experiments.aerial_transfer.command','main'),
    ('benchmark','vehicle-reference'): ('utils.experiments.vehicle_reference.command','main'),
    ('benchmark','vehicle-recovery'): ('utils.experiments.vehicle_recovery.command','main'),
    ('benchmark','vehicle-vru'): ('utils.vehicle_reid.vru_benchmark','main'),
    ('benchmark','vehicle-retrieval'): ('utils.vehicle_reid.retrieval','main'),
    ('demo','vehicle'): ('utils.vehicle_reid.pipeline','main'),
}


def help_parser(kind):
    examples='\n'.join(f'  {name:<24} {description}' for name,description in COMMANDS[kind].items())
    return argparse.ArgumentParser(prog=f'{kind}.py',
        description=f'Current vehicle ReID {kind} commands. Without a command, only help is printed.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Commands:\n'+examples+'\n\nHistorical commands: '+kind+'.py legacy --help\n'
        'Old command spellings still work. All long runs are user-launched.')


def legacy(kind,args):
    module=import_module('utils.cli.legacy.'+kind)
    original=sys.argv
    try:
        # Explicit "legacy default" retains the original no-mode workflows.
        rest=[] if args==['default'] else args or ['--help']
        sys.argv=[kind+'.py',*rest]
        return module.main()
    finally:sys.argv=original


def dispatch(kind,args=None):
    args=list(sys.argv[1:] if args is None else args)
    if not args or args==['--help'] or args==['-h']:
        help_parser(kind).print_help();return
    command=args[0]
    if command=='legacy':return legacy(kind,args[1:])
    if kind=='benchmark' and command=='vehicle':
        name='retrieval' if any(flag in args[1:] for flag in ['--stage','--limit-identities']) else 'benchmark'
        return import_module('utils.vehicle_reid.'+name).main(args[1:])
    target=DIRECT.get((kind,command))
    if target:
        function=getattr(import_module(target[0]),target[1])
        if command=='milestones':
            if len(args)>1:help_parser(kind).error('milestones accepts no run flags')
            return function()
        return function(args[1:])
    return legacy(kind,args)
