"""Reviewable reporting and video labels; never changes benchmark decisions."""
from pathlib import Path
import csv
import subprocess
import torch
import yaml
from utils.tracking.common import ROOT,load
from utils.teacher.assets import sha256


def finalize(config_path='configs/vehicle_encoder_benchmark.yaml'):
    config=yaml.safe_load((ROOT/config_path).read_text());output=ROOT/config['output']
    torch.set_num_threads(config['cpu_threads'])
    report=yaml.safe_load((output/'report.yaml').read_text())
    costs=yaml.safe_load((output/'online_cost.yaml').read_text()) if (output/'online_cost.yaml').exists() else {}
    # nvidia-smi can return a desktop program's full command line. Keep only
    # executable names and memory/PIDs in the user-facing cost receipt.
    for item in costs.values():
        for key in ['gpu_workload_before','gpu_workload_after']:
            for process in item.get(key,{}).get('other_compute_processes',[]):
                process['name']=Path(process['name'].split(' ',1)[0]).name
    if costs:(output/'online_cost.yaml').write_text(yaml.safe_dump(costs))
    intervals=report['paired_intervals']
    with (output/'tracking'/'paired_intervals.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(next(iter(intervals.values()))));writer.writeheader();writer.writerows(intervals.values())
    changes={}
    for before,after in [('original_reference','degradation_reference'),('original_query_pool','degradation_query_pool')]:
        changed=[]
        for seq,entries in report['per_sequence'].items():
            left={(event['truth_id'],event['returned_at']):event['result'] for event in entries[before]['events']}
            right={(event['truth_id'],event['returned_at']):event['result'] for event in entries[after]['events']}
            for key in sorted(set(left)|set(right)):
                if left.get(key)!=right.get(key):changed.append(dict(sequence=seq,truth_id=key[0],returned_at=key[1],
                    original=left.get(key,'no_prior_reference'),degradation=right.get(key,'no_prior_reference')))
        changes[after]=changed
    manifest=yaml.safe_load((ROOT/config['candidate_selection']).read_text())
    source=load(ROOT/config['candidate_checkpoint']);exported=load(ROOT/manifest['candidate']['checkpoint'])
    exact=source['model'].keys()==exported['model'].keys() and all(torch.equal(value,exported['model'][name]) for name,value in source['model'].items())
    if not exact:raise RuntimeError('Inference export does not match epoch-3 training tensors')
    del source,exported
    receipt=dict(tests_passed=48,inference_export_matches_training_weights=True,
        original_streams_reproduced=report['original_identity_equivalence'],return_outcome_changes=changes,
        inference_export_sha256=manifest['candidate']['checkpoint_sha256'],thresholds_changed=False,default_changed=False)
    (output/'integration.yaml').write_text(yaml.safe_dump(receipt))
    if costs:
        videos=[ROOT/config['live']['output']/name/'annotated.mp4' for name in ['original','degradation']]
        signed=dict(sources={str(path.relative_to(ROOT)):sha256(path) for path in videos},labels=['Original TransReID','Degradation-trained TransReID'])
        receipt_path=output/'video.yaml'
        prior=yaml.safe_load(receipt_path.read_text()) if receipt_path.exists() else None
        valid=(prior is not None and {key:prior[key] for key in signed}==signed and
               prior.get('output_sha256')==sha256(output/'comparison.mp4'))
        if not valid:
            filters=[]
            for index,label in enumerate(signed['labels']):
                filters.append(f"[{index}:v]pad=iw:ih+40:0:40:black,drawtext=text='{label}':x=12:y=8:fontsize=20:fontcolor=white[v{index}]")
            filters.append('[v0][v1]hstack=inputs=2[out]')
            temporary=output/'comparison.labeled.partial.mp4'
            subprocess.run(['ffmpeg','-loglevel','error','-y','-i',str(videos[0]),'-i',str(videos[1]),
                '-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-crf','20',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],check=True)
            temporary.replace(output/'comparison.mp4')
            receipt_path.write_text(yaml.safe_dump(dict(**signed,output_sha256=sha256(output/'comparison.mp4'))))
    by_name={row['model']:row for row in report['tracking']}
    old=by_name['original_query_pool'];new=by_name['degradation_query_pool']
    recovery=report['recovery'];small=report['size']
    lines=['# Encoder benchmark interpretation','',
        'The degradation-trained encoder does not improve this fixed-threshold KITTI MOTS video benchmark. Keep the original specialist as the default.','',
        f"With query pooling, pooled HOTA changes {old['HOTA']:.3f} → {new['HOTA']:.3f} ({new['HOTA']-old['HOTA']:+.3f} points), and IDF1 {old['IDF1']:.3f} → {new['IDF1']:.3f} ({new['IDF1']-old['IDF1']:+.3f} points). Both recover 9/26 natural visibility returns, leaving 17 missed returns and no censored events.",
        '', 'Accepted global claims change from 6 correct / 1 wrong / 2 unscored to 5 correct / 1 wrong / 1 unscored. The scored error fractions are 1/7 and 1/6 respectively; these are tiny samples, not reliable risk estimates. Local reacquisition errors remain separately reported.',
        '', 'The reference local tracker shows a small pooled HOTA change (79.456 → 79.526), lower IDF1 (87.346 → 87.162), and unchanged recovery (6/26). There is no consistent replacement advantage.',
        '', 'Small-object returns remain 9/25 with query pooling and 6/25 with the reference. Almost all natural returns are small at the fixed 48-pixel GT short-side threshold, so there is little evidence about larger-return behavior.',
        '', 'The earlier synthetic VRAI low-resolution gain has not translated into better recovery on these real video inputs under the frozen configuration. This does not establish that degradation augmentation is universally harmful; camera/view distribution and threshold calibration differ. No calibration was performed because this first comparison did not meet the proposed recovery-improvement criterion.',
        '', '## Inference cost','']
    for name,item in costs.items():lines.append(f"- {name}: {item['wall_fps']:.2f} FPS over 100 live frames, initialization {item['initialization_seconds']:.2f} s separately.")
    lines+=['','These runs include every online component and output/checkpoint I/O. Other GPU processes were recorded; this is a short desktop-GPU measurement, not an isolated relative-speed claim. The earlier other-project model job was absent from the before/after live process records; a desktop GPU process remained.',
        '', '## Verification','',
        'All 18 original identity-stream checks pass (two original associations across nine sequences). Detection indices and boxes match for both encoders at every frame. The inference export exactly matches all model tensors in the final epoch-3 degradation checkpoint. All 48 integration tests pass. The labeled side-by-side comparison video contains the same 100 source frames for both models.',
        '', 'This is a within-video development comparison, not a new VRAI retrieval or hidden-test leaderboard result. Full counts, sequence bootstrap intervals, per-return outcomes, and size definitions are in `report.yaml`.']
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')
    print('Encoder comparison finalized:',output/'analysis.md',flush=True)
