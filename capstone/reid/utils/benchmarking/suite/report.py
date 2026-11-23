"""One short scorecard; separate protocols, models, scopes and prior runs."""
import csv
from datetime import datetime, timezone
from html import escape
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]


def records(path):
    if not path.is_file():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def document(path):
    return yaml.safe_load(path.read_text()) if path.is_file() else {}


def number(value, scale=1):
    return '—' if value is None or value == '' else f'{float(value)*scale:.10f}'.rstrip('0').rstrip('.')


def collect(base, locations=None):
    at = lambda name: (locations or {}).get(name, base/name)
    tables = []
    def add(key, title, headers, rows, note):
        tables.append(dict(key=key, title=title, headers=headers, rows=rows, note=note))

    joint = records(at('joint')/'vrai.csv')
    rows = [['Joint', 'VRAI held-out', r['pooling'], number(r['mAP']), number(r['Rank1'])] for r in joint]
    aerial_paths=[] if document(at('aerial')/'report.yaml').get('smoke_check') else sorted(at('aerial').glob('mixed_*.yaml'))
    for path in aerial_paths:
        for r in document(path):
            rows.append([r['variant'], 'VRU-Large local' if r['domain']=='vru' else 'VRAI held-out',
                         'specialist', number(r['map'], 100), number(r['rank1'], 100)])
    add('retrieval', 'Aerial retrieval', ['Model', 'Dataset / split', 'Feature path', 'mAP %', 'Rank-1 %'], rows,
        'Joint and specialists have different training histories. VRAI multi-positive and VRU one-reference protocols are separate. Calibration never appears as held-out accuracy.')

    rows = []
    for stage in ['transfer', 'reference']:
        for path in sorted(at(stage).glob('*.yaml')):
            r = document(path)
            if not isinstance(r, dict) or 'retrieval' not in r:
                continue
            for fit, point in r.get('open_set', {}).items():
                rows.append([r['model'], stage, fit, number(r['retrieval']['map'], 100),
                    number(r['retrieval']['rank1'], 100), number(point.get('known_recovery'), 100),
                    number(point.get('unknown_far'), 100), number(point.get('wrong_known_rate'), 100)])
    add('transfer', 'Cross-dataset retrieval and rejection', ['Model', 'Protocol', 'Source fit', 'mAP %',
        'Rank-1 %', 'Correct known acceptance %', 'Unknown FAR %', 'Wrong known acceptance %'], rows,
        'VisDrone same-video chronological GT crops; not cross-camera tracking. Thresholds are source-fitted and unchanged on target. Joint transfer retrieval is not implemented here.')

    rows = []
    for stage, label in [('recovery', 'TransReID'), ('joint_recovery', 'Joint')]:
        result = document(at(stage)/'report.yaml')
        for scenario, methods in result.get('aggregate', {}).items():
            for method, r in methods.items():
                c = r['counts']
                rows.append([label, scenario, method, r['recovered'], r['missed'], r['censored'], r['unenrolled'],
                    c.get('global_correct', 0)+c.get('local_correct', 0),
                    c.get('global_false', 0)+c.get('local_false', 0),
                    c.get('global_unscored', 0)+c.get('local_unscored', 0),
                    c.get('no_confident_identity', 0), number(r['mean_confirmation_delay_frames'])])
    add('recovery', 'Within-video recovery diagnostics', ['Encoder', 'Scenario', 'Association', 'Recovered',
        'Missed', 'Censored', 'Unenrolled', 'Correct reconnects', 'False reconnects', 'Unscored reconnects',
        'Abstention observations', 'Delay frames'], rows,
        'Identical GT boxes, reliability gates and association rules. Three target blackouts are diagnostic, not proof of reliability. Fixed specialist thresholds are transferred unchanged to joint embeddings. Return events and reconnection claims have different denominators; natural and injected rows overlap footage and must not be pooled.')

    rows = [['Joint', number(r['HOTA']), number(r['IDF1']), number(r['sMOTSA']), r['IDSW']]
            for r in records(at('joint')/'kitti_mots/summary.csv')]
    add('kitti', 'Ground-camera within-video tracking — KITTI MOTS', ['Model', 'HOTA', 'IDF1', 'sMOTSA', 'ID switches'], rows,
        'All nine validation sequences. Strong local continuity does not establish cross-camera ReID or long-term recovery.')
    rows = []
    for path in sorted(at('cityflow').glob('*/report.yaml')):
        r = document(path)
        if not isinstance(r, dict) or 'metrics' not in r or r.get('smoke_check'):
            continue
        m = r['metrics']
        counts = r['claim_audit']['counts']
        rows.append([r.get('model', path.parent.name), number(m['idf1'], 100), number(m['idp'], 100),
                     number(m['idr'], 100), counts.get('correct', 0), counts.get('false', 0), counts.get('unscored', 0)])
    add('cityflow', 'Ground-camera cross-camera tracking — CityFlow validation', ['Encoder', 'IDF1 %',
        'ID precision %', 'ID recall %', 'Correct sparse claims', 'False sparse claims', 'Unscored sparse claims'], rows,
        'Provided YOLOv3 detections, S02/S05 validation. Official IDF1 and sparse-claim audits are different metrics. This stage evaluates specialists, not the joint model; detector-generation cost is unavailable.')

    rows = []
    timings = records(at('joint')/'timing.csv')
    if timings:
        frames = sum(int(r['frames']) for r in timings)
        mean = sum(float(r['mean_ms'])*int(r['frames']) for r in timings)/frames
        rows.append(['Joint', 'KITTI full-frame path', '1 frame', number(mean), ', '.join(sorted({r['hardware'] for r in timings})),
                     'Decode + model + association + RLE; cached sequence timings may be reused; excludes MP4 rendering'])
    for path in sorted(at('reference').glob('*.yaml')):
        r = document(path)
        if not isinstance(r, dict) or 'cost' not in r:
            continue
        for batch in r['cost']['batches']:
            rows.append([r['model'], 'Specialist encoder only', f'{batch["batch_size"]} crops', number(batch['median_ms']),
                         r['cost']['hardware'], r['cost']['scope']])
    add('cost', 'Inference cost — compare only matching scopes', ['Model', 'Scope', 'Work unit', 'Latency ms', 'Hardware', 'Included / excluded'], rows,
        'Encoder-batch latency cannot be compared directly with full-frame latency. These are not fresh detector-to-annotated-video FPS measurements. A live end-to-end speed claim remains open.')
    rows=[]
    for r in records(at('detectors')/'summary.csv'):
        rows.append([r['detector'],r['reid'],number(r['HOTA']),number(r['DetA']),number(r['AssA']),
            number(r['IDF1']),number(r['sMOTSA']),r['IDSW'],number(r['pipeline_ms']),number(r['video_ms']),
            r['hardware'],r['recovered'],r['eligible'],r['false_reconnections']])
    add('detectors','Detector/mask replacement — frozen specialist and local tracker',
        ['Detector','ReID','HOTA','DetA','AssA','IDF1','sMOTSA','ID switches','Pipeline ms','Fresh video ms',
         'Hardware','Recovered returns','Eligible returns','False reconnects'],rows,
        'All nine KITTI validation sequences; fixed TransReID and local tracker. Co-DINO is KITTI/VRAI fine-tuned and computes an unused joint embedding. RF-DETR exposure (COCO baseline or KITTI fine-tuning) and resolution are recorded in each run manifest. Different training, precision and environments: practical replacement test, not a matched-training architecture claim. Pipeline timing includes decode/model/masks/ReID/association/RLE/checkpoints; separate fresh 200-frame video timing includes annotation and encoding. Offline GT scoring and initialization are excluded. Visibility-gap returns are local diagnostics, not cross-camera identity recovery.')
    return tables


def table(headers, rows):
    def cell(value):
        if isinstance(value,float): return f'{value:.2f}'
        if isinstance(value,str) and '.' in value:
            try: return f'{float(value):.2f}'
            except ValueError: pass
        return escape(str(value))
    body = ''.join('<tr>'+''.join(f'<td>{cell(v)}</td>' for v in row)+'</tr>' for row in rows)
    if not rows:
        body = f'<tr><td colspan="{len(headers)}">Not measured in this suite yet.</td></tr>'
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{cell(v)}</th>' for v in headers)+\
        '</tr></thead><tbody>'+body+'</tbody></table></div>'


def render(config, state):
    from .scorecard import publish
    return publish(config, state)
