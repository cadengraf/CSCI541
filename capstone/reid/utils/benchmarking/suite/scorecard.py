"""Bounded latest/previous/best metrics; publish one HTML only on improvement."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import fcntl
from html import escape
import yaml

ROOT = Path(__file__).resolve().parents[3]
PRIMARY = {'retrieval': (3, 1), 'transfer': (3, 1), 'recovery': (3, 1),
           'kitti': (1, 1), 'cityflow': (1, 1), 'cost': (3, -1), 'detectors': (2, 1)}
KEY_COLUMNS = {'retrieval': [0,1,2], 'transfer': [0,1,2], 'recovery': [0,1,2],
               'kitti': [0], 'cityflow': [0], 'cost': [0,1,2,4,5], 'detectors': [0,1]}
STAGES = {'retrieval': ['joint','aerial'], 'transfer': ['transfer','reference'],
          'recovery': ['recovery','joint_recovery'], 'kitti': ['joint'],
          'cityflow': ['cityflow'], 'cost': ['joint','reference'], 'detectors': ['detectors']}


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def protocol(config, table, row):
    """Compare evaluation inputs, not checkpoint bytes or training history."""
    from .report import document
    joint = table in {'kitti','cost'} and row[0]=='Joint' or table=='retrieval' and row[0]=='Joint'
    base = ROOT/config['output']/config['run']
    if table=='detectors':
        frozen=document(base/'manifest.yaml').get('protocol',{})
        return dict(dataset=frozen.get('dataset'),sequences=sorted(frozen.get('frames',{})),
            frozen_protocol_sha256=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest())
    metadata = document(base/'joint/evaluation_protocol.yaml')
    if joint and metadata:
        return metadata['kitti' if table in {'kitti','cost'} else 'vrai']
    if joint:
        settings = document(ROOT/'configs/teacher.yaml')
        data = settings.get('data', {})
        paths = [ROOT/data[k] for k in ['vrai_query','vrai_gallery'] if k in data]
        if table in {'kitti','cost'}:
            paths = [ROOT/data['kitti_root']/'instances_txt'/f'{s:04d}.txt' for s in data.get('val_sequences', [])]
        return {str(p): hash_file(p) for p in paths if p.is_file()}
    # Semantic benchmark settings + actual annotations/manifests. Writable paths,
    # model weights, source code and training settings do not define a test split.
    chosen = []
    def source(stage, fallback):
        path=base/'configs'/f'{stage}.yaml'
        return str(path.relative_to(ROOT)) if path.is_file() else fallback
    if table=='retrieval':
        aerial=source('aerial','configs/aerial_benchmark.yaml')
        chosen=[document(ROOT/aerial).get('training_config','configs/aerial_training.yaml'),aerial]
    elif table=='transfer': chosen=[source('transfer','configs/aerial_transfer.yaml'),source('reference','configs/vehicle_reference.yaml')]
    elif table=='recovery': chosen=[source('recovery','configs/vehicle_recovery.yaml'),'configs/vehicle_reid.yaml','configs/vehicle_reid_query_pool.yaml']
    elif table=='cityflow': chosen=[source('cityflow','configs/cityflow.yaml')]
    elif table=='cost': chosen=[source('reference','configs/vehicle_reference.yaml')]
    result = {}
    relevant = {'dataset_root','protocol','observations','cases','blackout','evaluation',
                'association','detections','scenes','split','quality','local','memory','cost','false_acceptance_budget',
                'limit_identities','check_frames','max_frames','frames'}
    for index,name in enumerate(chosen):
        settings = document(ROOT/name)
        result[str(index)] = {k:v for k,v in settings.items() if k in relevant}
        data = settings.get('data', {})
        for key,value in data.items():
            if 'test' in key and isinstance(value,str) and (ROOT/value).is_file():
                result[value] = hash_file(ROOT/value)
        for key in ['calibration_query','calibration_gallery']:
            if settings.get(key) and (ROOT/settings[key]).is_file():
                result[key] = hash_file(ROOT/settings[key])
        if settings.get('dataset_root'):
            directory = ROOT/settings['dataset_root']
            for p in sorted((directory/'annotations').glob('*.txt')):
                result[str(p)] = hash_file(p)
    return result


def ingest(ledger, run, tables, protocols, checkpoint=None):
    """Only retain latest, previous and per-protocol best; repeated refreshes are idempotent."""
    improved = []
    for t in tables:
        index, direction = PRIMARY[t['key']]
        for row in t['rows']:
            try:
                value = float(row[index])
            except (ValueError,TypeError):
                continue
            if not math.isfinite(value): continue
            identity = [row[i] for i in KEY_COLUMNS[t['key']]]
            context = protocols(t['key'], row)
            key = hashlib.sha256(json.dumps([t['key'],identity,context], sort_keys=True).encode()).hexdigest()
            entry = ledger.setdefault('rows', {}).setdefault(key, dict(table=t['key'], title=t['title'],
                headers=t['headers'], note=t['note'], protocol=context))
            entry.update(title=t['title'],headers=t['headers'],note=t['note'])
            latest = dict(run=run, values=row, primary=value, checkpoint=checkpoint if row[0]=='Joint' else None)
            if entry.get('latest') == latest: continue
            if entry.get('latest', {}).get('run') != run:
                entry['previous'] = entry.get('latest')
            previous, best = entry.get('previous'), entry.get('best')
            entry['delta_previous'] = value-previous['primary'] if previous else None
            entry['delta_best'] = value-best['primary'] if best else None
            promotes = best is None or direction*(value-best['primary']) > 1e-9
            # Recovery/MTMC improvements must not buy more false reconnects.
            if promotes and best and t['key'] in {'recovery','cityflow'}:
                false_index = 8 if t['key']=='recovery' else 5
                promotes = int(row[false_index]) <= int(best['values'][false_index])
            entry.update(latest=latest, outcome=('baseline' if best is None else 'improved') if promotes else 'no improvement')
            if promotes:
                entry['best'] = deepcopy(latest)
                improved.append(key)
    return improved


def joint_protocol(settings):
    data=settings.get('data', {})
    paths=[ROOT/data[k] for k in ['vrai_query','vrai_gallery'] if k in data]
    kitti=[ROOT/data['kitti_root']/'instances_txt'/f'{s:04d}.txt' for s in data.get('val_sequences', [])]
    return dict(vrai={str(p):hash_file(p) for p in paths if p.is_file()},
                kitti={str(p):hash_file(p) for p in kitti if p.is_file()})


def seed(ledger, config):
    from .report import collect
    archive_config={**config,'run':'_historical_source'}
    # Existing completed historical scores, not new evaluations or promotions of
    # calibration metrics. Keep their model roles/protocols explicit.
    for name in ['teacher','distilled_joint','distillation_control','crop_training/square_256']:
        directory = ROOT/'metrics'/name
        tables=collect(ROOT/'metrics/unmeasured',dict(joint=directory))
        ingest(ledger, 'archived:'+name, tables, lambda k,r: protocol(archive_config,k,r))
    locations = dict(joint=ROOT/'metrics/crop_training/square_256', aerial=ROOT/'metrics/aerial_benchmark',
        transfer=ROOT/'metrics/aerial_transfer', reference=ROOT/'metrics/vehicle_reference',
        recovery=ROOT/'metrics/vehicle_recovery', cityflow=ROOT/'metrics/cityflow')
    tables = collect(ROOT/'metrics', locations)
    # Retrieval joint already seeded with its actual historical run name.
    for t in tables:
        if t['key'] in {'retrieval','kitti','cost'}: t['rows']=[r for r in t['rows'] if r[0]!='Joint']
    ingest(ledger, 'archived:existing-references', tables, lambda k,r: protocol(archive_config,k,r))
    ledger['seeded'] = True


def publish(config, state):
    from .report import collect, table, document
    from .command import write
    directory = ROOT/'metrics/scorecard'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'scores.yaml'
    with (directory/'scorecard.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = document(path) or dict(version=1, rows={})
        if not ledger.get('seeded'): seed(ledger, config)
        base = ROOT/config['output']/config['run']
        tables = collect(base)
        # Partial/failed stages never establish a best model.
        ready = {k for k,v in state.get('stages',{}).items() if v.get('status')=='complete'}
        for t in tables:
            def completed(row):
                key=t['key']
                if key in {'kitti','retrieval','cost'} and row[0]=='Joint': return 'joint' in ready
                if key=='retrieval': return 'aerial' in ready
                if key=='transfer': return row[1] in ready
                if key=='recovery': return ('joint_recovery' if row[0]=='Joint' else 'recovery') in ready
                return bool(set(STAGES[key]) & ready)
            t['rows']=[r for r in t['rows'] if completed(r)]
        saved_config = state.get('config', config)
        checkpoint = dict(path=saved_config['joint_checkpoint'],
            sha256=state.get('signature',{}).get('files',{}).get(saved_config['joint_checkpoint']))
        improved = ingest(ledger, config['run'], tables, lambda k,r: protocol(config,k,r), checkpoint)
        write(path, ledger)
        print('\nModel comparison (primary metric; recovery also requires no extra false reconnects):', flush=True)
        for entry in ledger['rows'].values():
            if entry['latest']['run'] != config['run']: continue
            latest=entry['latest']; previous=entry.get('previous'); best=entry['best']
            print(f'{entry["table"]} | {latest["values"][:3]} | latest {latest["primary"]:.4f} | '
                  f'previous {previous["primary"] if previous else "—"} | best {best["primary"]:.4f} | {entry["outcome"]}', flush=True)
        output = ROOT/'milestones.html'
        if not improved and output.exists() and ledger.get('html_initialized'):
            print(f'No new best: HTML unchanged. Latest/previous results saved in {path}', flush=True)
            return
        sections=[]
        refs=document(ROOT/'configs/published_metrics.yaml')
        for kind in PRIMARY:
            entries=[e for e in ledger['rows'].values() if e['table']==kind]
            if not entries: continue
            rows=[]
            for e in entries:
                latest=e['latest']; previous=e.get('previous'); best=e['best']
                test_id=hashlib.sha256(json.dumps(e['protocol'],sort_keys=True).encode()).hexdigest()[:8]
                rows.append([*latest['values'],test_id,previous['primary'] if previous else '—',
                    e['delta_previous'] if e['delta_previous'] is not None else '—',
                    e['delta_best'] if e['delta_best'] is not None else '—',
                    best['primary'], best['run'], e['outcome']])
                if kind=='retrieval':
                    rows[-1].extend([previous['values'][4] if previous else '—',best['values'][4]])
            if kind=='retrieval':
                for (name,backbone),values in zip(refs.get('methods',[]),refs.get('vru',[])):
                    rows.append([f'Published {name} ({backbone})','VRU-Large clean · reported','reference',values[4],values[5],
                                 'reported','—','—','—',values[4],'published','Different recipe/protocol','—',values[5]])
            headers=[*entries[0]['headers'],'Test ID','Previous primary','Δ previous','Δ prior best','Best primary','Best run','Outcome']
            if kind=='retrieval': headers.extend(['Previous Rank-1','Best Rank-1'])
            sections.append('<h2>'+escape(entries[0]['title'])+'</h2>'+table(headers,rows)+
                '<p class="note">'+escape(entries[0]['note'])+'</p>')
        paper=[]
        for dataset, names, metrics in [('vru',['Small','Medium','Large'],['mAP','R1']),
                                        ('uav',['Validation','Test'],['mAP','R1','R5'])]:
            rows=[]
            for (method,backbone),values in zip(refs.get('methods',[]),refs.get(dataset,[])):
                for weather_index,weather in enumerate(['Normal','Foggy','Rainy']):
                    for split_index,split in enumerate(names):
                        offset=weather_index*6+split_index*len(metrics)
                        rows.append([method,backbone,weather,split,*values[offset:offset+len(metrics)]])
            paper.append('<h3>'+('VRU' if dataset=='vru' else 'UAV-VeID')+'</h3>'+table(
                ['Method','Backbone','Condition','Split',*metrics],rows))
        sections.append('<details><summary>Full published comparison tables and dataset splits</summary>'+''.join(paper)+
            table(['Dataset','Split','Images','Identities','Cameras','Multi-view','Multi-scale'],
                  [r+['yes','yes'] for r in refs.get('datasets',[])])+
            '<p class="note">'+escape(refs.get('source',''))+'. Published recipes differ; unmeasured local cells remain unmeasured.</p></details>')
        videos=[]
        for p in sorted(base.rglob('*.mp4')):
            videos.append('<a href="'+escape(str(p.relative_to(ROOT)))+'">'+escape(str(p.relative_to(base)))+'</a>')
        if videos: sections.append('<h2>Videos for this improved run</h2><p>'+' · '.join(videos)+'</p>')
        html='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ReID best-model scorecard</title><style>body{font:14px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#173338;background:#f5f7f4}h2{font-size:18px;margin-top:25px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #d9e2dc;padding:8px;text-align:left;white-space:nowrap}th{background:#e8efe9}.note{font-size:12px;color:#526660}details{margin:24px 0}summary{cursor:pointer;font-weight:600}a{color:#12685b}</style><main><h1>ReID: latest vs previous vs best</h1><p>One scorecard. Best rows advance only on comparable improvements. This HTML refreshes only when a new best is found; unsuccessful runs print their comparison table in the terminal and update the compact metric ledger.</p><p class="note">Primary metrics: retrieval/transfer mAP, recovery count with no extra false reconnects, KITTI HOTA, CityFlow IDF1, latency (lower). Best is per benchmark, not one overall winner. Different split/protocol/hardware groups are kept separate. Calibration is excluded.</p>'+''.join(sections)+'<p><a href="docs/roadmap.md">Milestones</a> · <a href="docs/benchmark_suite.md">Run guide</a> · <a href="metrics/scorecard/scores.yaml">Latest / previous / best records</a></p></main></html>'
        temporary=output.with_suffix('.partial.html');temporary.write_text(html);temporary.replace(output)
        ledger['html_initialized']=True
        write(path,ledger)
        print(f'Updated best-model scorecard: {output}', flush=True)
