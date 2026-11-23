"""Cached pixel/file audit; protect held-out images and quarantine label conflicts."""
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import numpy as np
from PIL import Image
import yaml
from utils.tracking.common import ROOT
from utils.teacher.assets import sha256


def read_manifest(path,domain):
    with path.open() as file:rows=list(csv.DictReader(file))
    return [dict(path=str(Path(row['path']).resolve()),id=domain+':'+row['id'],
                 original_id=row['id'],domain=domain,camera=row.get('camera','')) for row in rows]


def split(config):
    data={name:read_manifest(ROOT/value,'vrai' if name.startswith('vrai') else 'vru') for name,value in config['data'].items()}
    vru_ids=sorted({row['id'] for row in data['vru_train']});rng=random.Random(config['seed']);rng.shuffle(vru_ids)
    holdout=set(vru_ids[:config['vru_development_identities']])
    train=data['vrai_train']+[row for row in data['vru_train'] if row['id'] not in holdout]
    vru_by_id=defaultdict(list)
    for row in data['vru_train']:
        if row['id'] in holdout:vru_by_id[row['id']].append(row)
    vru_q=[];vru_g=[]
    for identity in sorted(vru_by_id):
        rows=sorted(vru_by_id[identity],key=lambda row:row['path']);index=rng.randrange(len(rows))
        vru_g.append(rows[index]);vru_q.extend(row for i,row in enumerate(rows) if i!=index)
    development={}
    for domain,query,gallery in [('vrai',data['vrai_query'],data['vrai_gallery']),('vru',vru_q,vru_g)]:
        ids=sorted({row['id'] for row in query})
        for name,allowed in [('fit',set(ids[::2])),('selection',set(ids[1::2]))]:
            development[domain+'_'+name+'_query']=[row for row in query if row['id'] in allowed]
            development[domain+'_'+name+'_gallery']=[row for row in gallery if row['id'] in allowed]
    test=[row for name,rows in data.items() if '_test_' in name for row in rows]
    held=[row for rows in development.values() for row in rows]+test
    if {row['id'] for row in train}&{row['id'] for row in held}:raise ValueError('Train/development/test identity labels overlap')
    return train,development,test


def image_record(path):
    path=Path(path);stat=path.stat()
    try:
        raw=path.read_bytes()
        with Image.open(path) as image:
            image=image.convert('RGB');pixels=np.asarray(image);w,h=image.size
            gray=np.asarray(image.convert('L').resize((16,16),Image.Resampling.BILINEAR))
            tiny=np.asarray(image.convert('L').resize((9,8),Image.Resampling.BILINEAR))
            bits=(tiny[:,1:]>tiny[:,:-1]).reshape(-1)
            dhash=''.join(f'{value:02x}' for value in np.packbits(bits))
        digest=hashlib.sha256(f'{w},{h}:'.encode()+pixels.tobytes()).hexdigest()
        return (str(path),stat.st_size,stat.st_mtime_ns,hashlib.sha256(raw).hexdigest(),digest,dhash,w,h,gray.tobytes(),float(pixels.std()),None)
    except Exception as error:
        return (str(path),stat.st_size,stat.st_mtime_ns,None,None,None,0,0,None,0.,str(error))


def counts(rows):
    return {domain:dict(images=sum(row['domain']==domain for row in rows),
        identities=len({row['id'] for row in rows if row['domain']==domain})) for domain in ['vrai','vru']}


def prepare(config):
    root=ROOT/config['state_root'];root.mkdir(parents=True,exist_ok=True)
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    train,development,test=split(config)
    signature=dict(data={key:sha256(ROOT/value) for key,value in config['data'].items()},
        holdout=config['vru_development_identities'],seed=config['seed'],audit=config['audit'],implementation=sha256(Path(__file__)))
    ready=root/'prepared.yaml'
    if ready.exists():
        prior=yaml.safe_load(ready.read_text())
        if prior['signature']!=signature:raise ValueError('Audit inputs changed; use a new state root')
        for name,digest in prior['prepared_manifests'].items():
            if sha256(root/name)!=digest:raise ValueError('Prepared training manifest changed')
        print('Reusing completed data audit',flush=True);return prior
    held=[row for rows in development.values() for row in rows]+test
    all_rows=train+held;paths=sorted({row['path'] for row in all_rows})
    database=sqlite3.connect(root/'audit.sqlite')
    database.execute('CREATE TABLE IF NOT EXISTS images (path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, sha TEXT, pixel TEXT, dhash TEXT, width INTEGER, height INTEGER, gray BLOB, std REAL, error TEXT)')
    cached={row[0]:(row[1],row[2]) for row in database.execute('SELECT path,size,mtime FROM images')}
    pending=[]
    for path in paths:
        stat=Path(path).stat()
        if cached.get(path)!=(stat.st_size,stat.st_mtime_ns):pending.append(path)
    print(f'Auditing {len(paths)} unique image paths; {len(pending)} need hashing/decoding',flush=True)
    with ThreadPoolExecutor(max_workers=config['audit']['workers']) as pool:
        for index,row in enumerate(pool.map(image_record,pending)):
            database.execute('INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?)',row)
            if (index+1)%config['audit']['checkpoint_images']==0:
                database.commit();print(f'Image audit {index+1}/{len(pending)} committed',flush=True)
    database.commit()
    allowed_paths=set(paths)
    records={row[0]:row for row in database.execute('SELECT * FROM images') if row[0] in allowed_paths}
    database.close()
    errors=[path for path in paths if records[path][-1]]
    if errors:raise ValueError(f'Corrupt images found; inspect audit.sqlite ({len(errors)})')
    held_pixels=defaultdict(set);held_hashes=defaultdict(list)
    for row in held:
        rec=records[row['path']]
        held_pixels[rec[4]].add(row['id']);held_hashes[rec[5]].append(row)
    pixel_ids=defaultdict(set)
    for row in train:
        if records[row['path']][9]>=config['audit']['flat_std_floor']:pixel_ids[records[row['path']][4]].add(row['id'])
    excluded=defaultdict(set);near=[]
    for row in train:
        rec=records[row['path']]
        if rec[9]<config['audit']['flat_std_floor']:continue
        if rec[4] in held_pixels:excluded[row['id']].add('exact_pixels_overlap_heldout')
        if len(pixel_ids[rec[4]])>1:excluded[row['id']].add('exact_pixels_conflicting_train_labels')
        for other in held_hashes.get(rec[5],[]):
            target=records[other['path']]
            if rec[4]==target[4]:continue
            if abs((rec[6]/rec[7])/(target[6]/target[7])-1)>config['audit']['near_duplicate_max_aspect_error']:continue
            a=np.frombuffer(rec[8],np.uint8).astype(np.float32);b=np.frombuffer(target[8],np.uint8).astype(np.float32)
            if float(np.abs(a-b).mean())<=config['audit']['near_duplicate_max_gray_mae']:
                excluded[row['id']].add('near_pixels_overlap_heldout');near.append(dict(train_id=row['id'],held_id=other['id'],train=row['path'],held=other['path']))
    # The warm-start model already trained on every original VRAI train row.
    # Removing new training duplicates alone cannot undo that exposure.
    warm_pixels={records[row['path']][4] for row in train if row['domain']=='vrai'}
    prior_exposed={row['id'] for row in held if records[row['path']][4] in warm_pixels}
    held_conflicts={identity for identities in held_pixels.values() if len(identities)>1 for identity in identities}
    dev_excluded=prior_exposed|held_conflicts
    before_development={name:len(rows) for name,rows in development.items()}
    development={name:[row for row in rows if row['id'] not in dev_excluded] for name,rows in development.items()}
    cleaned=[];seen=set();removed=Counter()
    for row in train:
        rec=records[row['path']]
        if row['id'] in excluded:removed['quarantined_identity_images']+=1;continue
        if rec[9]<config['audit']['flat_std_floor']:removed['flat_images']+=1;continue
        key=(row['id'],rec[4])
        if key in seen:removed['within_identity_duplicate_images']+=1;continue
        seen.add(key);cleaned.append(row)
    grouped=defaultdict(list)
    for row in cleaned:grouped[row['id']].append(row)
    valid=set()
    for identity,rows in grouped.items():
        if len(rows)<2:excluded[identity].add('fewer_than_two_distinct_images')
        elif rows[0]['domain']=='vrai' and len({row['camera'] for row in rows})<2:excluded[identity].add('lost_camera_diversity_after_dedup')
        else:valid.add(identity)
    cleaned=[row for row in cleaned if row['id'] in valid]
    manifests={'train.csv':cleaned,**{name+'.csv':rows for name,rows in development.items()}}
    prepared={}
    for name,rows in manifests.items():
        with (root/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=['path','id','original_id','domain','camera']);writer.writeheader();writer.writerows(rows)
        prepared[name]=sha256(root/name)
    result=dict(signature=signature,prepared_manifests=prepared,available_training=counts(train),usable_training=counts(cleaned),
        development={name:counts(rows) for name,rows in development.items()},test_images_audited_only=len(test),
        quarantined_identities={key:sorted(value) for key,value in excluded.items()},removed_images=dict(removed),
        prior_warmstart_exposed_heldout_identities=sorted(prior_exposed),
        development_excluded_identities=sorted(dev_excluded & {row['id'] for rows in split(config)[1].values() for row in rows}),
        development_images_before_filter=before_development,
        test_prior_exposed_identities=sorted(prior_exposed & {row['id'] for row in test}),
        near_duplicate_pairs=near[:100],near_duplicate_pair_count=len(near),
        physical_identity_note='Dataset-prefixed labels prevent collisions; visual hashing cannot prove that every physical identity is disjoint across independently collected datasets.')
    ready.write_text(yaml.safe_dump(result));(output/'data_audit.yaml').write_text(yaml.safe_dump(result))
    print(yaml.safe_dump(dict(usable_training=result['usable_training'],quarantined_identities=len(excluded),removed_images=dict(removed))),flush=True)
    return result
