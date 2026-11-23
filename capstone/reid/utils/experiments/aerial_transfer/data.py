"""Chronological GT-box protocol; no crops written to disk."""
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
import sqlite3
from PIL import Image
from utils.teacher.assets import sha256
from utils.tracking.common import ROOT


def crop(row):
    with Image.open(row['path']) as image:
        x,y,w,h=row['box'];right=min(image.width,x+w);bottom=min(image.height,y+h)
        left=max(0,x);top=max(0,y)
        if right<=left or bottom<=top:raise ValueError('Empty clipped GT box')
        return image.convert('RGB').crop((left,top,right,bottom))


def partition(rows,end,protocol):
    """A fixed midpoint separates enrollment and later queries for every track."""
    groups=defaultdict(list)
    for row in rows:groups[row['id']].append(row)
    half=protocol['minimum_gap_frames']//2
    gallery=[];query=[]
    for identity,observations in sorted(groups.items()):
        observations=sorted(observations,key=lambda row:row['frame'])
        references=[r for r in observations if r['frame']<=end//2-half and
                    min(r['box'][2:])>=protocol['minimum_reference_side'] and r['truncation']==0 and r['occlusion']<=1]
        if references:gallery.append(references[0])
        last=-10**9;count=0
        for row in observations:
            if row['frame']<end//2+(protocol['minimum_gap_frames']-half) or min(row['box'][2:])<protocol['minimum_query_side']:continue
            if row['frame']-last<protocol['query_spacing_frames']:continue
            if count>=protocol['maximum_queries_per_identity']:break
            query.append(row);last=row['frame'];count+=1
    return gallery,query


def prepare(config):
    root=ROOT/config['dataset_root'];protocol=config['protocol'];gallery=[];query=[];annotations={};skipped=[]
    for path in sorted((root/'annotations').glob('*.txt')):
        annotations[str(path.relative_to(ROOT))]=sha256(path);rows=[];end=0
        for fields in csv.reader(path.open()):
            values=[int(float(v)) for v in fields];frame,target,x,y,w,h,score,category,truncation,occlusion=values[:10]
            end=max(end,frame)
            # Track zero is a valid VisDrone identity, not an ignore region.
            if score!=1 or target<0 or category not in protocol['vehicle_classes'] or min(w,h)<=0:continue
            image=root/'sequences'/path.stem/f'{frame:07d}.jpg'
            if not image.exists():raise FileNotFoundError(image)
            rows.append(dict(id=f'{path.stem}:{target}',sequence=path.stem,frame=frame,path=str(image.resolve()),
                             box=[x,y,w,h],category=category,truncation=truncation,occlusion=occlusion))
        g,q=partition(rows,end,protocol)
        if len(g)<2 or not q or not ({r['id'] for r in g}&{r['id'] for r in q}):
            skipped.append(dict(sequence=path.stem,reason='Fewer than two references or no later known queries'));continue
        gallery+=g;query+=q
    if not gallery or not query:raise ValueError('No usable vehicle sequences')
    # Exact-pixel comparison against both continuation and original warm-start train crops.
    connection=sqlite3.connect(f'file:{ROOT/config["training_audit"]}?mode=ro',uri=True)
    train_pixels=set();training_hashes={}
    for manifest in config['training_manifests']:
        training_hashes[manifest]=sha256(ROOT/manifest)
        with (ROOT/manifest).open() as file:
            for row in csv.DictReader(file):
                found=connection.execute('SELECT pixel FROM images WHERE path=?',(str(Path(row['path']).resolve()),)).fetchone()
                if not found or not found[0]:raise ValueError('Training crop absent from completed pixel audit')
                train_pixels.add(found[0])
    connection.close();pixels=[]
    for row in gallery+query:
        image=crop(row);w,h=image.size;digest=hashlib.sha256(f'{w},{h}:'.encode()+image.tobytes()).hexdigest()
        if digest in train_pixels:raise ValueError('Selected VisDrone crop duplicates prior training pixels')
        pixels.append(digest)
    return dict(gallery=gallery,query=query),dict(annotations=annotations,training_manifests=training_hashes,
        crop_pixels=pixels,exact_training_pixel_matches=0,skipped_sequences=skipped,
        caveat='Pixel comparison cannot prove physical identity separation; source provenance excludes VisDrone fine-tuning for these checkpoints')
