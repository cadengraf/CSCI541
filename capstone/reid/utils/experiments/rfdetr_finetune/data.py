"""Native COCO-RLE manifests and image links; no duplicated images or mask PNGs."""
import hashlib
import json
from pathlib import Path
from PIL import Image
from pycocotools import mask as rle
from utils.full_frame.protocol import TRAIN, VALIDATION
from utils.benchmarking.suite.command import digest,write


def check_splits(config):
    train=set(config['train_sequences']);dev=set(config['development_sequences'])
    test=set(config['benchmark_sequences'])
    if len(train)!=len(config['train_sequences']) or len(dev)!=len(config['development_sequences']):
        raise ValueError('Duplicate sequence in training/development split')
    if not train or not dev or train&dev or train|dev!=set(TRAIN) or test!=set(VALIDATION) or (train|dev)&test:
        raise ValueError('Use disjoint subsets of the 12 KITTI training sequences; keep all nine validation sequences out')


def prepare(config,root):
    check_splits(config)
    source=root/config['dataset_root'];target=root/config['data_output'];target.mkdir(parents=True,exist_ok=True)
    inputs={};metadata={};counts={}
    for split,sequences in [('train',config['train_sequences']),('valid',config['development_sequences'])]:
        folder=target/split;folder.mkdir(exist_ok=True)
        images=[];annotations=[]
        for sequence in sequences:
            seq=f'{sequence:04d}';path=source/'instances_txt'/f'{seq}.txt'
            inputs[str(path)]=digest(path);records={}
            for line in path.read_text().splitlines():
                frame,identity,category,h,w,encoded=line.split(' ',5)
                if int(category)!=1:continue
                record=dict(size=[int(h),int(w)],counts=encoded.encode())
                area=float(rle.area(record))
                if area<=0:raise ValueError('Empty car mask')
                records.setdefault(int(frame),[]).append((record,area,int(identity)))
            paths=sorted((source/'training/image_02'/seq).glob('*.png'))
            if not paths or [int(p.stem) for p in paths]!=list(range(len(paths))):raise ValueError(f'Incomplete sequence {seq}')
            metadata[seq]=[[p.name,p.stat().st_size,p.stat().st_mtime_ns] for p in paths]
            with Image.open(paths[0]) as image:width,height=image.size
            for index,path in enumerate(paths):
                name=f'{seq}_{path.name}';link=folder/name
                if link.is_symlink():
                    if link.resolve()!=path.resolve():raise ValueError('Existing image link points to another input')
                elif link.exists():raise ValueError('Refusing to replace a real file with an image link')
                else:link.symlink_to(path.resolve())
                image_id=len(images)+1
                images.append(dict(id=image_id,file_name=name,width=width,height=height))
                for encoded,area,identity in records.get(index,[]):
                    if encoded['size']!=[height,width]:raise ValueError('Mask/frame dimensions differ')
                    annotations.append(dict(id=len(annotations)+1,image_id=image_id,category_id=0,
                        bbox=rle.toBbox(encoded).tolist(),area=area,iscrowd=0,
                        segmentation=dict(size=encoded['size'],counts=encoded['counts'].decode())))
            if set(records)-set(range(len(paths))):raise ValueError('Annotation points outside available frames')
        document=dict(info=dict(description='KITTI training-partition car masks; internal sequence development split'),
            images=images,annotations=annotations,categories=[dict(id=0,name='car',supercategory='vehicle')])
        path=folder/'_annotations.coco.json';content=json.dumps(document,separators=(',',':'))
        if path.exists() and path.read_text()!=content:raise ValueError('Prepared COCO input changed; choose a new data_output')
        if not path.exists():
            temporary=path.with_suffix('.partial');temporary.write_text(content);temporary.replace(path)
        counts[split]=dict(images=len(images),instances=len(annotations),sequences=sequences,
                           manifest_sha256=hashlib.sha256(content.encode()).hexdigest())
    audit=dict(inputs=inputs,frames=metadata,splits=counts,benchmark_sequences=config['benchmark_sequences'],
        label_mapping={'KITTI_car_1':'training_car_0'},format='COCO compressed RLE; linked original RGB frames')
    path=target/'prepared.yaml'
    import yaml
    if path.exists() and yaml.safe_load(path.read_text())!=audit:raise ValueError('Training inputs changed since preparation')
    if not path.exists():write(path,audit)
    return audit
