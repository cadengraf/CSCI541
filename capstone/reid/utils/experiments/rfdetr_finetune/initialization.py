"""Preserve COCO car logits and grouped query layout instead of flat truncation."""
import torch
import hashlib
from utils.benchmarking.suite.command import digest


def convert_weights(source,car_id,num_queries,groups):
    state=source['model']
    for name in ['refpoint_embed.weight','query_feat.weight']:
        if state[name].shape[0]!=num_queries*groups:raise ValueError('Pretrained query layout differs from pinned metadata')
    converted={}
    heads=[]
    for key,value in state.items():
        if key.startswith('class_embed.') or 'enc_out_class_embed.' in key:
            if value.shape[0]<=car_id+1:raise ValueError('Unexpected COCO classifier shape')
            converted[key]=value[[car_id,value.shape[0]-1]].clone()
            heads.append(key)
        else:converted[key]=value
    if 'class_embed.weight' not in heads:raise ValueError('Missing detector classification head')
    return dict(model=converted,model_name='RFDETRSegLarge',
        args=dict(num_queries=num_queries,group_detr=groups,num_classes=1,class_names=['car']))


def prepare(config,comparison,root):
    path=root/config['data_output']/'car-group-aware-init.pth'
    metadata=path.with_suffix('.yaml')
    spec=comparison['rfdetr'];source=root/spec['checkpoint']
    if hashlib.md5(source.read_bytes()).hexdigest()!=spec['md5']:
        raise ValueError('RF-DETR initialization differs from the pinned COCO asset')
    signature=dict(source_sha256=digest(source),car_id=spec['coco_car_id'],
        num_queries=config['pretrained_num_queries'],groups=config['pretrained_group_detr'])
    import yaml
    from utils.benchmarking.suite.command import write
    if path.exists():
        if not metadata.exists() or yaml.safe_load(metadata.read_text())!={**signature,'sha256':digest(path)}:
            raise ValueError('Converted initialization checkpoint changed')
        return path
    weights=torch.load(source,map_location='cpu',weights_only=False)
    result=convert_weights(weights,spec['coco_car_id'],config['pretrained_num_queries'],config['pretrained_group_detr'])
    temporary=path.with_suffix('.partial');torch.save(result,temporary);temporary.replace(path)
    write(metadata,{**signature,'sha256':digest(path)})
    return path
