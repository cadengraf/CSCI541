"""Train-only confusing-ID graph from the completed mixed baseline encoder."""
from collections import defaultdict
import random
import torch
import torch.nn.functional as F
from PIL import Image
from utils.tracking.common import ROOT,load,save
from utils.teacher.assets import sha256
from utils.reid_teacher.model import build
from utils.reid_teacher.data import transform


def run(config,settings,data,signature):
    source=ROOT/config['checkpoint_dir']/'mixed_baseline/best.pt'
    source_hash=sha256(source);signed=dict(inputs=signature,source_sha256=source_hash)
    path=ROOT/config['state_root']/'neighbors.pt'
    previous=load(path) if path.exists() else None
    if previous is not None and previous['signature']!=signed:raise ValueError('Aerial negative-mining inputs changed')
    if previous is not None and previous['complete']:return previous['neighbors']
    grouped=defaultdict(list)
    for row in data['train']:grouped[row['id']].append(row)
    ids=sorted(grouped);rng=random.Random(config['seed']+91);rows=[]
    for identity in ids:
        pool=grouped[identity];n=min(config['mining']['samples_per_identity'],len(pool))
        if pool[0]['domain']=='vrai':
            cameras=defaultdict(list)
            for row in pool:cameras[row['camera']].append(row)
            selected=[]
            for camera in sorted(cameras):selected+=rng.sample(cameras[camera],min(n//2,len(cameras[camera])))
        else:selected=rng.sample(pool,n)
        rows.extend(selected)
    saved=load(source);mapping={identity:index for index,identity in enumerate(ids)}
    if saved['identity_mapping']!=mapping:raise ValueError('Mixed baseline identities differ from training pool')
    model=build(settings,len(ids),pretrained=False).cuda();model.load_state_dict(saved['model']);del saved
    model.eval().requires_grad_(False);conversion=transform(settings['model']['image_size'])
    chunks=[] if previous is None else previous['chunks'];begin=sum(len(value) for value in chunks)
    with torch.inference_mode():
        for start in range(begin,len(rows),32):
            images=[]
            for row in rows[start:start+32]:
                with Image.open(row['path']) as image:images.append(conversion(image.convert('RGB')))
            chunks.append(F.normalize(model(torch.stack(images).cuda()).float(),dim=1).cpu().half())
            if (start//32+1)%40==0 or start+32>=len(rows):
                save(dict(signature=signed,complete=False,chunks=chunks),path)
                print(f'Mixed training negative mining: {min(start+32,len(rows))}/{len(rows)}',flush=True)
    vectors=torch.cat(chunks).float();positions=defaultdict(list)
    for index,row in enumerate(rows):positions[row['id']].append(index)
    prototypes=torch.stack([F.normalize(vectors[positions[identity]].mean(0),dim=0) for identity in ids]).cuda()
    neighbors={}
    for domain in ['vrai','vru']:
        indices=[i for i,identity in enumerate(ids) if identity.startswith(domain+':')]
        features=prototypes[indices];similarity=features@features.T;similarity.fill_diagonal_(-float('inf'))
        k=min(config['mining']['top_k'],len(indices)-1);columns=similarity.topk(k,dim=1).indices.cpu().tolist()
        for row,index in enumerate(indices):neighbors[ids[index]]=[ids[indices[column]] for column in columns[row]]
    save(dict(signature=signed,complete=True,neighbors=neighbors,training_crops=len(rows),identities=len(ids)),path)
    print(f'Fixed training-only neighbor graph prepared for {len(ids)} identities',flush=True)
    return neighbors
