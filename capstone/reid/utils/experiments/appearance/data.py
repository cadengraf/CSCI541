"""Training-only crop degradation and identity-balanced hard-neighbor sampling."""
from io import BytesIO
from PIL import Image, ImageFilter
import torch
from utils.reid_teacher.data import Batches


class Degradation:
    def __init__(self,settings):self.spec=settings

    def __call__(self,image,rng):
        s=self.spec
        if rng.random()>=s['probability']:return image
        w,h=image.size
        scale=rng.uniform(*s['crop_scale'])
        dx=rng.uniform(-s['translation'],s['translation'])*w
        dy=rng.uniform(-s['translation'],s['translation'])*h
        left=(w-w*scale)/2+dx;top=(h-h*scale)/2+dy
        image=image.transform((w,h),Image.Transform.EXTENT,
            (left,top,left+w*scale,top+h*scale),resample=Image.Resampling.BILINEAR,fillcolor=(128,128,128))
        if rng.random()<s['downsample_probability']:
            side=min(min(w,h),rng.randint(*s['short_side']))
            image=image.resize((max(1,round(w*side/min(w,h))),max(1,round(h*side/min(w,h)))),Image.Resampling.BILINEAR)
        if rng.random()<s['blur_probability']:
            image=image.filter(ImageFilter.GaussianBlur(rng.uniform(*s['blur_radius'])))
        if rng.random()<s['jpeg_probability']:
            buffer=BytesIO();image.save(buffer,format='JPEG',quality=rng.randint(*s['jpeg_quality']))
            buffer.seek(0)
            with Image.open(buffer) as compressed:image=compressed.convert('RGB').copy()
        return image


class Sampler(Batches):
    def __init__(self,records,settings,variant,experiment,neighbors=None):
        super().__init__(records,settings)
        self.variant=variant;self.experiment=experiment
        self.neighbors=neighbors
        self.degrade=Degradation(experiment['augmentation'])
        if variant['hard_sampling']:
            if neighbors is None or set(neighbors)!=set(self.identities):
                raise ValueError('Hard-neighbor graph must contain training identities only')
            if any(other==identity or other not in self.mapping for identity,others in neighbors.items() for other in others):
                raise ValueError('Invalid training negative identity')

    def choose(self,rng):
        count=self.settings['training']['identities_per_batch']
        if not self.variant['hard_sampling']:return rng.sample(self.identities,count)
        anchor=rng.choice(self.identities)
        n=min(self.experiment['mining']['hard_identities_per_batch']-1,count-1,len(self.neighbors[anchor]))
        chosen=[anchor]+rng.sample(self.neighbors[anchor],n)
        remaining=[identity for identity in self.identities if identity not in chosen]
        return chosen+rng.sample(remaining,count-len(chosen))

    def records(self,rng):
        result=[];count=self.settings['training']['images_per_identity']//2
        for identity in self.choose(rng):
            for camera in rng.sample(sorted(self.grouped[identity]),2):
                paths=self.grouped[identity][camera]
                selected=rng.sample(paths,count) if len(paths)>=count else rng.choices(paths,k=count)
                result.extend((path,self.mapping[identity]) for path in selected)
        return result

    def sample(self,rng,augmentation_rng):
        images=[];labels=[]
        for path,label in self.records(rng):
            with Image.open(path) as raw:image=raw.convert('RGB')
            if self.variant['augmentation']:image=self.degrade(image,augmentation_rng)
            images.append(self.transform(image));labels.append(label)
        return torch.stack(images),torch.tensor(labels)
