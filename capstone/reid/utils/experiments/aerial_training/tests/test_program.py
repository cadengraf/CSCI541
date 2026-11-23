from collections import Counter
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
import yaml
from utils.experiments.aerial_training.data import Sampler
from utils.experiments.aerial_training.model import initialize
from utils.experiments.aerial_training.audit import image_record


class AerialProgramTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[4]
        self.config=yaml.safe_load((root/'configs/aerial_training.yaml').read_text())
        self.settings=yaml.safe_load((root/'configs/reid_teacher.yaml').read_text())
        self.settings['training'].update(self.config['training'])
        self.rows=[]
        for domain in ['vrai','vru']:
            for identity in range(6):
                for image in range(4):
                    self.rows.append(dict(id=f'{domain}:{identity}',domain=domain,original_id=str(identity),
                        camera=str(image//2) if domain=='vrai' else '',path=f'{domain}_{identity}_{image}.jpg'))

    def test_balanced_namespaced_batches_preserve_real_camera_rules(self):
        sampler=Sampler(self.rows,self.settings,self.config,self.config['variants']['mixed_baseline'])
        rows=sampler.records(random.Random(1));counts=Counter(row['domain'] for row,_ in rows)
        self.assertEqual(counts,dict(vrai=16,vru=16))
        self.assertEqual(len({label for _,label in rows}),8)
        self.assertNotEqual(sampler.mapping['vrai:1'],sampler.mapping['vru:1'])
        for identity in {row['id'] for row,_ in rows}:
            selected=[row for row,_ in rows if row['id']==identity]
            self.assertEqual(len({row['path'] for row in selected}),4)
            self.assertEqual(len({row['camera'] for row in selected}),2 if identity.startswith('vrai:') else 1)

    def test_degradation_does_not_change_identity_or_image_sampling(self):
        one=Sampler(self.rows,self.settings,self.config,self.config['variants']['mixed_baseline'])
        two=Sampler(self.rows,self.settings,self.config,self.config['variants']['mixed_degradation'])
        self.assertEqual(one.records(random.Random(42)),two.records(random.Random(42)))

    def test_hard_graph_cannot_reference_other_domains_or_heldout_ids(self):
        sampler=Sampler(self.rows,self.settings,self.config,self.config['variants']['mixed_baseline'])
        graph={identity:[other for other in sampler.identities if other!=identity and other.split(':')[0]==identity.split(':')[0]] for identity in sampler.identities}
        hard=Sampler(self.rows,self.settings,self.config,self.config['variants']['mixed_hard_sampling'],graph)
        self.assertEqual(len(hard.records(random.Random(2))),32)
        graph['vrai:0'].append('vru:0')
        with self.assertRaises(ValueError):Sampler(self.rows,self.settings,self.config,self.config['variants']['mixed_hard_sampling'],graph)

    def test_decoded_pixel_hash_finds_identical_pixels_in_different_files(self):
        with TemporaryDirectory() as folder:
            pixels=np.random.default_rng(1).integers(0,255,(20,24,3),dtype=np.uint8)
            one=Path(folder)/'one.png';two=Path(folder)/'two.bmp'
            Image.fromarray(pixels).save(one);Image.fromarray(pixels).save(two)
            first=image_record(one);second=image_record(two)
            self.assertNotEqual(first[3],second[3]);self.assertEqual(first[4],second[4])

    def test_warmstart_classifier_rows_are_remapped_by_dataset_and_identity(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.base=torch.nn.Linear(2,2,bias=False);self.classifier=torch.nn.Linear(2,3,bias=False)
                with torch.no_grad():self.classifier.weight.fill_(.5)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'source.pt'
            torch.save(dict(model={'base.weight':torch.full((2,2),3.),'classifier.weight':torch.tensor([[1.,2.],[3.,4.]])},
                identity_mapping={'1':0,'2':1}),path)
            mapping={'vrai:1':0,'vru:1':1,'vrai:2':2}
            with patch('utils.experiments.aerial_training.model.ROOT',Path(folder)),patch('utils.experiments.aerial_training.model.build',return_value=Tiny()):
                model=initialize({},dict(warmstart='source.pt'),mapping,'cpu')
            torch.testing.assert_close(model.classifier.weight[0],torch.tensor([1.,2.]))
            torch.testing.assert_close(model.classifier.weight[1],torch.tensor([.5,.5]))
            torch.testing.assert_close(model.classifier.weight[2],torch.tensor([3.,4.]))
            torch.testing.assert_close(model.base.weight,torch.full((2,2),3.))


if __name__=='__main__':unittest.main()
