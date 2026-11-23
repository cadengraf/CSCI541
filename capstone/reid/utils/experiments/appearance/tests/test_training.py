import copy
import random
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml
from utils.experiments.appearance.data import Degradation,Sampler
from utils.experiments.appearance.losses import adasp
from utils.experiments.appearance.evaluation import retrieval
from utils.experiments.appearance.command import combine_decision


class TrainingTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[4]
        self.experiment=yaml.safe_load((root/'configs/appearance_training.yaml').read_text())
        self.settings=yaml.safe_load((root/'configs/reid_teacher.yaml').read_text())
        self.settings['training'].update(self.experiment['training'])
        self.rows=[dict(id=str(i),camera=str(c),path=f'{i}_{c}_{j}.jpg') for i in range(12) for c in range(2) for j in range(2)]

    def test_degradation_is_reproducible_and_preserves_valid_rgb(self):
        raw=Image.fromarray(np.random.default_rng(1).integers(0,256,(80,120,3),dtype=np.uint8))
        convert=Degradation(self.experiment['augmentation'])
        one=convert(raw,random.Random(1));two=convert(raw,random.Random(1))
        np.testing.assert_array_equal(np.array(one),np.array(two))
        self.assertEqual(one.mode,'RGB')
        self.assertGreaterEqual(min(one.size),24)
        clean=Degradation({**self.experiment['augmentation'],'probability':0})(raw,random.Random(1))
        np.testing.assert_array_equal(np.array(raw),np.array(clean))

    def test_augmented_control_samples_identical_records(self):
        one=Sampler(self.rows,self.settings,self.experiment['variants']['continuation'],self.experiment)
        two=Sampler(self.rows,self.settings,self.experiment['variants']['degradation'],self.experiment)
        self.assertEqual(one.records(random.Random(42)),two.records(random.Random(42)))

    def test_hard_sampler_retains_unique_balanced_cross_camera_ids(self):
        ids=sorted({row['id'] for row in self.rows})
        neighbors={identity:[other for other in ids if other!=identity][:4] for identity in ids}
        sampler=Sampler(self.rows,self.settings,self.experiment['variants']['hard_sampling'],self.experiment,neighbors)
        rows=sampler.records(random.Random(42));groups={}
        for path,label in rows:groups.setdefault(label,[]).append(path)
        self.assertEqual(len(groups),8)
        self.assertTrue(all(len(value)==4 and len({path.split('_')[1] for path in value})==2 for value in groups.values()))
        neighbors[ids[0]].append('evaluation_identity')
        with self.assertRaises(ValueError):Sampler(self.rows,self.settings,self.experiment['variants']['hard_sampling'],self.experiment,neighbors)

    def test_adasp_is_permutation_invariant_with_finite_gradients(self):
        torch.manual_seed(1)
        values=torch.randn(12,16,requires_grad=True);labels=torch.arange(3).repeat_interleave(4)
        expected=adasp(values,labels)
        order=torch.randperm(len(labels))
        torch.testing.assert_close(expected,adasp(values[order],labels[order]))
        expected.backward();self.assertTrue(torch.isfinite(values.grad).all())
        with self.assertRaises(ValueError):adasp(values[:5],labels[:5])

    def test_retrieval_metric_has_multiple_gallery_positives(self):
        query=torch.tensor([[1.,0.],[0.,1.]])
        gallery=torch.tensor([[1.,0.],[.8,.2],[0.,1.]])
        result=retrieval(query,gallery,['a','b'],['a','a','b'])
        self.assertEqual(result['rank1'],1)
        self.assertEqual(result['map'],1)

    def test_combination_requires_each_independent_gain(self):
        baseline=dict(eligible=True,selection=dict(known_recovery=.5),closed_set=dict(map=.9))
        improved=copy.deepcopy(baseline);improved['selection']['known_recovery']=.52
        entries={name:dict(best=copy.deepcopy(improved)) for name in ['degradation','hard_sampling']}
        entries['continuation']=dict(best=baseline)
        self.assertTrue(combine_decision(self.experiment,entries)['run'])
        entries['hard_sampling']['best']['eligible']=False
        self.assertFalse(combine_decision(self.experiment,entries)['run'])


if __name__=='__main__':unittest.main()
