from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
import yaml

from utils.teacher.model import CoDINOJoint, IdentityHead
from utils.experiments.appearance.losses import adasp
from utils.experiments.joint_metric.model import MetricJoint, metric_losses
from utils.experiments.joint_metric import command


class MetricTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.head=IdentityHead(8,12,30.,.15)
        self.x=F.normalize(torch.randn(8,8),dim=1).detach().requires_grad_()
        self.labels=torch.arange(4).repeat_interleave(2)

    def test_only_vrai_metric_changes_and_gradients_remain_finite(self):
        old=self.head.losses(self.x,self.labels,.3)
        new=metric_losses(self.head.losses,self.x,self.labels,.3,True,.04)
        torch.testing.assert_close(old['loss_identity'],new['loss_identity'],rtol=0,atol=0)
        torch.testing.assert_close(new['loss_triplet'],adasp(self.x,self.labels,.04))
        sum(new.values()).backward()
        self.assertTrue(torch.isfinite(self.x.grad).all())
        self.assertTrue(torch.isfinite(self.head.classifier.grad).all())
        self.assertGreater(self.x.grad.abs().sum().item(),0)

    def test_control_is_exact_original_loss(self):
        expected=self.head.losses(self.x,self.labels,.3)
        actual=metric_losses(self.head.losses,self.x,self.labels,.3,False,.04)
        for name in expected:torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)

    def test_model_routes_unbalanced_kitti_to_original_and_adds_no_state(self):
        def initialize(model,settings,*args,**kwargs):
            torch.nn.Module.__init__(model)
            model.reid=IdentityHead(8,12,30.,.15)
        uneven=torch.tensor([0,0,1,2,3,3,3,4])
        def kitti(model,x,labels):return model.reid.losses(x,labels,.3)
        with patch.object(CoDINOJoint,'__init__',initialize), patch.object(CoDINOJoint,'crop_embeddings',lambda self,x:x), patch.object(CoDINOJoint,'kitti_losses',kitti):
            model=MetricJoint({'matched_reid':{'vrai_metric':'adasp','metric_temperature':.04}})
            original_keys=set(model.state_dict())
            model.crop_embeddings(self.x)
            losses=model.reid.losses(self.x,self.labels,.3)
            torch.testing.assert_close(losses['loss_triplet'],adasp(self.x,self.labels,.04))
            losses=model.kitti_losses(self.x,uneven)
            expected=IdentityHead.losses(model.reid,self.x,uneven,.3)
            for name in expected:torch.testing.assert_close(losses[name],expected[name])
            self.assertEqual(original_keys,set(model.state_dict()))
            self.assertEqual(original_keys,set(self.head.state_dict(prefix='reid.')))

    def test_benchmark_handoff_preserves_baseline_and_never_enables_training(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'configs').mkdir()
            base=dict(run='baseline',joint_checkpoint='original',joint_last_checkpoint='original_last',
                stages=[dict(id=n,enabled=True) for n in ['joint','aerial','transfer','reference','recovery','joint_recovery','cityflow']])
            path=root/'configs/benchmark_suite.yaml';path.write_text(yaml.safe_dump(base));before=path.read_bytes()
            with patch.object(command,'ROOT',root):paths=command.benchmark_configs()
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(len(paths),2)
            for p,arm in zip(paths,['control','candidate']):
                config=yaml.safe_load(p.read_text())
                self.assertEqual(config['joint_checkpoint'],f'checkpoints/joint_metric/{arm}.best.pt')
                self.assertEqual({s['id'] for s in config['stages'] if s['enabled']},{'joint','recovery','joint_recovery'})


if __name__=='__main__':unittest.main()
