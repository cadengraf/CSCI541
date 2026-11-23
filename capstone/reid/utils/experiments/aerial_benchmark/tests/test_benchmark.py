from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from utils.tracking.common import ROOT,load
from utils.experiments.aerial_benchmark.command import extract,open_scores
from utils.vehicle_reid.operating_points import scores


class TestBenchmark(unittest.TestCase):
    def test_chunked_open_scores_match_existing_cpu_protocol(self):
        torch.manual_seed(1)
        query=F.normalize(torch.randn(12,8),dim=1);gallery=F.normalize(torch.randn(6,8),dim=1)
        qids=[str(i%6) for i in range(12)];gids=[str(i) for i in range(6)]
        expected=scores(query,gallery,qids,gids);actual=open_scores(query,gallery,qids,gids,torch.device('cpu'))
        for key in expected:np.testing.assert_array_equal(expected[key],actual[key])

    def test_interrupted_extraction_resumes_committed_parts_and_reuses_complete_result(self):
        class Model(torch.nn.Module):
            def __init__(self,fail=False):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.calls=0;self.fail=fail
            def forward(self,images):
                self.calls+=1
                if self.fail and self.calls==4:raise RuntimeError('simulated interruption')
                return images.mean((2,3)).repeat(1,1280)*self.weight
        with TemporaryDirectory(dir=ROOT/'data') as directory:
            root=Path(directory);data={'query':[],'gallery':[]}
            for group,n in [('query',12),('gallery',6)]:
                for i in range(n):
                    identity=i%6;path=root/f'{group}_{i}.png'
                    Image.new('RGB',(8,8),(20+identity*30,100,200-identity*20)).save(path)
                    data[group].append(dict(path=str(path),id=str(identity)))
            source=root/'model.pt';torch.save(dict(best_metrics={'vru':{'fit':{'threshold':.3,'margin':.01}}}),source)
            config=dict(state_root=str(root.relative_to(ROOT)/'state'),batch_size=2,checkpoint_batches=2)
            selection={'models':{'toy':{'checkpoint':str(source.relative_to(ROOT)),'sha256':'test'}}}
            conversion=lambda image:torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                extract(config,selection,Model(True),data,conversion,'vru','toy')
            path=ROOT/config['state_root']/'toy/vru/state.pt';state=load(path)
            self.assertEqual(sum(part['rows'] for part in state['parts']['query']),4)
            model=Model();result=extract(config,selection,model,data,conversion,'vru','toy')
            self.assertEqual(result['queries'],12);self.assertEqual(result['gallery'],6)
            state=load(path);self.assertEqual(sum(part['rows'] for part in state['parts']['query']),12)
            before=model.calls;self.assertEqual(result,extract(config,selection,model,data,conversion,'vru','toy'))
            self.assertEqual(before,model.calls)


if __name__=='__main__':unittest.main()
