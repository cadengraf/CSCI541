import unittest
import torch
from utils.experiments.aerial_transfer.data import partition
from utils.experiments.aerial_transfer.evaluation import evaluate


class TransferTests(unittest.TestCase):
    def test_references_precede_every_query_and_track_zero_is_valid(self):
        protocol=dict(minimum_gap_frames=60,minimum_reference_side=16,minimum_query_side=12,
                      query_spacing_frames=30,maximum_queries_per_identity=8)
        rows=[dict(id='seq:0',frame=f,box=[0,0,20,20],truncation=0,occlusion=0) for f in [1,69,70,100,130,145,160,190]]
        gallery,query=partition(rows,200,protocol)
        self.assertEqual([r['frame'] for r in gallery],[1])
        self.assertEqual([r['frame'] for r in query],[130,160,190])
        self.assertGreaterEqual(min(r['frame'] for r in query)-max(r['frame'] for r in gallery),60)

    def test_unknowns_withholding_and_sequence_isolation(self):
        gallery=[];query=[];gv=[];qv=[]
        for sequence in ['a','b']:
            for i in range(4):
                gallery.append(dict(id=f'{sequence}:{i}',sequence=sequence,frame=1));gv.append(torch.eye(4)[i])
            for i in range(5):
                query.append(dict(id=f'{sequence}:{i}',sequence=sequence,frame=100,box=[0,0,20,20],occlusion=0))
                qv.append(torch.eye(4)[min(i,3)])
        result,rows=evaluate(dict(query=torch.stack(qv),gallery=torch.stack(gv)),dict(query=query,gallery=gallery),
                             {'fixed':dict(threshold=.9,margin=.1)},dict(withheld_every=5))
        self.assertEqual(result['retrieval']['queries'],8)
        self.assertEqual(result['retrieval']['rank1'],1.)
        p=result['open_set']['fixed'];self.assertEqual(p['known_queries'],6)
        self.assertEqual(p['natural_unknown']['queries'],2)
        self.assertEqual(p['natural_unknown']['accepted'],2)
        self.assertEqual(p['withheld_unknown']['accepted'],0)
        for row in rows:self.assertEqual(gallery[row['open_prediction']]['sequence'],row['sequence'])


if __name__=='__main__':unittest.main()
