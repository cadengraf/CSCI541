from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml
from utils.benchmarking.suite import command, report, scorecard


class ScorecardTests(unittest.TestCase):
    def tables(self, value):
        return [dict(key='retrieval',title='Retrieval',headers=['Model','Split','Path','mAP','R1'],note='',
                     rows=[['Joint','VRAI held-out','mask',value,20.]])]

    def test_regression_records_latest_and_preserves_best(self):
        ledger={}
        for run,value in [('a',10.),('b',9.),('c',11.)]:
            changes=scorecard.ingest(ledger,run,self.tables(value),lambda k,r:{'split':'fixed'})
            self.assertEqual(bool(changes),run!='b')
        entry=next(iter(ledger['rows'].values()))
        self.assertEqual(entry['best']['run'],'c')
        self.assertEqual(entry['previous']['run'],'b')
        self.assertEqual(entry['delta_previous'],2.)
        self.assertEqual(entry['delta_best'],1.)
        scorecard.ingest(ledger,'c',self.tables(11.),lambda k,r:{'split':'fixed'})
        self.assertEqual(entry['previous']['run'],'b')

    def test_different_protocols_do_not_compete(self):
        ledger={}
        scorecard.ingest(ledger,'a',self.tables(90.),lambda k,r:{'query':'one'})
        scorecard.ingest(ledger,'b',self.tables(10.),lambda k,r:{'query':'two'})
        self.assertEqual(len(ledger['rows']),2)
        self.assertEqual({e['best']['run'] for e in ledger['rows'].values()},{'a','b'})

    def test_more_recoveries_with_more_false_claims_does_not_promote(self):
        ledger={}
        def rows(recovered,false):
            return [dict(key='recovery',title='Recovery',headers=[],note='',rows=[
                ['Joint','natural','query_pool',recovered,0,0,0,4,false,0,0,0]])]
        scorecard.ingest(ledger,'a',rows(2,0),lambda k,r:{})
        self.assertFalse(scorecard.ingest(ledger,'b',rows(3,1),lambda k,r:{}))
        self.assertEqual(next(iter(ledger['rows'].values()))['best']['run'],'a')

    def test_html_changes_only_on_improvement_and_never_writes_per_run_pages(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            patches=[patch.object(module,'ROOT',root) for module in [command,report,scorecard]]
            for p in patches:p.start();self.addCleanup(p.stop)
            config=dict(run='one',output='metrics/suites',joint_checkpoint='best.pt')
            state=dict(status='complete',stages=dict(joint=dict(status='complete')))
            def record(run,value):
                config['run']=run
                base=root/config['output']/run/'joint';base.mkdir(parents=True)
                (base/'vrai.csv').write_text(f'model,pooling,mAP,Rank1\njoint,mask,{value},20\n')
                scorecard.publish(config,state)
            record('one',10.)
            first=(root/'milestones.html').read_bytes()
            record('two',9.)
            self.assertEqual(first,(root/'milestones.html').read_bytes())
            ledger=yaml.safe_load((root/'metrics/scorecard/scores.yaml').read_text())
            entry=next(iter(ledger['rows'].values()))
            self.assertEqual(entry['latest']['run'],'two')
            self.assertEqual(entry['best']['run'],'one')
            record('three',11.)
            self.assertNotEqual(first,(root/'milestones.html').read_bytes())
            self.assertEqual(list(root.rglob('*.html')),[root/'milestones.html'])


if __name__=='__main__':unittest.main()
