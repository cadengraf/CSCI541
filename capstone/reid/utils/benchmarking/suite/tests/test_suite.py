"""Fixture-only suite integration: no real evaluator, model or dataset jobs."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml

from utils.benchmarking.suite import command, report, scorecard


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(run='fixture', output='metrics/suites', state_root='data/suites', device='auto',
            continue_on_error=True, joint_checkpoint='weights/best.pt', joint_last_checkpoint='weights/last.pt',
            stages=[dict(id='joint', command='joint', depends=[], enabled=True)])
        self.patches = [patch.object(command, 'ROOT', self.root), patch.object(report, 'ROOT', self.root), patch.object(scorecard, 'ROOT', self.root)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    @property
    def base(self):
        return self.root/'metrics/suites/fixture'

    def config_file(self, config=None):
        path = self.root/'suite.yaml'
        path.write_text(yaml.safe_dump(config or self.config))
        return path

    def fake_job(self, args, log):
        spec = yaml.safe_load((self.root/args[-1]).read_text())
        output = self.root/spec['output']
        output.mkdir(parents=True, exist_ok=True)
        (output/'vrai.csv').write_text('model,pooling,mAP,Rank1\njoint,mask,24.67,21.01\n')
        for name in command.REQUIRED[args[3]]:
            path = output/name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text('{}\n' if path.suffix == '.yaml' else 'fixture\n')
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('fixture evaluator completed\n')

    def test_resume_and_changed_evidence(self):
        with patch.object(command, 'fingerprint', return_value={'sha256':'fixed','files':{}}), patch.object(
            command, 'execute', side_effect=self.fake_job
        ) as execute:
            command.run(self.config)
            command.run(self.config)
            self.assertEqual(execute.call_count, 1)
            (self.base/'joint/vrai.csv').write_text('model,pooling,mAP,Rank1\njoint,mask,0,0\n')
            command.run(self.config)
            self.assertEqual(execute.call_count, 2)
        self.assertEqual(yaml.safe_load((self.base/'suite.yaml').read_text())['status'], 'complete')

    def test_changed_model_fingerprint_requires_new_run(self):
        with patch.object(command, 'fingerprint', return_value={'sha256':'one'}), patch.object(command, 'execute', side_effect=self.fake_job):
            command.run(self.config)
        with patch.object(command, 'fingerprint', return_value={'sha256':'two'}), patch.object(command, 'execute') as execute:
            with self.assertRaisesRegex(ValueError, 'new run name'):
                command.run(self.config)
            execute.assert_not_called()

    def test_failure_blocks_dependent_but_continues_independent_and_retries(self):
        config = deepcopy(self.config)
        config['stages'] += [dict(id='joint_recovery', command='joint-recovery', depends=['joint']),
                             dict(id='aerial', command='aerial', config='aerial.yaml', depends=[])]
        (self.root/'aerial.yaml').write_text('output: old\nstate_root: old\n')
        def job(args, log):
            if args[3] == 'joint':
                raise subprocess.CalledProcessError(1, args)
            self.fake_job(args, log)
        with patch.object(command, 'fingerprint', return_value={'sha256':'fixed'}), patch.object(command, 'execute', side_effect=job):
            with self.assertRaises(SystemExit):
                command.run(config)
        stages = yaml.safe_load((self.base/'suite.yaml').read_text())['stages']
        self.assertEqual(stages['joint']['status'], 'failed')
        self.assertEqual(stages['joint_recovery']['status'], 'blocked')
        self.assertEqual(stages['aerial']['status'], 'complete')
        self.assertTrue((self.root/'milestones.html').exists())

    def test_report_only_never_calls_an_evaluator(self):
        path = self.config_file()
        with patch.object(command, 'execute') as execute, patch.object(command, 'fingerprint') as fingerprints:
            command.main(['--config', str(path), '--report-only'])
        execute.assert_not_called()
        fingerprints.assert_not_called()
        self.assertTrue((self.root/'milestones.html').exists())
        self.assertFalse((self.base/'report.html').exists())

    def test_training_commands_are_rejected(self):
        config = deepcopy(self.config)
        config['stages'][0]['command'] = 'train'
        with self.assertRaisesRegex(ValueError, 'no training'):
            command.read_config(self.config_file(config))

    def test_rebased_configs_preserve_original_and_rewire_dependencies(self):
        source = self.root/'source.yaml'
        original = 'output: metrics/old\nstate_root: data/old\ntransfer_selection: metrics/old/selection.pt\n'
        source.write_text(original)
        stage = dict(id='reference', command='vehicle-reference', config='source.yaml')
        args = command.prepare_stage(self.config, stage)
        generated = yaml.safe_load((self.root/args[-1]).read_text())
        self.assertEqual(source.read_text(), original)
        self.assertEqual(generated['output'], 'metrics/suites/fixture/reference')
        self.assertEqual(generated['transfer_selection'], 'metrics/suites/fixture/transfer/selection.pt')

    def test_streamed_process_logs_and_exit_status(self):
        log = self.root/'logs/fixture.log'
        command.execute([sys.executable, '-c', 'print("fixture stdout")'], log)
        self.assertIn('fixture stdout', log.read_text())
        with self.assertRaises(subprocess.CalledProcessError):
            command.execute([sys.executable, '-c', 'raise SystemExit(3)'], log)

    def test_report_keeps_fractions_and_percentages_in_correct_units(self):
        (self.base/'joint').mkdir(parents=True)
        (self.base/'aerial').mkdir()
        (self.base/'joint/vrai.csv').write_text('model,pooling,mAP,Rank1\njoint,mask,24.67,21.01\n')
        (self.base/'aerial/mixed_baseline.yaml').write_text(yaml.safe_dump([
            dict(variant='mixed_baseline', domain='vru', map=.75, rank1=.65)]))
        rows = report.collect(self.base)[0]['rows']
        self.assertEqual(rows[0][-2:], ['24.67','21.01'])
        self.assertEqual(rows[1][-2:], ['75','65'])

    def test_checkpoint_contents_are_fingerprinted_not_only_file_names(self):
        (self.root/'weights').mkdir()
        for name in ['best.pt','last.pt']:
            (self.root/'weights'/name).write_bytes(b'first')
        first = command.fingerprint(self.config)
        (self.root/'weights/best.pt').write_bytes(b'other')
        self.assertNotEqual(first['sha256'], command.fingerprint(self.config)['sha256'])

    def test_successful_process_without_metrics_is_not_complete(self):
        with patch.object(command, 'fingerprint', return_value={'sha256':'fixed'}), patch.object(command, 'execute'):
            with self.assertRaises(SystemExit):
                command.run(self.config)
        state = yaml.safe_load((self.base/'suite.yaml').read_text())
        self.assertEqual(state['stages']['joint']['status'], 'failed')

    def test_only_completed_suite_scratch_is_removed(self):
        config={**self.config,'state_root':'data/benchmark_suite'}
        scratch=self.root/'data/benchmark_suite/fixture'
        scratch.mkdir(parents=True)
        (scratch/'features.pt').write_bytes(b'fixture')
        predictions=scratch/'joint_predictions';predictions.mkdir()
        (predictions/'0002.txt').write_text('official evaluator input')
        protected=self.root/'data/dataset';protected.mkdir()
        (protected/'original.txt').write_text('preserve')
        command.release_completed_state(config)
        self.assertFalse(scratch.exists())
        self.assertTrue((protected/'original.txt').exists())
        self.assertTrue((self.base/'joint/predictions/0002.txt').exists())
        scratch.mkdir()
        command.release_completed_state({**config,'retain_completed_state':True})
        self.assertTrue(scratch.exists())


if __name__ == '__main__':
    unittest.main()
