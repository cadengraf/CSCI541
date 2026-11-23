"""CLI checks must never launch a training, evaluation or download job."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from utils.cli.commands import dispatch
from utils.cli.config import ROOT,CONFIG,path_from_config


class CommandTests(unittest.TestCase):
    def test_no_mode_and_help_are_safe_for_all_four_entry_points(self):
        for name in ['train','benchmark','demo','prepare']:
            for flags in [[],['--help']]:
                result=subprocess.run([sys.executable,str(ROOT/f'{name}.py'),*flags],cwd=ROOT,capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('Commands:',result.stdout)
                self.assertIn('user-launched',result.stdout)

    def test_direct_routes_forward_flags_without_running_implementation(self):
        calls=[]
        module=SimpleNamespace(main=lambda args:calls.append(args))
        with patch('utils.cli.commands.import_module',return_value=module) as loader:
            dispatch('benchmark',['cityflow','--config','example.yaml'])
            loader.assert_called_once_with('utils.experiments.cityflow.command')
        self.assertEqual(calls,[['--config','example.yaml']])

    def test_report_command_cannot_be_given_inference_flags(self):
        calls=[]
        with patch('utils.cli.commands.import_module',return_value=SimpleNamespace(run=lambda:calls.append('report'))):
            dispatch('benchmark',['milestones'])
            with self.assertRaises(SystemExit):dispatch('benchmark',['milestones','--check'])
        self.assertEqual(calls,['report'])

    def test_legacy_arguments_and_sys_argv_are_restored(self):
        previous=sys.argv;seen=[]
        def fake():seen.append(sys.argv.copy())
        with patch('utils.cli.commands.import_module',return_value=SimpleNamespace(main=fake)):
            dispatch('train',['compare-crops','--help'])
        self.assertIs(sys.argv,previous)
        self.assertEqual(seen,[['train.py','compare-crops','--help']])

    def test_vehicle_stage_compatibility_selects_retrieval(self):
        with patch('utils.cli.commands.import_module',return_value=SimpleNamespace(main=lambda args:args)) as loader:
            self.assertEqual(dispatch('benchmark',['vehicle','--stage','final']),['--stage','final'])
            loader.assert_called_once_with('utils.vehicle_reid.retrieval')

    def test_config_paths_remain_at_project_root(self):
        self.assertEqual(ROOT,Path(__file__).resolve().parents[3])
        self.assertEqual(CONFIG,ROOT/'config.yaml')
        self.assertEqual(path_from_config('data/example'),ROOT/'data/example')
        self.assertIsNone(path_from_config(None))


if __name__=='__main__':unittest.main()
