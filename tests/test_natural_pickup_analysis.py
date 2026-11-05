"""Ground paired natural Q-and-pickup claims on strict committed journals."""
import importlib.util,json,pathlib,tempfile,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('transfer_report',ROOT/'tools/analyze-natural-pickup-transfer.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

class NaturalTransferReportTest(unittest.TestCase):
    def test_existing_real_seed24_journals_validate(self):
        a=ROOT/'logs/fresh_eval_naturalpair24shadow1105.jsonl'
        b=ROOT/'logs/fresh_eval_naturalpair24live1105.jsonl'
        if a.is_file() and b.is_file():
            with patch.object(report,'print',create=True):
                r=report.compare('naturalpair24shadow1105','naturalpair24live1105')
            self.assertEqual(r['shadow']['worlds'],1)
            self.assertGreater(r['visible_memory']['pickup_actions'],0)
            self.assertEqual(r['visible_memory']['pickups_during_pickup_actions'],0)

    def test_invalid_journal_is_rejected_without_faking_success(self):
        with tempfile.TemporaryDirectory() as d,patch.object(report,'ROOT',pathlib.Path(d)):
            p=pathlib.Path(d)/'logs';p.mkdir()
            (p/'fresh_eval_broken.jsonl').write_text(json.dumps({'stage':'config','mode':'eval'})+'\n')
            with self.assertRaises(ValueError):report.read('broken')
            with self.assertRaises(FileNotFoundError):report.read('no_such_tag')

if __name__=='__main__':unittest.main()
