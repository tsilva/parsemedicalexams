import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/stratified_sampling.py'
SPEC = importlib.util.spec_from_file_location('audit_sampler_fixture', SCRIPT)
sampler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampler)
CONFIG = {'priority_tags': {}, 'category_balance': {}, 'random_seed': 42}


def doc(name, profile='fixture', **fields):
    return {'profile': profile, 'doc_stem': name, 'doc_dir': f'/synthetic/{profile}/{name}',
            'page_count': 1, 'min_confidence': None, 'tags': [], **fields}


def report(*documents):
    return {'hard_failure': [{'file': entry['doc_dir'] + '/page.md', 'doc': entry['doc_stem'],
                              'snippets': ['Synthetic failure marker']} for entry in documents]}


class SamplingTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(42)

    def sample(self, inventory, target, required=(), config=CONFIG):
        with contextlib.redirect_stdout(io.StringIO()):
            return sampler.stratified_sample(inventory, target, config, required)

    def test_all_reported_failures_are_included_before_random_fill(self):
        inventory = [doc(f'item-{index}') for index in range(8)]
        required = sampler.mandatory_documents(inventory, report(inventory[-1]))
        selected = self.sample(inventory, 3, required)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0]['doc_stem'], inventory[-1]['doc_stem'])
        self.assertEqual(selected[0]['sample_reason'], 'SENTINEL (all)')

    def test_mandatory_failures_can_exceed_target_without_being_dropped(self):
        inventory = [doc('first'), doc('second'), doc('third')]
        required = sampler.mandatory_documents(inventory, report(*inventory))
        self.assertEqual(len(self.sample(inventory, 1, required)), 3)

    def test_existing_priority_coverage_is_preserved_after_mandatory_cases(self):
        inventory = [doc('priority', tags=['LOW_CONF']), doc('failed')]
        config = {**CONFIG, 'priority_tags': {'LOW_CONF': {'coverage': 1.0, 'priority': 1000}}}
        selected = self.sample(inventory, 1, {sampler.document_key(inventory[1])}, config)
        self.assertEqual({entry['doc_stem'] for entry in selected}, {'priority', 'failed'})

    def test_duplicate_failure_hits_and_groups_do_not_duplicate_documents(self):
        inventory = [doc('first')]
        evidence = report(*inventory)
        evidence['other'] = copy.deepcopy(evidence['hard_failure'])
        required = sampler.mandatory_documents(inventory, evidence)
        self.assertEqual(len(self.sample(inventory, 1, required)), 1)

    def test_same_stem_in_different_profiles_remains_two_documents(self):
        inventory = [doc('same', profile='one'), doc('same', profile='two')]
        required = sampler.mandatory_documents(inventory, report(*inventory))
        self.assertEqual(len(self.sample(inventory, 1, required)), 2)

    def test_duplicate_inventory_identity_is_not_silently_discarded(self):
        with self.assertRaises(ValueError):
            self.sample([doc('same'), doc('same')], 2)

    def test_source_inventory_objects_are_not_mutated(self):
        inventory = [doc('first'), doc('second')]
        before = copy.deepcopy(inventory)
        self.sample(inventory, 2)
        self.assertEqual(inventory, before)

    def test_missing_ambiguous_and_mismatched_report_identity_fail(self):
        first = doc('first')
        for inventory, evidence in [([first], report(doc('absent'))),
                                    ([first, {**first, 'profile': 'another'}], report(first)),
                                    ([first], {'hard_failure': [{'file': '/synthetic/first/page.md', 'doc': 'wrong'}]})]:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                sampler.mandatory_documents(inventory, evidence)

    def test_malformed_reports_and_missing_directory_fail(self):
        for evidence in [[], {'failure': None}, {'failure': [None]}, {'failure': [{'doc': 'one'}]}]:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                sampler.mandatory_documents([doc('one')], evidence)
        entry = doc('one');del entry['doc_dir']
        with self.assertRaises(ValueError):
            sampler.mandatory_documents([entry], report(doc('one')))

    def test_empty_report_and_zero_goal_are_valid(self):
        self.assertEqual(sampler.mandatory_documents([doc('one')], {}), set())
        self.assertEqual(self.sample([doc('one')], 0), [])

    def test_unknown_confidence_is_reported_as_unknown_in_both_views(self):
        inventory = [doc('one'), doc('two', min_confidence=0.95)]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            sampler.print_sample_summary(inventory, inventory)
            sampler.print_sample_list(inventory)
        self.assertIn('Unknown/invalid: 1 documents', stream.getvalue())
        self.assertIn('Conf: unknown', stream.getvalue())

    def test_nonfinite_out_of_range_and_nonnumeric_confidence_remain_unknown(self):
        for value in [None, True, '0.9', float('nan'), float('inf'), -1, 2]:
            self.assertIsNone(sampler.known_confidence(doc('one', min_confidence=value)))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.inventory = self.root / 'inventory.json'
        self.config = self.root / 'config.json'
        self.sentinel = self.root / 'sentinel.json'
        self.output = self.root / 'sample.json'
        self.documents = [doc(f'item-{index}') for index in range(12)]
        self.inventory.write_text(json.dumps(self.documents))
        self.config.write_text(json.dumps(CONFIG))
        self.sentinel.write_text(json.dumps(report(self.documents[-1])))

    def run_cli(self, *extra):
        return subprocess.run([sys.executable, '-B', str(SCRIPT), '--inventory', str(self.inventory),
                               '--config', str(self.config), '--sentinel-report', str(self.sentinel),
                               '--target-size', '3', '--output', str(self.output), *extra],
                              capture_output=True, text=True)

    def test_cli_consumes_report_and_preserves_metadata_inputs(self):
        before = [path.read_bytes() for path in [self.inventory, self.config, self.sentinel]]
        result = self.run_cli('--seed', '0', '--list')
        self.assertEqual(result.returncode, 0, result.stderr)
        selected = json.loads(self.output.read_text())
        self.assertEqual(selected[0]['doc_stem'], self.documents[-1]['doc_stem'])
        self.assertEqual(before, [path.read_bytes() for path in [self.inventory, self.config, self.sentinel]])
        self.assertIn('Using random seed: 0', result.stdout)

    def test_seed_zero_is_repeatable_and_no_longer_replaced_by_default(self):
        self.assertEqual(self.run_cli('--seed', '0').returncode, 0)
        first = self.output.read_bytes()
        self.assertEqual(self.run_cli('--seed', '0').returncode, 0)
        self.assertEqual(self.output.read_bytes(), first)

    def test_missing_explicit_configuration_stops_without_a_sample(self):
        self.config.unlink()
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertFalse(self.output.exists())

    def test_unmatched_failure_stops_without_a_sample(self):
        self.sentinel.write_text(json.dumps(report(doc('missing'))))
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertFalse(self.output.exists())

    def test_direct_symlink_and_hardlink_input_collisions_are_rejected(self):
        for source in [self.inventory, self.config, self.sentinel]:
            for kind in ['direct', 'symlink', 'hardlink']:
                output = source if kind == 'direct' else self.root / (source.stem + '-' + kind)
                if kind == 'symlink':output.symlink_to(source)
                if kind == 'hardlink':output.hardlink_to(source)
                before = source.read_bytes()
                result = self.run_cli('--output', str(output))
                self.assertEqual(result.returncode, 2)
                self.assertEqual(source.read_bytes(), before)

    def test_existing_scanner_report_feeds_the_sampler_without_manual_list_edits(self):
        selected = self.documents[-1]
        directory = self.root / 'extracted' / selected['doc_stem']
        directory.mkdir(parents=True)
        (directory / 'page.md').write_text('Request too large. Try with a smaller file.\n')
        selected['doc_dir'] = str(directory)
        self.inventory.write_text(json.dumps(self.documents))
        scanner = SCRIPT.with_name('sentinel_scan.py')
        result = subprocess.run([sys.executable, '-B', str(scanner), '--output-dir', str(directory.parent),
                                 '--json-output', str(self.sentinel)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.run_cli().returncode, 0)
        sample = json.loads(self.output.read_text())
        self.assertEqual(sample[0]['doc_stem'], selected['doc_stem'])
        self.assertEqual(sample[0]['sample_reason'], 'SENTINEL (all)')


if __name__ == '__main__':
    unittest.main()
