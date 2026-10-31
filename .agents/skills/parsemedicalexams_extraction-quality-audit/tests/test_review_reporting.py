import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import report_generator as reporter
import verification_framework as verifier


def doc(name, profile='fixture', tags=()):
    return {'profile': profile, 'doc_stem': name, 'doc_dir': f'/synthetic/{profile}/{name}',
            'page_count': 1, 'tags': list(tags), 'min_confidence': None}


def result(document, score=8):
    return {**document, 'verification': {'transcription_quality': {'overall_score': score},
                                        'recommendations': ['Review the recorded fixture issue']}}


class VerificationIdentityTests(unittest.TestCase):
    def test_new_template_uses_null_for_unrated_dimensions(self):
        template = verifier.create_verification_template()
        for dimension in ('metadata_accuracy', 'transcription_quality', 'summary_quality', 'multi_page_coherence'):
            scores = [value for field, value in template[dimension].items() if field == 'score' or field.endswith('_score')]
            self.assertTrue(scores)
            self.assertTrue(all(value is None for value in scores))
        self.assertIsNone(template['transcription_quality']['overall_score'])

    def test_same_filename_from_one_profile_does_not_complete_another(self):
        first, second = doc('same', 'one'), doc('same', 'two')
        keys = verifier.verified_document_keys([result(first)])
        self.assertIn(('one', 'same'), keys)
        self.assertNotIn(('two', 'same'), keys)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            verifier.print_progress([first, second], [result(first)])
        self.assertIn('Verified: 1 (50.0%)', stream.getvalue())
        self.assertIn('Remaining: 1 (50.0%)', stream.getvalue())

    def test_unbound_old_results_are_retained_without_marking_a_profile_done(self):
        records = [{'doc_stem': 'same'}]
        self.assertEqual(verifier.verified_document_keys(records), set())
        self.assertEqual(records, [{'doc_stem': 'same'}])

    def test_out_of_sample_records_do_not_inflate_progress(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            verifier.print_progress([doc('one')], [result(doc('elsewhere'))])
        self.assertIn('Verified: 0 (0.0%)', stream.getvalue())

    def test_progress_includes_zero_and_excludes_missing_ratings(self):
        documents = [doc('zero'), doc('missing')]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            verifier.print_progress(documents, [result(documents[0], 0), result(documents[1], None)])
        self.assertIn('Average Quality Score: 0.00/10', stream.getvalue())

    def test_empty_sample_progress_does_not_divide_by_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            verifier.print_progress([], [])


class ReviewReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name).resolve() / 'report.md'
        self.low = doc('low', tags=['LOW_CONF'])
        self.complex = doc('complex', tags=['COMPLEX'])
        self.normal = doc('normal')
        self.inventory = [self.low, self.complex, self.normal]

    def generate(self, sample, records, sentinel=None):
        with contextlib.redirect_stdout(io.StringIO()):
            reporter.generate_report(self.inventory, sample, records, self.output, sentinel)
        return self.output.read_text()

    def test_coverage_uses_actual_denominators_and_matched_review_records(self):
        text = self.generate([self.low, self.normal], [result(self.low)])
        self.assertIn('Known critical cases selected: 1/2 (50.0%)', text)
        self.assertIn('matching verification record: 1/2 (50.0%)', text)
        self.assertNotIn('**Confidence Level:**', text)
        self.assertNotIn('95%', text)

    def test_supplied_sentinel_report_adds_known_failures_to_the_denominator(self):
        sentinel = {'failure': [{'file': self.normal['doc_dir'] + '/page.md', 'doc': 'normal'}]}
        text = self.generate([self.low, self.normal], [result(self.low)], sentinel)
        self.assertIn('Known critical cases selected: 2/3 (66.7%)', text)
        self.assertIn('matching verification record: 1/3 (33.3%)', text)

    def test_no_known_critical_cases_is_not_claimed_as_full_coverage(self):
        self.inventory = [self.normal]
        text = self.generate([self.normal], [])
        self.assertIn('N/A (no known critical cases)', text)
        appendix = text.split('### A. Coverage and Evidence Limits', 1)[1].split('### B. Methodology', 1)[0]
        self.assertNotIn('100%', appendix)

    def test_unbound_and_other_profile_records_are_excluded_visibly(self):
        text = self.generate([self.low], [result(self.low), result(doc('low', 'other')), {'doc_stem': 'low'}])
        self.assertIn('**Documents Verified:** 1', text)
        self.assertIn('excluded from aggregates: 2', text)

    def test_duplicate_results_and_unknown_sample_identity_stop_before_writing(self):
        for sample, records in [([self.low], [result(self.low), result(self.low)]),
                                ([doc('absent')], [])]:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                self.generate(sample, records)
            self.assertFalse(self.output.exists())

    def test_real_zero_scores_are_included_and_missing_scores_remain_na(self):
        records = [result(self.low, 10), result(self.normal, 0)]
        text = self.generate([self.low, self.normal], records)
        self.assertIn('**Average Observed Quality Score:** 5.00/10', text)
        self.assertIn('| Overall Quality | 5.0/10', text)
        self.assertIn('| Metadata Accuracy | N/A | N/A |', text)

    def test_no_scores_do_not_create_a_failing_grade_or_readiness_claim(self):
        text = self.generate([self.low], [])
        self.assertIn('N/A (no observed overall score)', text)
        self.assertNotIn('F (Failing)', text)
        self.assertNotIn('None Required', text)
        self.assertNotIn('No critical issues identified', text)
        self.assertIn('No overall score is available', text)

    def test_high_scores_do_not_clear_recorded_issues_or_authorize_actions(self):
        text = self.generate([self.low], [result(self.low, 10)])
        self.assertNotIn('No immediate changes are required', text)
        self.assertNotIn('Minimal to no critical errors', text)
        self.assertIn('does not establish production readiness', text)
        self.assertIn('Review the recorded fixture issue', text)

    def test_cli_uses_sentinel_evidence_and_preserves_input_bytes(self):
        root = self.output.parent
        sentinel = {'failure': [{'file': self.normal['doc_dir'] + '/page.md', 'doc': 'normal'}]}
        inputs = {'inventory': self.inventory, 'sample': [self.low, self.normal],
                  'results': [result(self.low)], 'sentinel-report': sentinel}
        args, before = [], {}
        for flag, value in inputs.items():
            path = root / (flag + '.json')
            path.write_text(json.dumps(value))
            before[path] = path.read_bytes()
            args.extend(['--' + flag, str(path)])
        command = [sys.executable, str(SCRIPTS / 'report_generator.py'), *args, '--output', str(self.output)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('Known critical cases selected: 2/3 (66.7%)', self.output.read_text())
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        # A report may not replace any supplied metadata, including through aliases.
        source = root / 'inventory.json'
        for target in (source, root / 'alias.json', root / 'hardlink.json'):
            if target.name == 'alias.json':
                target.symlink_to(source)
            elif target.name == 'hardlink.json':
                import os
                os.link(source, target)
            blocked = subprocess.run(command[:-1] + [str(target)], capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertEqual(source.read_bytes(), before[source])

    def test_invalid_scores_remain_unavailable(self):
        for value in [None, True, '9', -1, 11, float('nan')]:
            self.assertIsNone(reporter.known_score(value))


if __name__ == '__main__':
    unittest.main()
