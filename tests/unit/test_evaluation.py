"""Analytic fixtures, deliberately unrelated to field-quality ground truth."""
from copy import deepcopy
import sys
import unittest
import json
from pathlib import Path
import subprocess
import tempfile

from vac.evaluation import evaluate, score


def document():
    return dict(schema='vac-evaluation-v1', provenance=dict(
        algorithm_version='synthetic-v1', code_revision='test-only',
        reference_sha256='0' * 64, parameters={}, annotation_revision='synthetic-v1',
        hypothesis='test only'), studies=[dict(well_id='w1', study_id='s1',
        input_sha256='1' * 64, split='evaluation', axis='uniform_sample_index',
        truth=[0, 0, 1, 1, 2, 2, 3, 3], prediction=[0, 0, 1, 1, 2, 2, 3, 3])])


class EvaluationTests(unittest.TestCase):
    def test_perfect_and_missing_class(self):
        result = score([0, 0, 1, 1], [0, 0, 1, 1])
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['expert_segment_best_iou'], [1, 1])
        self.assertEqual(result['boundary_mean_distance_samples'], 0)
        self.assertIsNone(result['per_class']['absent']['recall'])

    def test_shift_and_confusion(self):
        result = score([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result['confusion'][0], [1, 1, 0, 0])
        self.assertEqual(result['accuracy'], .75)
        self.assertEqual(result['boundary_mean_distance_samples'], 1)
        self.assertEqual(result['expert_segment_best_iou'], [.5, 2 / 3])

    def test_undefined_and_unmatched_boundaries(self):
        result = score([0, 0, 1, 1], [3, 3, 3, 3])
        self.assertEqual(result['predicted_undefined_fraction'], 1)
        self.assertEqual(result['accuracy'], 0)
        self.assertEqual(result['boundary_unmatched'], 1)
        self.assertIsNone(result['boundary_mean_distance_samples'])
        self.assertEqual(result['expert_segment_best_iou'], [0, 0])

    def test_invalid_labels_and_alignment(self):
        for truth, prediction in [([], []), ([0], [0, 1]), ([None], [0]),
                                  ([True], [0]), ([4], [0]), ([0.0], [0])]:
            with self.assertRaises(ValueError):
                score(truth, prediction)

    def test_split_leakage_and_duplicate_study(self):
        for change in ({'split': 'train', 'study_id': 's2', 'input_sha256': '2' * 64},
                       {'split': 'train', 'study_id': 's2', 'well_id': 'w2'},
                       {'well_id': 'w2'}):
            value = document()
            other = deepcopy(value['studies'][0])
            other.update(change)
            value['studies'].append(other)
            with self.assertRaises(ValueError):
                evaluate(value)

    def test_provenance_and_no_mutation(self):
        value = document()
        before = deepcopy(value)
        self.assertEqual(evaluate(value)['studies']['s1']['accuracy'], 1)
        self.assertEqual(value, before)
        del value['provenance']['annotation_revision']
        with self.assertRaises(ValueError):
            evaluate(value)
        self.assertNotIn('PyQt5', sys.modules)

    def test_cli_repeat_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            source.write_text(json.dumps(document()))
            outputs = [Path(directory) / name for name in ('a.json', 'b.json')]
            for output in outputs:
                subprocess.run([sys.executable, '-m', 'vac.evaluation', str(source),
                                str(output)], check=True, capture_output=True)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            before = outputs[0].read_bytes()
            result = subprocess.run([sys.executable, '-m', 'vac.evaluation', str(source),
                                     str(outputs[0])], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(outputs[0].read_bytes(), before)

    def test_requires_evaluation_and_explicit_axis(self):
        for change in ({'split': 'train'}, {'axis': 'meters'}, {'input_sha256': 'bad'}):
            value = document()
            value['studies'][0].update(change)
            with self.assertRaises(ValueError):
                evaluate(value)


if __name__ == '__main__':
    unittest.main()
