"""Headless end-to-end compatibility and independently recorded stage-3 branches."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from vac.calculation import ClassificationError, classify
from vac.domain.models import MeasurementSet, ProcessingParameters, QualityIntervals
from vac.io import lvm, twf
from vac.processing import preprocessing, corrections, interpretation, pdf, sums
from vac.services.analysis import analyze
from test_calculation import read_table

FIXTURES = ROOT / 'tests/fixtures/stage-4'


class HeadlessTests(unittest.TestCase):
    def test_examples_from_files_to_classification_without_qt_or_writes(self):
        for equipment in ['new', 'old']:
            with self.subTest(equipment=equipment):
                baseline = ROOT / 'tests/baselines/stage-1' / equipment / '01-load'
                settings = json.loads((baseline / 'settings.json').read_text())
                suffix = '' if equipment == 'new' else '_old'
                sample = ROOT / 'пример' / ('новое оборудование' if equipment == 'new' else 'старое оборудование')
                manifest = json.loads((baseline.parent / 'manifest.json').read_text())
                count = 9 if equipment == 'new' else 13
                order = manifest['read_order'][:count]
                # Replay the recorded OS traversal; filesystem order is not portable.
                if equipment == 'new':
                    directories = {}
                    for name in order:
                        relative = Path(name)
                        directories.setdefault(relative.parent, []).append(relative.name)
                    walk = [(str(sample / parent), [], names) for parent, names in directories.items()]
                    with patch('vac.io.lvm.os.walk', return_value=walk):
                        source, files = lvm.load_directory(sample, settings['spinBox_mean'])
                else:
                    with patch('vac.io.twf.os.listdir', return_value=order):
                        source, files = twf.load_directory(sample)
                self.assertEqual(files, [Path(p).name for p in order])
                parameters = ProcessingParameters(settings['doubleSpinBox_int_min'], settings['doubleSpinBox_int_max'],
                    settings['doubleSpinBox_coeff_norm' + suffix], settings['doubleSpinBox_coeff_func' + suffix],
                    settings['spinBox_coeff_dif_res' + suffix], settings['spinBox_mean_win' + suffix])
                original_signals, original_stats, original_raw = source.signals.copy(deep=True), source.statistics.copy(deep=True), source.original
                expected = read_table(baseline, 'all_signals' + suffix)
                expected_stats = read_table(baseline, 'all_stat' + suffix)
                with tempfile.TemporaryDirectory() as tmp, patch.object(pd.DataFrame, 'to_csv', side_effect=AssertionError('Unexpected export')):
                    previous = Path.cwd()
                    try:
                        os.chdir(tmp)
                        for _ in range(2):
                            result = analyze(source, parameters)
                            actual = result.measurements
                            self.assertEqual(list(actual.signals), list(expected))
                            for col in expected:
                                np.testing.assert_allclose(actual.signals[col], expected[col], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=str(col))
                            pd.testing.assert_frame_equal(actual.statistics, expected_stats)
                            pd.testing.assert_frame_equal(source.signals, original_signals)
                            pd.testing.assert_frame_equal(source.statistics, original_stats)
                            pd.testing.assert_frame_equal(source.original, original_raw)
                        self.assertEqual(list(Path(tmp).iterdir()), [])
                    finally:
                        os.chdir(previous)
                self.assertFalse(any(name in sys.modules for name in ['PyQt5', 'pyqtgraph', 'matplotlib', 'docx']))

    def test_pdf_against_stage3_oracle_and_repeat(self):
        metadata = json.loads((FIXTURES / 'metadata.json').read_text())
        self.assertEqual(hashlib.sha256((FIXTURES / 'arrays.npz').read_bytes()).hexdigest(), metadata['arrays_sha256'])
        with np.load(FIXTURES / 'arrays.npz') as arrays:
            for equipment in ['new', 'old']:
                info = metadata['pdf'][equipment]
                measurements = MeasurementSet(equipment)
                source = pd.DataFrame(arrays['input_pdf'], columns=['depth', 1, 2, 3])
                measurements.interval_min, measurements.interval_max = info['interval']
                parameters = ProcessingParameters(0, 20, 0, 0, info['difference_scale'], info['window'])
                expected = pd.DataFrame(arrays['pdf_' + equipment], columns=info['columns'])
                for _ in range(2):
                    measurements.signals = source.copy(deep=True)
                    pdf.calculate_pdf(measurements, parameters, [1, 2, 3])
                    pd.testing.assert_frame_equal(measurements.signals, expected, check_exact=False, rtol=1e-12, atol=1e-12)
                with self.assertRaises(ValueError):
                    pdf.calculate_pdf(measurements, parameters, [1])

    def test_corrections_against_stage3_oracle_and_preserve_automatic(self):
        metadata = json.loads((FIXTURES / 'metadata.json').read_text())
        with np.load(FIXTURES / 'arrays.npz') as arrays:
            source = pd.DataFrame(arrays['input_correction'], columns=metadata['correction_columns'])
            before = source.copy(deep=True)
            for key, info in metadata['corrections'].items():
                with self.subTest(operation=key):
                    function = getattr(corrections, info['operation'])
                    result = function(source, info['x1'], info['x2'], info['y'], info['shift'], info['linear'])
                    expected = pd.DataFrame(arrays[key], columns=info['columns'])
                    pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
                    pd.testing.assert_frame_equal(source, before)
                    pd.testing.assert_frame_equal(function(source, info['x1'], info['x2'], info['y'], info['shift'], info['linear']), result)
                    committed = corrections.fix_corr(result)
                    pd.testing.assert_series_equal(committed.first_sig, result.corr_sig, check_names=False)
                    self.assertTrue((committed.corr_coeff == 1).all())
            added = corrections.plus_func(source)
            np.testing.assert_allclose(added.corr_sig, source.first_sig + source.func)
            pd.testing.assert_frame_equal(source, before)

    def test_interval_boundaries_and_errors(self):
        measurements = MeasurementSet('new')
        measurements.signals = pd.DataFrame({'depth': np.arange(120, dtype=float), '1_diff_result': np.arange(120), '1_func': np.ones(120)})
        measurements.interval_min, measurements.interval_max = 0, 100
        prepared = interpretation.prepare_cement(measurements, 1)
        self.assertEqual(prepared.depth.tolist(), list(range(86, 100)))
        measurements.interval_min = 90
        self.assertEqual(interpretation.prepare_cement(measurements, 1).depth.tolist(), list(range(90, 100)))
        measurements.interval_min = 100
        self.assertTrue(interpretation.prepare_cement(measurements, 1).empty)
        intervals = QualityIntervals()
        changed = corrections.add_interval(intervals, prepared.depth, 87, 90, 'undefined')
        self.assertEqual(intervals.undefined, [])
        self.assertEqual(changed.undefined, [[1, 4]])
        classified = classify(pd.Series([0.] * 100), 1, 2, True, QualityIntervals(undefined=[[20, 60]]))
        self.assertEqual(classified[59], 3)
        self.assertEqual(classified[60], 0)
        with self.assertRaises(ClassificationError):
            interpretation.classify_channel(pd.Series([], dtype=float), 1, 2, False)
        errors = []
        self.assertEqual(interpretation.classify_channel(pd.Series([], dtype=float), 1, 2, False, errors), [])
        self.assertEqual(len(errors), 1)
        for value, expected in [(0.323299, 'red'), (0.3233, 'yellow'), (0.680599, 'yellow'), (0.6806, 'green')]:
            self.assertEqual(interpretation.damping_class(value), expected)

    def test_readers_and_explicit_file_order(self):
        with tempfile.TemporaryDirectory(prefix='VAC кириллица ') as tmp:
            root = Path(tmp)
            path = root / 'данные.TWF'
            path.write_bytes(b'\0' * 130 + struct.pack('<4i', -2147483648, -1, 0, 2147483647))
            self.assertEqual(twf.read_twf(path), (-2147483648, -1, 0, 2147483647))
            self.assertEqual(twf.read_twf_record(path).header, b'\0' * 130)
            malformed = root / 'неполный.TWF'
            for data in [b'short', b'\0' * 130, b'\0' * 131]:
                malformed.write_bytes(data)
                with self.assertRaises(ValueError):
                    twf.read_twf(malformed)
            malformed.unlink()
            invalid_lvm = root / 'неверный.lvm'
            invalid_lvm.write_text('0\tглубина\tсигнал\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                lvm.read_lvm(invalid_lvm)
            depth = preprocessing.twf_depth(4)
            interpolated = preprocessing.interpolate_twf(twf.read_twf(path), depth)
            self.assertEqual(depth[0], 0)
            self.assertEqual(interpolated[0], -2147483648)
            self.assertLess(depth[-1], 4 * 0.515)
            with self.assertRaises(ValueError):
                lvm.load_directory(root, 1)
            with self.assertRaises(ValueError):
                twf.load_directory(root)
            # Make TWF order explicit to prove there is no hidden sort/noise renaming.
            names = [f'{i}.TWF' for i in range(13)]
            for i, name in enumerate(names):
                (root / name).write_bytes(b'\0' * 130 + struct.pack('<4i', i + 1, i + 2, i + 3, i + 4))
            order = names[6:] + names[:6]
            with patch('vac.io.twf.os.listdir', return_value=order):
                measurements, files = twf.load_directory(root)
            self.assertEqual(files, order)
            self.assertEqual(measurements.statistics.at[1, 'name'], order[1])
            before = measurements.original
            measurements.signals.loc[0, 1] = -999
            pd.testing.assert_frame_equal(measurements.original, before)

    def test_sums_reject_unequal_depth_and_no_selection(self):
        parameters = ProcessingParameters(0, 5, 0, 0, 1, 1)
        for equipment, function in [('new', sums.sum_signals), ('old', sums.sum_signals_old)]:
            measurements = MeasurementSet(equipment)
            key1, key2 = ('1_envelope', '2_envelope') if equipment == 'new' else (1, 2)
            measurements.signals = pd.DataFrame({'depth': np.arange(10), key1: np.ones(10), key2: np.ones(10)})
            measurements.statistics.at[1, 'max_h'] = 5
            measurements.statistics.at[2, 'max_h'] = 6
            slot = 9 if equipment == 'new' else 13
            self.assertFalse(function(measurements, parameters, [], slot))
            with self.assertRaises(sums.SignalLengthError):
                function(measurements, parameters, [1, 2], slot)


if __name__ == '__main__':
    unittest.main()
