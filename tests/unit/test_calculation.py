"""Run the extracted calculation against both numerical legacy fixtures without Qt."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from vac.calculation import calculate_new, calculate_old, classify, get_nearest_value
from vac.domain.models import MeasurementSet, ProcessingParameters, QualityIntervals

ROOT = Path(__file__).resolve().parents[2]


def read_table(directory, name):
    info = json.loads((directory / 'tables.json').read_text())[name]
    result = pd.DataFrame(index=info['index'])
    with np.load(directory / (name + '.npz')) as arrays:
        for i, column in enumerate(info['columns']):
            data = info['data'][str(i)]
            result[column] = arrays[str(i)] if isinstance(data, dict) else data
    if name.startswith('all_stat'):
        for col in result.columns:
            if col != 'name':
                result[col] = pd.to_numeric(result[col]).astype(object)
    return result


class CalculationTests(unittest.TestCase):
    def test_no_qt(self):
        self.assertNotIn('PyQt5', sys.modules)

    def test_both_equipment_and_repeat(self):
        for equipment, calculate in [('new', calculate_new), ('old', calculate_old)]:
            with self.subTest(equipment=equipment):
                directory = ROOT / 'tests/fixtures/current' / equipment / 'calculation'
                suffix = '' if equipment == 'new' else '_old'
                settings = json.loads((directory / 'settings.json').read_text())
                measurements = MeasurementSet(equipment)
                expected = read_table(directory, 'all_signals' + suffix)
                measurements.signals = expected.copy(deep=True)
                measurements.statistics = read_table(directory, 'all_stat' + suffix)
                params = ProcessingParameters(settings['doubleSpinBox_int_min'], settings['doubleSpinBox_int_max'],
                    settings['doubleSpinBox_coeff_norm' + suffix], settings['doubleSpinBox_coeff_func' + suffix],
                    settings['spinBox_coeff_dif_res' + suffix], settings['spinBox_mean_win' + suffix])
                measurements.interval_min = expected.index[expected.depth == get_nearest_value(expected.depth, params.interval_min)][0]
                measurements.interval_max = expected.index[expected.depth == get_nearest_value(expected.depth, params.interval_max)][0]
                measurements.capture_original()
                raw = measurements.original
                expected_stats = measurements.statistics.copy(deep=True)
                # Remove outputs so a no-op implementation cannot pass.
                keep = [c for c in expected if c in ('depth', 'noise') or isinstance(c, int) or str(c).endswith('_envelope')]
                measurements.signals = expected[keep].copy(deep=True)
                for _ in range(2):
                    calculate(measurements, params)
                    for col in measurements.signals:
                        np.testing.assert_allclose(measurements.signals[col], expected[col], rtol=1e-12, atol=1e-12, equal_nan=True)
                    pd.testing.assert_frame_equal(measurements.statistics, expected_stats)
                    pd.testing.assert_frame_equal(measurements.original, raw)

    def test_manual_intervals_are_explicit(self):
        signal = pd.Series([0.] * 100)
        intervals = QualityIntervals(undefined=[[20, 60]])
        edited = classify(signal, 1, 2, True, intervals)
        untouched = classify(signal, 1, 2, True, QualityIntervals())
        self.assertEqual(edited[30], 3)
        self.assertEqual(untouched, [0] * 100)


if __name__ == '__main__':
    unittest.main()
