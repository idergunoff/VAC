"""Function-relative cementograms from real channels and sums, through the GUI."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vac.app import create_window
from vac.services import snapshot


class FunctionCementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='VAC функция ')
        self.previous = Path.cwd()
        os.chdir(self.scratch.name)
        self.window, self.c = create_window()
        self.ui = self.c.ui
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        os.chdir(self.previous)
        self.scratch.cleanup()

    def select_only(self, checkbox):
        for group in (self.ui.new_device, self.ui.old_device):
            for check in group.findChildren(QtWidgets.QCheckBox):
                with QtCore.QSignalBlocker(check):
                    check.setChecked(check is checkbox)

    def assert_function_cement(self, equipment, checkbox):
        self.select_only(checkbox)
        measurements = getattr(self.c.state, equipment)
        number = int(checkbox.text().split()[0])
        maximum = self.c.calc_max_int(measurements.signals, measurements.statistics, number) if equipment == 'new' else measurements.interval_max
        expected = measurements.signals.iloc[max(86, measurements.interval_min):maximum]
        before = measurements.signals.copy(deep=True)
        self.ui.workspace.select(2)
        self.app.processEvents()
        self.assertTrue(self.ui.pushButton_cement_func.isVisible())
        self.ui.pushButton_cement_func.click()
        result = self.c.state.interpretation
        suffix = '' if equipment == 'new' else '_old'
        scale = getattr(self.ui, 'spinBox_coeff_dif_res' + suffix).value()
        np.testing.assert_allclose(result.table.first_sig, expected[str(number) + '_rel_ampl'] * scale)
        np.testing.assert_array_equal(result.table.depth, expected.depth)
        np.testing.assert_array_equal(self.ui.graphicsView.listDataItems()[0].yData, result.table.corr_sig)
        self.assertEqual(result.signal.equipment, equipment)
        self.assertTrue(self.ui.editButton.isEnabled())
        self.assertTrue(self.ui.pushButton_save_cement.isEnabled())
        self.assertIn('по функции', self.ui.label_info.text())
        pd.testing.assert_frame_equal(measurements.signals, before)
        metadata, arrays = snapshot.encode(result)
        restored = snapshot.decode(metadata, arrays)
        pd.testing.assert_frame_equal(restored.automatic_table, result.table)
        return result.table.copy(deep=True)

    def test_real_channels_function_adjustments_and_sums_for_both_devices(self):
        for equipment, suffix, folder, checkbox_name in [
            ('new', '', 'новое оборудование', 'checkBox_signal1'),
            ('old', '_old', 'старое оборудование', 'checkBox_signal_old_1'),
        ]:
            with self.subTest(equipment=equipment):
                with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory',
                                  return_value=str(ROOT / 'пример' / folder)):
                    getattr(self.c, 'open_dir' + suffix)()
                check = getattr(self.ui, checkbox_name)
                measurements = getattr(self.c.state, equipment)
                signals = measurements.signals
                shifted = signals['1_envelope-1' if equipment == 'new' else '1-1']
                # Formula from the last monolithic version (4e305bc): the
                # original relative deviation, differentiated and smoothed.
                expected = ((shifted - signals['1_func']) / shifted).diff().rolling(
                    50, min_periods=1, center=True).mean()
                np.testing.assert_allclose(signals['1_rel_ampl'], expected, equal_nan=True)
                first = self.assert_function_cement(equipment, check)
                getattr(self.ui, 'doubleSpinBox_coeff_func' + suffix).setValue(.2)
                adjusted = self.assert_function_cement(equipment, check)
                self.assertFalse(first.first_sig.equals(adjusted.first_sig))
                getattr(self.ui, 'spinBox_coeff_dif_res' + suffix).setValue(700)
                enlarged = self.assert_function_cement(equipment, check)
                np.testing.assert_allclose(enlarged.first_sig, adjusted.first_sig * 2)

                # The existing conversion retains its derivative-based result.
                self.ui.pushButton_int_cement.click()
                self.assertIn('по производной', self.ui.label_info.text())
                derivative = self.c.state.interpretation.table
                start = max(86, measurements.interval_min)
                np.testing.assert_allclose(derivative.first_sig,
                    signals['1_diff_result'].iloc[start:start + len(derivative)])

                # Legacy sums use the relative deviation without the extra
                # differentiation. Conversion applies the same working gain.
                for slot in range(1, 4):
                    self.select_only(check)
                    getattr(self.ui, 'pushButton_sum' + str(slot) + suffix).click()
                    summed = getattr(self.ui, 'checkBox_sum' + str(slot) + suffix)
                    self.assert_function_cement(equipment, summed)

    def test_small_function_values_threshold_precision_and_switching_methods(self):
        raw = np.r_[np.zeros(86), np.full(40, -.003), np.full(40, .001), np.full(40, .005)]
        for equipment, suffix, checkbox_name in [
            ('new', '', 'checkBox_signal1'), ('old', '_old', 'checkBox_signal_old_1')
        ]:
            with self.subTest(equipment=equipment):
                measurements = getattr(self.c.state, equipment)
                measurements.signals = pd.DataFrame({
                    'depth': np.arange(len(raw), dtype=float), '1_rel_ampl': raw,
                    '1_diff_result': 20 + raw * 1000, '1_func': np.ones(len(raw))})
                measurements.interval_min, measurements.interval_max = 0, len(raw) - 1
                measurements.statistics.at[1, 'max_h'] = len(raw) - 1
                measurements.statistics.at[1, 'k_damp'] = .5
                with QtCore.QSignalBlocker(self.ui.doubleSpinBox_int_max):
                    self.ui.doubleSpinBox_int_max.setValue(len(raw) - 1)
                check = getattr(self.ui, checkbox_name)
                check.setEnabled(True)
                self.select_only(check)

                for button in (self.ui.pushButton_int_cement, self.ui.pushButton_cement_func) * 2:
                    button.click()
                    table = self.c.state.interpretation.table
                    low, high = table.first_sig.min(), table.first_sig.max()
                    self.assertAlmostEqual(self.ui.doubleSpinBox_defect1.value(), low + (high-low)/3, places=4)
                    self.assertAlmostEqual(self.ui.doubleSpinBox_defect2.value(), low + 2*(high-low)/3, places=4)
                np.testing.assert_allclose(table.corr_sig, raw[86:-1] * 350)
                self.ui.doubleSpinBox_defect1.setValue(.1234)
                self.ui.doubleSpinBox_defect2.setValue(.5678)
                self.assertEqual(self.ui.doubleSpinBox_defect1.value(), .1234)
                self.assertEqual(self.ui.doubleSpinBox_defect2.value(), .5678)
                self.assertEqual(self.ui.doubleSpinBox_defect1.singleStep(), .01)
                with QtCore.QSignalBlocker(self.ui.checkBox_abs_def):
                    self.ui.checkBox_abs_def.setChecked(True)
                quality = self.c.calc_defect(table.corr_sig, .1234, .5678, self.c.state.interpretation.intervals)
                self.assertEqual(quality, [0]*40 + [1]*40 + [2]*39)

                # Comparison uses the same units as the editable cementogram.
                for field in ('checkBox_envelop', 'checkBox_mean', 'checkBox_func',
                              'checkBox_norm', 'checkBox_diff_result'):
                    with QtCore.QSignalBlocker(getattr(self.ui, field + suffix)):
                        getattr(self.ui, field + suffix).setChecked(False)
                with QtCore.QSignalBlocker(getattr(self.ui, 'checkBox_rel_ampl' + suffix)):
                    getattr(self.ui, 'checkBox_rel_ampl' + suffix).setChecked(True)
                self.ui.graphicsView.clear()
                getattr(self.c, 'plot_graph' + suffix)(1, 1, [2, 2], 0)
                np.testing.assert_allclose(self.ui.graphicsView.listDataItems()[0].yData, raw[:-1] * 350)

    def test_invalid_selection_or_interval_keeps_current_interpretation(self):
        before = self.c.state.interpretation
        for selected in ([], [self.ui.checkBox_signal1, self.ui.checkBox_signal_old_1]):
            for check in selected:
                with QtCore.QSignalBlocker(check):
                    check.setChecked(True)
            self.ui.pushButton_cement_func.click()
            self.assertIs(self.c.state.interpretation, before)
            self.assertIn('ровно один', self.ui.label_info.text())
        self.select_only(self.ui.checkBox_signal_old_1)
        self.ui.pushButton_cement_func.click()
        self.assertIs(self.c.state.interpretation, before)
        self.assertIn('рассчитайте', self.ui.label_info.text())
        self.c.state.old.signals = pd.DataFrame({
            'depth': [0., 1.], '1_rel_ampl': [1., 2.], '1_func': [1., 1.]})
        self.c.state.old.interval_min, self.c.state.old.interval_max = 0, 2
        self.ui.pushButton_cement_func.click()
        self.assertIs(self.c.state.interpretation, before)
        self.assertIn('нет данных', self.ui.label_info.text())

    def test_function_cement_edit_undo_and_word_export(self):
        with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory',
                          return_value=str(ROOT / 'пример' / 'новое оборудование')):
            self.c.open_dir()
        before = self.assert_function_cement('new', self.ui.checkBox_signal1)
        self.ui.editButton.click()
        self.assertTrue(self.c.curve_editor.active)
        self.ui.pushButton_plus_func.click()
        self.assertFalse(self.c.state.interpretation.table.corr_sig.equals(before.corr_sig))
        self.ui.undoButton.click()
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, before)
        path = Path(self.scratch.name) / 'по функции.docx'
        with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=(str(path), 'Word (*.docx)')):
            self.ui.pushButton_save_cement.click()
        self.assertTrue(path.exists())
        np.testing.assert_array_equal(self.c.state.interpretation.table.corr_sig, before.corr_sig)


if __name__ == '__main__':
    unittest.main()
