"""User workflows across compact controls, edits, undo and direct Word export."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets
from vac.app import create_window
from vac.domain.models import SignalIdentity


class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window, self.c = create_window()
        self.ui = self.c.ui
        self.window.resize(1366, 768)
        self.window.show()
        table = pd.DataFrame({'depth': [0., 1., 2., 3., 4.],
                              'first_sig': [1., 2., 3., 2., 1.],
                              'corr_sig': [1., 2., 3., 2., 1.],
                              'func': [.1, .2, .3, .2, .1], 'corr_coeff': 1.})
        self.c.state.interpretation.table = table
        self.c.state.interpretation.signal = SignalIdentity('new', 0, '1 Тест')
        self.c.state.interpretation.automatic_table = table.copy(deep=True)
        self.c.draw_cement()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        plt.close('all')
        self.app.processEvents()

    def test_edit_undo_redo_and_invalidation(self):
        before = self.c.state.interpretation.table.copy(deep=True)
        self.ui.editButton.click()
        self.app.processEvents()
        self.assertTrue(self.ui.editTools.isVisible())
        self.assertEqual(self.ui.workspace.pages.itemText(self.ui.workspace.pages.currentIndex()), 'Цементограмма')
        self.ui.pushButton_plus_func.click()
        after = self.c.state.interpretation.table.copy(deep=True)
        self.assertFalse(before.equals(after))
        self.assertEqual(len(self.c.state.interpretation.corrections), 1)
        self.ui.undoButton.click()
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, before)
        self.assertEqual(self.c.state.interpretation.corrections, [])
        self.ui.redoButton.click()
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, after)
        self.assertEqual(len(self.c.state.interpretation.corrections), 1)
        self.c.clear_interpretation()
        self.assertFalse(self.ui.redoButton.isEnabled())
        self.assertFalse(self.ui.undoButton.isEnabled())
        self.assertFalse(self.ui.editButton.isEnabled())
        self.assertFalse(self.ui.pushButton_save_cement.isEnabled())

    def test_interval_undo_and_new_edit_discards_redo(self):
        self.ui.doubleSpinBox_undefine_max.setValue(3)
        self.ui.doubleSpinBox_undefine_min.setValue(1)
        label = self.ui.label_int_undefine.text()
        self.ui.pushButton_add_undefine.setEnabled(True)
        self.ui.pushButton_add_undefine.click()
        self.assertTrue(self.c.state.interpretation.intervals.good)
        self.ui.undoButton.click()
        self.assertFalse(self.c.state.interpretation.intervals.good)
        self.assertEqual(self.ui.label_int_undefine.text(), label)
        self.ui.pushButton_plus_func.click()
        self.assertFalse(self.ui.redoButton.isEnabled())

    def test_multiple_selected_signals_do_not_replace_interpretation(self):
        before = self.c.state.interpretation
        with patch.object(self.c, 'choice_signal'):
            blockers = [QtCore.QSignalBlocker(self.ui.checkBox_signal1),
                        QtCore.QSignalBlocker(self.ui.checkBox_signal2)]
            self.ui.checkBox_signal1.setChecked(True)
            self.ui.checkBox_signal2.setChecked(True)
            del blockers
        self.ui.pushButton_int_cement.click()
        self.assertIs(self.c.state.interpretation, before)
        self.assertIn('ровно один', self.ui.label_info.text())
        self.assertEqual(self.ui.workspace.pages.itemText(self.ui.workspace.pages.currentIndex()), 'Данные')

    def test_sum_buttons_average_selected_signals_for_both_devices(self):
        root = Path(__file__).resolve().parents[2]
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory(prefix='VAC sums ') as tmp:
                os.chdir(tmp)
                for tab, equipment, suffix, folder, first_sum in [
                    (0, 'new', '', 'новое оборудование', 9),
                    (1, 'old', '_old', 'старое оборудование', 13),
                ]:
                    with self.subTest(equipment=equipment):
                        with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory',
                                          return_value=str(root / 'пример' / folder)):
                            getattr(self.c, 'open_dir' + suffix)()
                        self.ui.workspace.select(0)
                        self.ui.equipmentTabs.setCurrentIndex(tab)
                        device = getattr(self.ui, equipment + '_device')
                        checks = device.findChildren(QtWidgets.QCheckBox)
                        for check in checks:
                            check.setChecked(False)
                        selected = [check for check in checks
                                    if check.isEnabled() and int(check.text().split()[0]) < first_sum][:2]
                        self.assertEqual(len(selected), 2)
                        for check in selected:
                            check.setChecked(True)
                        channels = [int(check.text().split()[0]) for check in selected]
                        signals = getattr(self.c.state, equipment).signals
                        columns = [str(n) + '_envelope' if equipment == 'new' else n for n in channels]
                        expected = (signals[columns[0]] + signals[columns[1]]) / 2
                        for slot in range(1, 4):
                            button = getattr(self.ui, 'pushButton_sum' + str(slot) + suffix)
                            for _ in range(3):
                                self.app.processEvents()
                            self.assertTrue(button.isVisible())
                            self.assertTrue(button.isEnabled())
                            button.click()
                            channel = first_sum + slot - 1
                            column = str(channel) + '_envelope' if equipment == 'new' else channel
                            pd.testing.assert_series_equal(signals[column], expected, check_names=False)
        finally:
            os.chdir(previous)

    def test_export_button_builds_current_report_without_preview(self):
        self.ui.lineEdit_number_skv.setText('Тест')
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='VAC компактный отчёт ') as tmp:
            try:
                os.chdir(tmp)
                with patch.object(plt.Figure, 'show', side_effect=AssertionError('Preview opened')):
                    self.ui.pushButton_save_cement.click()
                self.assertTrue(Path('цементограмма_по_скв_Тест.docx').exists())
                self.assertTrue(Path('temp.png').exists())
                self.assertEqual(plt.get_fignums(), [])
                self.assertIsNotNone(self.c.state.interpretation.options)
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
