"""Draft report layout, finite-data centering, cancellation and applied settings."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from vac.app import create_window
from vac.gui.report_alignment import ReportAlignmentDialog, centered_limits
from vac.io.legacy_options import load_trusted
from vac.reporting import figures


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window, self.c = create_window()
        self.c.state.interpretation.table = pd.DataFrame({
            'depth': [0., 50., 150., 250., 350.], 'corr_sig': [-2., 1., 4., 2., 1.]})
        self.before = self.c.state.interpretation.table.copy(deep=True)
        first, second = self.c.ui.doubleSpinBox_defect1, self.c.ui.doubleSpinBox_defect2
        with QtCore.QSignalBlocker(first), QtCore.QSignalBlocker(second):
            first.setValue(1.)
            second.setValue(3.)
        self.c.ui.doubleSpinBox_min_cement.setValue(0.)
        self.c.ui.doubleSpinBox_max_cement.setValue(5.)

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_auto_scale_is_symmetric_and_includes_edited_curve_with_padding(self):
        low, high = centered_limits([-2., 4., np.nan, np.inf], 1., 3.)
        self.assertAlmostEqual((low + high) / 2, 2., places=4)
        self.assertLess(low, -2.)
        self.assertGreater(high, 4.)
        self.assertGreater(centered_limits([2., 2.], 1., 3.)[1], 3.)
        for curve, first, second in [([np.nan], 1., 3.), ([1.], 3., 1.), ([1.], np.inf, 3.)]:
            with self.assertRaises(ValueError):
                centered_limits(curve, first, second)

    def test_auto_preview_cancel_and_manual_apply(self):
        dialog = ReportAlignmentDialog(self.c)
        dialog.show()
        for _ in range(10):
            self.app.processEvents()
        self.assertLessEqual(dialog.width(), self.app.primaryScreen().availableGeometry().width())
        for plot, _ in dialog.plots:
            self.assertLessEqual(plot.sceneBoundingRect().right(), dialog.preview.viewport().width())
        dialog.auto_button.click()
        values = dialog.values()
        self.assertAlmostEqual((values['min_cement'] + values['max_cement']) / 2, 2., places=4)
        self.assertEqual(self.c.ui.doubleSpinBox_min_cement.value(), 0.)
        self.assertEqual(len(dialog.plots), 4)
        for index, (plot, lines) in enumerate(dialog.plots):
            np.testing.assert_allclose(plot.viewRange()[0], [values['min_cement'], values['max_cement']])
            np.testing.assert_allclose(plot.viewRange()[1], [index * 100, (index + 1) * 100])
            self.assertTrue(plot.getViewBox().state['yInverted'])
            self.assertEqual([line.value() for line in lines], [1., 3.])
        dialog.reject()
        self.assertEqual(self.c.ui.doubleSpinBox_min_cement.value(), 0.)
        dialog = ReportAlignmentDialog(self.c)
        dialog.fields['min_cement'].setValue(8.)
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.fields['defect1'].setValue(2.)
        dialog.fields['defect2'].setValue(4.)
        dialog.auto_button.click()
        expected = dialog.values()
        dialog.apply_button.click()
        self.assertEqual(dialog.result(), QtWidgets.QDialog.Accepted)
        for name, value in expected.items():
            self.assertEqual(getattr(self.c.ui, 'doubleSpinBox_' + name).value(), value)
            self.assertEqual(self.c.state.interpretation.options['doubleSpinBox_' + name], value)
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.before)

    def test_apply_and_save_uses_alignment_before_export(self):
        def edit_and_accept(dialog):
            dialog.auto_align()
            dialog.save()
            return dialog.result()
        with patch.object(ReportAlignmentDialog, 'exec_', edit_and_accept), \
             patch.object(self.c, 'export_report') as export:
            self.c.align_report()
            export.assert_called_once_with()
        self.assertLess(self.c.ui.doubleSpinBox_min_cement.value(), -2.)
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.before)

    def test_applied_scale_reaches_report_image_and_settings_file(self):
        dialog = ReportAlignmentDialog(self.c)
        dialog.auto_align()
        expected = dialog.values()
        dialog.apply()
        self.c.ui.lineEdit_number_skv.setText('17')
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='VAC выравнивание ') as tmp:
            try:
                os.chdir(tmp)
                target = Path(tmp, 'отчёт.docx')
                with patch('vac.controller.choose_report_path', return_value=target), \
                     patch('vac.controller.figures.cementogramma', wraps=figures.cementogramma) as render:
                    self.c.export_report()
                self.assertTrue(target.is_file())
                options = load_trusted(Path(tmp, 'options_17.vac'))
                for name, value in expected.items():
                    self.assertEqual(options['doubleSpinBox_' + name], value)
                self.assertEqual(options['signal'], self.before['corr_sig'].tolist())
                self.assertEqual(render.call_args.args[1:5], (
                    expected['min_cement'], expected['max_cement'], expected['defect1'], expected['defect2']))
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
