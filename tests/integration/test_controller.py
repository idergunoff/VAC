"""GUI boundary: isolated owners, cancellation and invalidation before any data."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from PyQt5 import QtWidgets
from vac.app import create_window
from vac.domain.models import ProcessingParameters, ProcessingResult, SignalIdentity


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_isolated_windows_cancellation_and_lifetime(self):
        w1, c1 = create_window()
        w2, c2 = create_window()
        try:
            c1.state.new.signals = pd.DataFrame({'depth': [1., 2.], 1: [3., 4.]})
            before = c1.state.new
            c1.state.interpretation.table = pd.DataFrame({'depth': [1.]})
            with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=''):
                c1.open_dir()
                c1.open_dir_old()
            with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=('', '')):
                c1.open_sig()
                c1.open_sig_old()
            self.assertIs(before, c1.state.new)
            self.assertIsNotNone(c1.state.interpretation.table)
            c1.state.new.result = ProcessingResult(ProcessingParameters(0, 1, 0, 0, 1, 1), 0)
            c1.ui.doubleSpinBox_int_min.setValue(0.1)
            self.assertTrue(c1.state.new.result.stale)
            self.assertIsNone(c1.state.interpretation.table)
            self.assertTrue(c2.state.new.signals.empty)
            c1.state.interpretation.table = pd.DataFrame({'depth': [1.]})
            c1.bind_interpretation('new', '1')
            c1.record_correction('fix_corr')
            self.assertEqual(c1.state.interpretation.corrections[0].signal, SignalIdentity('new', 0, '1'))
            c1.begin_load('new')
            self.assertEqual(c1.state.new.revision, 1)
            self.assertIsNone(c1.state.interpretation.table)
            self.assertEqual(c1.state.interpretation.corrections, [])
            c1.calc_int_cement()  # no selection
            c1.draw_cement()
            c1.corr_sig()
            c1.save_cement()
        finally:
            w1.close()
            w2.close()

    def test_failed_reads_preserve_the_open_research(self):
        window, controller = create_window()
        try:
            with tempfile.TemporaryDirectory(prefix='VAC ошибки чтения ') as tmp:
                invalid = Path(tmp) / 'ошибка'
                invalid.write_bytes(b'not a measurement')
                for equipment, suffix in [('new', ''), ('old', '_old')]:
                    measurements = getattr(controller.state, equipment)
                    measurements.signals = pd.DataFrame({'depth': [0., 1.], 1: [2., 3.]})
                    controller.state.interpretation.table = pd.DataFrame({'depth': [1.]})
                    interpretation = controller.state.interpretation
                    with patch.object(QtWidgets.QFileDialog, 'getExistingDirectory', return_value=tmp):
                        getattr(controller, 'open_dir' + suffix)()
                    self.assertIs(getattr(controller.state, equipment), measurements)
                    self.assertIs(controller.state.interpretation, interpretation)
                    self.assertEqual(controller.ui.label_info.styleSheet(), 'color:red')
                    with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(invalid), '')):
                        getattr(controller, 'open_sig' + suffix)()
                    self.assertIs(getattr(controller.state, equipment), measurements)
                    self.assertIs(controller.state.interpretation, interpretation)
        finally:
            window.close()

    def test_committing_a_correction_keeps_automatic_curve(self):
        window, controller = create_window()
        try:
            table = pd.DataFrame({'depth': np.arange(100, dtype=float),
                                  'first_sig': np.sin(np.arange(100) / 8),
                                  'corr_sig': np.sin(np.arange(100) / 8),
                                  'func': np.ones(100), 'corr_coeff': np.ones(100)})
            controller.state.interpretation.table = table.copy(deep=True)
            controller.state.interpretation.automatic_table = table.copy(deep=True)
            controller.bind_interpretation('new', '1')
            controller.ui.checkBox_shift_corr.setChecked(True)
            controller.ui.doubleSpinBox_x1_line.setValue(10)
            controller.ui.doubleSpinBox_x2_line.setValue(50)
            with patch.object(controller, 'draw_cement'):
                controller.corr_sig()
                controller.fix_corr()
            pd.testing.assert_frame_equal(controller.state.interpretation.automatic_table, table)
            self.assertFalse(controller.state.interpretation.table.first_sig.equals(table.first_sig))
            self.assertEqual([c.operation for c in controller.state.interpretation.corrections], ['corr_sig', 'fix_corr'])
        finally:
            window.close()


if __name__ == '__main__':
    unittest.main()
