"""Comparison controls render in the existing plot without changing measurements."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from vac.app import create_window
from vac import reference


class ReferenceComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window, self.c = create_window()
        self.ui = self.c.ui

    def tearDown(self):
        self.window.close()

    def test_switch_equipment_references_and_frequencies_in_same_plot(self):
        plot = self.ui.graphicsView
        for equipment, count in [('new', 8), ('old', 12)]:
            device = getattr(self.c.state, equipment)
            data = {'depth': [10., 11., 12.]}
            data.update({str(n) + '_envelope' if equipment == 'new' else n:
                         [float(n)] * 3 for n in range(1, count + 1)})
            device.signals = pd.DataFrame(data)
            device.interval_min, device.interval_max = 0, 2
            before = device.signals.copy(deep=True)
            with patch('matplotlib.figure.Figure.show', side_effect=AssertionError('Separate window')):
                getattr(self.c, 'draw_graphics' if equipment == 'new' else 'draw_graphics_old')()
            self.assertIs(plot, self.ui.graphicsView)
            curves = plot.listDataItems()
            self.assertEqual(len(curves), 4)
            np.testing.assert_array_equal(curves[2].yData, getattr(reference, 'good_' + equipment))
            np.testing.assert_array_equal(curves[3].yData, getattr(reference, 'bad_' + equipment))
            self.assertEqual(len(curves[0].xData), 2)
            self.assertAlmostEqual(curves[0].yData[0], 3.5 if equipment == 'new' else 5.5)
            self.ui.referenceSelection.setCurrentIndex(2)
            self.ui.referenceFrequency.setCurrentIndex(1)
            self.assertEqual(len(plot.listDataItems()), 2)
            np.testing.assert_array_equal(plot.listDataItems()[1].yData, getattr(reference, 'bad_' + equipment))
            pd.testing.assert_frame_equal(device.signals, before)
            self.ui.referenceSelection.setCurrentIndex(0)
            self.ui.referenceFrequency.setCurrentIndex(0)
        self.ui.plotMode.setCurrentIndex(0)
        self.assertFalse(self.c.reference_comparison.active)
        self.assertIsNone(plot.getPlotItem().legend)

    def test_empty_equipment_and_return_to_cement(self):
        self.c.draw_graphics()
        self.assertEqual(len(self.ui.graphicsView.listDataItems()), 2)
        self.assertIn('Загрузите', self.ui.label_info.text())
        self.c.state.interpretation.table = pd.DataFrame({
            'depth': [0., 1.], 'corr_sig': [1., 2.], 'first_sig': [1., 2.]})
        self.ui.editButton.setChecked(True)
        self.c.draw_cement()
        self.assertFalse(self.c.reference_comparison.active)
        self.assertIsNotNone(self.c.curve_editor.curve)
        self.assertIsNone(self.ui.graphicsView.getPlotItem().legend)


if __name__ == '__main__':
    unittest.main()
