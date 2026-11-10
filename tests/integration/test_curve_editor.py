"""Real Qt pointer/keyboard events exercise preview, commit, history and reset."""
import os
import gc
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
from vac.app import create_window
from vac.domain.models import SignalIdentity
from vac.services.snapshot import encode, decode


class CurveEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.qt_errors = []
        self.exception_hook = patch.object(sys, 'excepthook', lambda *error: self.qt_errors.append(error))
        self.exception_hook.start()
        self.addCleanup(self.exception_hook.stop)
        self.window, self.c = create_window()
        self.ui = self.c.ui
        self.editor = self.c.curve_editor
        self.window.resize(1366, 768)
        self.window.show()
        self.window.activateWindow()
        self.source = pd.DataFrame({'depth': [0., 1., 2., 3., 4.],
            'corr_sig': [1., 2., 3., 2., 1.], 'first_sig': [1., 2., 3., 2., 1.],
            'func': .1, 'corr_coeff': 1.})
        self.c.state.interpretation.table = self.source.copy(deep=True)
        self.c.state.interpretation.automatic_table = self.source.copy(deep=True)
        self.c.state.interpretation.signal = SignalIdentity('new', 0, 'test')
        self.c.draw_cement()
        self.ui.editButton.click()
        self.editor.set_bounds(0., 4.)
        self.editor.vb.setRange(xRange=(-.2, 4.2), yRange=(0., 6.), padding=0)
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.assertEqual(self.qt_errors, [], 'Unhandled Qt callback error')

    def pixel(self, x, y):
        return self.ui.graphicsView.mapFromScene(self.editor.vb.mapViewToScene(QtCore.QPointF(x, y)))

    def mouse(self, kind, x, y):
        position = self.pixel(x, y)
        button = QtCore.Qt.NoButton if kind == QtCore.QEvent.MouseMove else QtCore.Qt.LeftButton
        buttons = QtCore.Qt.NoButton if kind == QtCore.QEvent.MouseButtonRelease else QtCore.Qt.LeftButton
        event = QtGui.QMouseEvent(kind, QtCore.QPointF(position), button, buttons, QtCore.Qt.NoModifier)
        QtWidgets.QApplication.sendEvent(self.ui.graphicsView.viewport(), event)
        self.app.processEvents()

    def drag(self, x, old, new):
        self.mouse(QtCore.QEvent.MouseButtonPress, x, old)
        self.mouse(QtCore.QEvent.MouseMove, x, new)
        self.mouse(QtCore.QEvent.MouseButtonRelease, x, new)

    def test_drag_previews_then_commits_one_history_step(self):
        self.mouse(QtCore.QEvent.MouseButtonPress, 2, 3)
        for y in [3.2, 3.5, 4.]:
            self.mouse(QtCore.QEvent.MouseMove, 2, y)
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.source)
        self.assertEqual(len(self.c.workspace_actions.undo_stack), 0)
        self.assertGreater(self.editor.curve.yData[2], 3.9)
        self.mouse(QtCore.QEvent.MouseButtonRelease, 2, 4)
        after = self.c.state.interpretation.table.copy(deep=True)
        self.assertAlmostEqual(after.corr_sig.iloc[2], 4., delta=.03)
        self.assertEqual(after.corr_sig.iloc[0], 1.)
        self.assertEqual(after.corr_sig.iloc[4], 1.)
        self.assertEqual(len(self.c.workspace_actions.undo_stack), 1)
        pd.testing.assert_frame_equal(self.c.state.interpretation.automatic_table, self.source)
        self.ui.undoButton.click()
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.source)
        self.ui.redoButton.click()
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, after)

    def test_drag_each_endpoint_previews_commits_and_undoes(self):
        for index, bounds, pinned in [(0, (0., 3.), [3, 4]), (4, (1., 4.), [0, 1])]:
            with self.subTest(index=index):
                self.editor.set_bounds(*bounds)
                self.mouse(QtCore.QEvent.MouseButtonPress, index, 1)
                self.assertEqual(self.editor.anchor, index)
                self.assertEqual(self.editor.gesture['kind'], 'curve')
                self.mouse(QtCore.QEvent.MouseMove, index, 2)
                pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.source)
                self.assertAlmostEqual(self.editor.curve.yData[index], 2., delta=.03)
                self.mouse(QtCore.QEvent.MouseButtonRelease, index, 2)
                after = self.c.state.interpretation.table.copy(deep=True)
                self.assertAlmostEqual(after.corr_sig.iloc[index], 2., delta=.03)
                np.testing.assert_array_equal(after.corr_sig.iloc[pinned], self.source.corr_sig.iloc[pinned])
                self.assertEqual(len(self.c.workspace_actions.undo_stack), 1)
                pd.testing.assert_frame_equal(self.c.state.interpretation.automatic_table, self.source)
                self.ui.undoButton.click()
                pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.source)
                self.ui.redoButton.click()
                pd.testing.assert_frame_equal(self.c.state.interpretation.table, after)
                self.ui.undoButton.click()

    def test_endpoint_buttons_and_escape(self):
        self.ui.curveStep.setValue(.25)
        for index in (0, 4):
            with self.subTest(index=index):
                self.mouse(QtCore.QEvent.MouseButtonPress, index, 1)
                self.mouse(QtCore.QEvent.MouseButtonRelease, index, 1)
                self.assertEqual(self.editor.anchor, index)
                self.assertEqual(len(self.c.workspace_actions.undo_stack), 0)
                self.ui.raiseCurveButton.click()
                self.assertEqual(self.c.state.interpretation.table.corr_sig.iloc[index], 1.25)
                self.ui.lowerCurveButton.click()
                np.testing.assert_array_equal(self.c.state.interpretation.table.corr_sig, self.source.corr_sig)
                self.ui.undoButton.click()
                self.ui.undoButton.click()
                self.mouse(QtCore.QEvent.MouseButtonPress, index, 1)
                self.mouse(QtCore.QEvent.MouseMove, index, 2)
                QtTest.QTest.keyClick(self.window, QtCore.Qt.Key_Escape)
                np.testing.assert_array_equal(self.editor.curve.yData, self.source.corr_sig)
                self.assertIsNone(self.editor.gesture)
                self.assertTrue(self.ui.redoButton.isEnabled())

    def test_internal_bounds_are_pinned_and_retreating_from_edge_excludes_it(self):
        self.editor.set_bounds(1., 3.)
        np.testing.assert_array_equal(self.editor.candidates(), [2])
        self.editor.set_bounds(0., 3.)
        np.testing.assert_array_equal(self.editor.candidates(), [0, 1, 2])
        self.editor.anchor = 0
        self.editor.set_bounds(.01, 3.)
        self.assertNotIn(0, self.editor.candidates())
        self.assertNotEqual(self.editor.anchor, 0)
        self.editor.set_bounds(0., 0.)
        self.assertIsNone(self.editor.anchor)
        self.assertFalse(self.ui.raiseCurveButton.isEnabled())

    def test_mouse_selection_snaps_to_data_endpoints(self):
        self.ui.selectInfluenceButton.click()
        self.mouse(QtCore.QEvent.MouseButtonPress, 4., 5.)
        self.mouse(QtCore.QEvent.MouseMove, 0., 5.)
        self.mouse(QtCore.QEvent.MouseButtonRelease, 0., 5.)
        self.assertEqual(self.editor.bounds, (0., 4.))
        np.testing.assert_array_equal(self.editor.candidates(), np.arange(5))

    def test_redraw_preserves_endpoints_beyond_spinbox_precision(self):
        table = self.source.copy(deep=True)
        table['depth'] += .000033
        self.c.state.interpretation.table = table
        self.c.draw_cement()
        self.editor.set_bounds(float(table.depth.min()), float(table.depth.max()))
        for index in (0, 4):
            self.editor.anchor = index
            self.c.draw_cement()
            self.assertEqual(self.editor.bounds, (float(table.depth.min()), float(table.depth.max())))
            self.assertEqual(self.editor.anchor, index)
            self.ui.raiseCurveButton.click()
            self.assertAlmostEqual(self.c.state.interpretation.table.corr_sig.iloc[index], 1.1)
            self.ui.undoButton.click()

    def test_click_selects_any_interior_sample_and_buttons_accumulate(self):
        self.mouse(QtCore.QEvent.MouseButtonPress, 1, 2)
        self.mouse(QtCore.QEvent.MouseButtonRelease, 1, 2)
        self.assertEqual(self.editor.anchor, 1)
        self.assertEqual(len(self.c.workspace_actions.undo_stack), 0)
        self.ui.curveStep.setValue(.25)
        self.ui.raiseCurveButton.click()
        self.ui.raiseCurveButton.click()
        self.ui.lowerCurveButton.click()
        self.assertEqual(self.c.state.interpretation.table.corr_sig.iloc[1], 2.25)
        self.assertEqual(len(self.c.workspace_actions.undo_stack), 3)
        self.ui.undoButton.click()
        self.assertEqual(self.c.state.interpretation.table.corr_sig.iloc[1], 2.5)
        self.ui.lowerCurveButton.click()
        self.assertFalse(self.ui.redoButton.isEnabled())

    def test_escape_cancels_preview_and_preserves_redo(self):
        self.ui.raiseCurveButton.click()
        self.ui.undoButton.click()
        self.mouse(QtCore.QEvent.MouseButtonPress, 2, 3)
        self.mouse(QtCore.QEvent.MouseMove, 2, 4)
        QtTest.QTest.keyClick(self.window, QtCore.Qt.Key_Escape)
        self.app.processEvents()
        self.assertIsNone(self.editor.gesture)
        np.testing.assert_array_equal(self.editor.curve.yData, self.source.corr_sig)
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.source)
        self.assertTrue(self.ui.redoButton.isEnabled())
        self.assertEqual(len(self.c.workspace_actions.undo_stack), 0)

    def test_region_selection_in_reverse_and_boundary_drag(self):
        self.ui.selectInfluenceButton.click()
        self.mouse(QtCore.QEvent.MouseButtonPress, 3.8, 5)
        self.mouse(QtCore.QEvent.MouseMove, .2, 5)
        self.mouse(QtCore.QEvent.MouseButtonRelease, .2, 5)
        np.testing.assert_allclose(self.editor.bounds, [.2, 3.8], atol=.02)
        self.assertFalse(self.ui.selectInfluenceButton.isChecked())
        self.assertEqual(self.editor.anchor, 2)
        self.mouse(QtCore.QEvent.MouseButtonPress, .2, 5)
        self.mouse(QtCore.QEvent.MouseMove, .8, 5)
        self.mouse(QtCore.QEvent.MouseButtonRelease, .8, 5)
        self.assertAlmostEqual(self.editor.bounds[0], .8, delta=.02)
        self.assertAlmostEqual(self.ui.doubleSpinBox_x1_line.value(), .8, delta=.02)
        self.assertEqual(len(self.c.workspace_actions.undo_stack), 0)
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.source)

    def test_numeric_bounds_and_empty_region(self):
        self.ui.doubleSpinBox_x1_line.setValue(1.5)
        self.assertAlmostEqual(self.editor.bounds[0], 1.5)
        self.ui.doubleSpinBox_x2_line.setValue(2.5)
        self.assertEqual(self.editor.anchor, 2)
        self.editor.set_bounds(2, 2)
        self.assertIsNone(self.editor.anchor)
        self.assertFalse(self.ui.raiseCurveButton.isEnabled())

    def test_cancel_boundary_move_and_no_motion_preserve_history(self):
        self.mouse(QtCore.QEvent.MouseButtonPress, 2, 5)
        self.mouse(QtCore.QEvent.MouseButtonRelease, 2, 5)
        self.assertEqual(self.editor.bounds, (0., 4.))
        self.mouse(QtCore.QEvent.MouseButtonPress, 0, 5)
        self.mouse(QtCore.QEvent.MouseMove, .8, 5)
        QtTest.QTest.keyClick(self.window, QtCore.Qt.Key_Escape)
        self.assertEqual(self.editor.bounds, (0., 4.))
        self.mouse(QtCore.QEvent.MouseButtonPress, 2, 3)
        self.mouse(QtCore.QEvent.MouseMove, 2, 4)
        self.mouse(QtCore.QEvent.MouseMove, 2, 3)
        self.mouse(QtCore.QEvent.MouseButtonRelease, 2, 3)
        self.assertEqual(len(self.c.workspace_actions.undo_stack), 0)
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.source)

    def test_editing_controls_fit_without_horizontal_scroll(self):
        for width, height in [(800, 600), (1366, 768), (1920, 1080)]:
            self.window.resize(width, height)
            for _ in range(4):
                self.app.processEvents()
            scroll = self.ui.workspace.pages.currentWidget()
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
            if height >= 768:
                self.assertTrue(scroll.viewport().rect().contains(
                    self.ui.raiseCurveButton.mapTo(scroll.viewport(), self.ui.raiseCurveButton.rect().bottomRight())))

    def test_closed_windows_release_event_filters_before_collection(self):
        for _ in range(5):
            window, controller = create_window()
            window.show()
            self.app.processEvents()
            window.close()
            del controller, window
            gc.collect()
            self.app.processEvents()

    def test_losing_window_focus_cancels_preview(self):
        self.mouse(QtCore.QEvent.MouseButtonPress, 2, 3)
        self.mouse(QtCore.QEvent.MouseMove, 2, 4)
        QtWidgets.QApplication.sendEvent(self.window, QtCore.QEvent(QtCore.QEvent.WindowDeactivate))
        self.assertIsNone(self.editor.gesture)
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.source)

    def test_switch_off_editing_or_to_signals_cancels_gesture(self):
        for action in [self.ui.editButton.click, self.c.choice_signal]:
            with self.subTest(action=action):
                if not self.ui.editButton.isChecked():
                    self.ui.editButton.click()
                self.c.draw_cement()
                self.mouse(QtCore.QEvent.MouseButtonPress, 2, 3)
                self.mouse(QtCore.QEvent.MouseMove, 2, 4)
                action()
                self.assertIsNone(self.editor.gesture)
                self.assertFalse(self.editor.active)
                self.assertFalse(self.ui.raiseCurveButton.isEnabled())
                pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.source)

    def test_invalidation_during_drag_clears_editor_and_history(self):
        self.ui.raiseCurveButton.click()
        self.mouse(QtCore.QEvent.MouseButtonPress, 2, 3.1)
        self.mouse(QtCore.QEvent.MouseMove, 2, 4)
        self.c.clear_interpretation()
        self.assertIsNone(self.c.state.interpretation.table)
        self.assertIsNone(self.editor.gesture)
        self.assertFalse(self.editor.active)
        self.assertFalse(self.ui.undoButton.isEnabled())

    def test_shortcuts_and_snapshot_preserve_metadata_and_original(self):
        self.ui.raiseCurveButton.click()
        after = self.c.state.interpretation.table.copy(deep=True)
        QtTest.QTest.keyClick(self.window, QtCore.Qt.Key_Z, QtCore.Qt.ControlModifier)
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, self.source)
        QtTest.QTest.keyClick(self.window, QtCore.Qt.Key_Y, QtCore.Qt.ControlModifier)
        pd.testing.assert_frame_equal(self.c.state.interpretation.table, after)
        restored = decode(*encode(self.c.state))
        pd.testing.assert_frame_equal(restored.interpretation.table, after)
        pd.testing.assert_frame_equal(restored.interpretation.automatic_table, self.source)
        correction = restored.interpretation.corrections[-1]
        self.assertEqual(correction.operation, 'local_shift')
        self.assertEqual(dict(correction.parameters), {'x1': 0., 'x2': 4., 'anchor': 2., 'delta': .1})

    def test_legacy_operation_keeps_local_edits_and_history_is_bounded(self):
        self.ui.raiseCurveButton.click()
        self.c.plus_func()
        self.assertAlmostEqual(self.c.state.interpretation.table.corr_sig.iloc[2], 3.2)
        self.ui.undoButton.click()
        self.assertAlmostEqual(self.c.state.interpretation.table.corr_sig.iloc[2], 3.1)
        for _ in range(35):
            self.ui.raiseCurveButton.click()
        self.assertEqual(len(self.c.workspace_actions.undo_stack), 30)
        for _ in range(30):
            self.ui.undoButton.click()
        self.assertFalse(self.ui.undoButton.isEnabled())


if __name__ == '__main__':
    unittest.main()
