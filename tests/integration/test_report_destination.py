"""Destination selection survives restart and cancellation has no side effects."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PyQt5 import QtCore, QtWidgets
from vac.gui.report_destination import choose_report_path, remember_report_path
from vac.app import create_window


class DestinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_directory_survives_new_settings_instance_and_cancel(self):
        with tempfile.TemporaryDirectory(prefix='VAC папка ') as tmp:
            settings_file = str(Path(tmp) / 'settings.ini')
            factory = lambda: QtCore.QSettings(settings_file, QtCore.QSettings.IniFormat)
            with patch('vac.gui.report_destination.report_settings', side_effect=factory):
                target = Path(tmp) / 'отчёт.docx'
                self.assertTrue(remember_report_path(target))
                with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=('', '')) as dialog:
                    self.assertIsNone(choose_report_path(None, '17'))
                    self.assertEqual(Path(dialog.call_args.args[2]).parent, Path(tmp))
                self.assertEqual(factory().value('directory'), tmp)
                with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=(str(target.with_suffix('')), '')):
                    self.assertEqual(choose_report_path(None, '17'), target)
                Path(tmp, 'removed').mkdir()
                remember_report_path(Path(tmp, 'removed', 'report.docx'))
                Path(tmp, 'removed').rmdir()
                with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=('', '')) as dialog:
                    choose_report_path(None, '17')
                    self.assertNotEqual(Path(dialog.call_args.args[2]).parent, Path(tmp, 'removed'))

    def test_cancel_export_does_not_render_or_write(self):
        window, controller = create_window()
        try:
            controller.state.interpretation.table = object()
            with patch('vac.controller.choose_report_path', return_value=None), \
                 patch.object(controller, 'cementogramma') as render, \
                 patch.object(controller, 'save_cement') as save:
                controller.export_report()
                render.assert_not_called()
                save.assert_not_called()
        finally:
            window.close()
