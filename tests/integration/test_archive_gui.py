"""Real samples through archive, GUI processing, edits, restart and Word export."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sqlite3
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/unit')]
from test_calculation import read_table
from vac.app import create_window
from vac.services import archive as service
from vac.gui.archive import capture_widgets, blocked
from vac.storage import Archive
from vac.storage.backup import backup, restore


@contextmanager
def sample_order(equipment):
    root = ROOT / 'пример' / ('новое оборудование' if equipment == 'new' else 'старое оборудование')
    order = json.loads((ROOT / 'tests/fixtures/current' / equipment / 'manifest.json').read_text())['read_order'][:9 if equipment == 'new' else 13]
    if equipment == 'new':
        folders = {}
        for name in order:
            relative = Path(name)
            folders.setdefault(relative.parent, []).append(relative.name)
        with patch('vac.io.lvm.os.walk', return_value=[(str(root / p), [], n) for p, n in folders.items()]):
            yield root
    else:
        with patch('vac.io.twf.os.listdir', return_value=order):
            yield root


class ArchiveGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='VAC архив кириллица ')
        self.path = Path(self.tmp.name) / 'исследования.sqlite'
        self.previous = Path.cwd()
        os.chdir(self.tmp.name)
        self.window, self.c = create_window()
        self.c.archive_menu.path = str(self.path)
        with Archive(self.path) as db:
            self.well = db.create_well('Скважина № 7')

    def tearDown(self):
        self.window.close()
        plt.close('all')
        os.chdir(self.previous)
        self.tmp.cleanup()

    def import_sample(self, equipment):
        with Archive(self.path) as db, sample_order(equipment) as root:
            return service.import_directory(db, self.well, root, equipment, mean=40)

    def test_both_full_cycles_and_new_versions(self):
        for equipment in ('new', 'old'):
            with self.subTest(equipment=equipment), patch.object(plt.Figure, 'show'):
                report = self.import_sample(equipment)
                self.assertFalse(report.errors)
                self.assertEqual(len(report.accepted), 9 if equipment == 'new' else 13)
                self.assertEqual(sum(a['role'] == 'noise' for a in report.assignments), 1)
                duplicate = self.import_sample(equipment)
                self.assertNotEqual(duplicate.study_id, report.study_id)
                self.assertEqual(len(duplicate.duplicates), len(report.accepted))
                menu = self.c.archive_menu
                menu.open_study(report.study_id)
                with patch.object(self.c, 'calc_to_int', side_effect=AssertionError('recalculation')), patch.object(self.c, 'calc_to_int_old', side_effect=AssertionError('recalculation')):
                    menu.open_study(report.study_id)
                self.assertIsNone(getattr(self.c.state, equipment).result)
                menu.recalculate()
                baseline = ROOT / 'tests/fixtures/current' / equipment / 'analysis'
                suffix = '' if equipment == 'new' else '_old'
                expected = read_table(baseline, 'all_signals' + suffix)
                actual = getattr(self.c.state, equipment)
                for col in expected:
                    np.testing.assert_allclose(actual.signals[col], expected[col], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=str(col))
                checkbox = self.c.ui.checkBox_signal1 if equipment == 'new' else self.c.ui.checkBox_signal_old_1
                checkbox.setChecked(True)
                self.c.calc_int_cement()
                self.c.ui.checkBox_shift_corr.setChecked(True)
                self.c.ui.doubleSpinBox_x1_line.setValue(20)
                self.c.ui.doubleSpinBox_x2_line.setValue(30)
                self.c.corr_sig(); self.c.fix_corr()
                self.c.ui.radioButton_NC.setChecked(True)
                self.c.ui.doubleSpinBox_undefine_min.setValue(40)
                self.c.ui.doubleSpinBox_undefine_max.setValue(45)
                self.c.add_undefine()
                self.c.cementogramma()
                self.c.ui.lineEdit_number_skv.setText('7')
                run_id = menu.save_current()
                subprocess.run([sys.executable, '-c',
                    'import sys; from vac.storage import Archive; from vac.services.archive import open_version; '
                    'db=Archive(sys.argv[1]); state, widgets, identity=open_version(db,sys.argv[2]); '
                    'assert state.interpretation.table is not None; assert identity; '
                    'assert "PyQt5" not in sys.modules; db.close()', str(self.path), run_id],
                    env=dict(os.environ, PYTHONPATH=str(ROOT)), check=True, timeout=30)
                recovery = Path(self.tmp.name) / ('backup-' + equipment + '.sqlite')
                backup(self.path, recovery)
                self.path = Path(self.tmp.name) / ('clean-' + equipment) / 'archive.sqlite'
                restore(recovery, self.path)
                before = copy.deepcopy(self.c.state)
                saved_widgets = capture_widgets(self.c.ui)
                self.window.close()
                self.window, self.c = create_window()
                menu = self.c.archive_menu
                menu.path = str(self.path)
                with patch.object(self.c, 'calc', side_effect=AssertionError('recalculation')), patch.object(self.c, 'calc_to_int', side_effect=AssertionError('recalculation')), patch.object(self.c, 'calc_to_int_old', side_effect=AssertionError('recalculation')):
                    menu.open_version(run_id)
                restored = getattr(self.c.state, equipment)
                pd.testing.assert_frame_equal(restored.signals, actual.signals, check_exact=True)
                pd.testing.assert_frame_equal(restored.statistics, actual.statistics, check_exact=True)
                pd.testing.assert_frame_equal(self.c.state.interpretation.table, before.interpretation.table, check_exact=True)
                pd.testing.assert_frame_equal(self.c.state.interpretation.automatic_table, before.interpretation.automatic_table, check_exact=True)
                self.assertEqual(self.c.state.interpretation.corrections, before.interpretation.corrections)
                self.assertEqual(self.c.state.interpretation.intervals, before.interpretation.intervals)
                self.assertEqual(capture_widgets(self.c.ui)['lineEdit_number_skv'], saved_widgets['lineEdit_number_skv'])
                target = Path(self.tmp.name) / ('отчёт ' + equipment + '.docx')
                with patch.object(service, 'code_hash', return_value='different-report-program'):
                    menu.save_report(target)
                with Archive(self.path) as db:
                    _, _, interpretation_id = db.open_gui_version(menu.run_id)
                    report_id = db.list_reports(interpretation_id)[0]
                    self.assertEqual(db.open_report(report_id)['payload'], target.read_bytes())
                    self.assertEqual(db.open_run(menu.run_id)['algorithm_version'], db.open_run(run_id)['algorithm_version'])
                    self.assertEqual(db.open_run(menu.run_id)['program_version'], 'different-report-program')
                    self.assertTrue(db.open_interpretation(interpretation_id)['intervals'])
                    old = db.open_gui_version(run_id)
                    self.assertIsNotNone(old[2])
                    self.assertEqual(len(db.list_runs(report.study_id)), 2)
                report_copy = Path(self.tmp.name) / ('report-backup-' + equipment + '.sqlite')
                backup(self.path, report_copy)
                recovered_report = Path(self.tmp.name) / ('report-restored-' + equipment + '.sqlite')
                restore(report_copy, recovered_report)
                with Archive(recovered_report) as recovered:
                    self.assertEqual(recovered.open_report(report_id)['payload'], target.read_bytes())
                menu.recalculate()
                self.assertIsNone(self.c.state.interpretation.table)
                second_id = menu.save_current()
                self.assertNotEqual(second_id, run_id)
                with Archive(self.path) as db:
                    self.assertEqual(len(db.list_runs(report.study_id)), 3)
                menu.open_version(run_id)
                pd.testing.assert_frame_equal(self.c.state.interpretation.table, before.interpretation.table, check_exact=True)

    def test_import_and_version_rollback_and_legacy_incompleteness(self):
        report = self.import_sample('new')
        with Archive(self.path) as db, sample_order('new') as root:
            before = len(db.list_studies(self.well))
            original = db._signal
            count = 0
            def fail(signal):
                nonlocal count
                count += 1
                if count == 3:
                    raise ValueError('injected failure')
                return original(signal)
            with patch.object(db, '_signal', side_effect=fail), self.assertRaises(service.ImportFailure) as error:
                service.import_directory(db, self.well, root, 'new')
            self.assertTrue(error.exception.report.errors)
            self.assertEqual(len(db.list_studies(self.well)), before)
            self.assertEqual(db._db.execute('PRAGMA foreign_key_check').fetchall(), [])
            service.import_directory(db, self.well, root, 'new')
            identity, limits = service.import_trusted_vac(db, report.study_id, ROOT / 'tests/fixtures/current/new/settings.vac')
            self.assertTrue(limits)
            self.assertEqual(db.open_legacy_attachment(identity)['payload'],
                             (ROOT / 'tests/fixtures/current/new/settings.vac').read_bytes())
            self.assertEqual(len(db.list_legacy_attachments(report.study_id)), 1)
            self.assertEqual(db.list_runs(report.study_id), [])
        menu = self.c.archive_menu
        menu.open_study(report.study_id); menu.recalculate()
        with Archive(self.path) as db:
            before = db._db.execute('SELECT count(*) FROM numeric_array').fetchone()[0]
            original = db._array
            def fail_snapshot(array):
                if db._db.execute('SELECT count(*) FROM gui_snapshot').fetchone()[0]:
                    raise ValueError('snapshot failure')
                return original(array)
            with patch.object(db, '_array', side_effect=fail_snapshot), self.assertRaises(ValueError):
                service.save_version(db, report.study_id, self.c.state, capture_widgets(self.c.ui))
            self.assertEqual(db.list_runs(report.study_id), [])
            self.assertEqual(db._db.execute('SELECT count(*) FROM numeric_array').fetchone()[0], before)
        version = menu.save_current()
        with patch.object(self.c, 'calc_to_int', side_effect=AssertionError('recalculation')):
            menu.open_version(version)
        self.assertIsNone(self.c.state.interpretation.table)


if __name__ == '__main__':
    unittest.main()
