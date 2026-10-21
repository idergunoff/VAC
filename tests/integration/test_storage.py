"""Archive contracts: lossless records, provenance, isolation and rollback."""
import copy
import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vac.storage import Archive, Source, Signal, InputChannel, Study, Run, Interpretation, Correction, Interval
from vac.storage import codec
from vac.storage.session import migrate
from vac.storage.session import connect
from vac.io import lvm, twf


def sample():
    axis = np.array([0., .515, 1.03], dtype='>f8')
    return Study('new', 'loader-test-1', [Source('сигнал 1.lvm', 'lvm', b'original')], [
        InputChannel('1', axis, np.array([np.nan, np.inf, -0.]), role='signal'),
        InputChannel('noise', axis, np.array([1., 2., 3.]), role='noise')], 'описание')


def run_for(db, study):
    ids = [c['id'] for c in db.get_channels(study)]
    return Run(ids, {'all_gui_settings': {'канал': 1}, 'operations': ['sum', 'quality']},
               'gui-v1', 'legacy-1', 'reference-sha256', 'program-sha',
               [Signal('sum', np.arange(3.), np.array([1., np.nan, np.inf]))],
               {ids[0]: {'damping': (np.float64(np.nan), None)}})


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='vac-архив ')
        self.path = Path(self.tmp.name) / 'архив 1.sqlite'
        self.db = Archive(self.path)
        self.well = self.db.create_well('Скважина 7', {'месторождение': None})

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def counts(self):
        tables = ['study', 'source_file', 'numeric_array', 'axis', 'channel', 'processing_run',
                  'result_signal', 'result_statistics', 'interpretation', 'manual_correction', 'quality_interval', 'report']
        return {t: self.db._db.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in tables}

    def test_empty_migration_repeat_checksum_and_unknown_version(self):
        migrate(self.db._db)
        self.assertEqual(self.db._db.execute('SELECT count(*) FROM schema_migration').fetchone()[0], 2)
        self.db._db.execute("UPDATE schema_migration SET sha256='bad'")
        with self.assertRaises(ValueError):
            migrate(self.db._db)
        self.db._db.execute("UPDATE schema_migration SET version='999_future.sql' WHERE version='001_initial.sql'")
        with self.assertRaises(ValueError):
            migrate(self.db._db)

    def test_migration_failure_rolls_back_ddl_and_journal(self):
        folder = Path(self.tmp.name) / 'migration'
        folder.mkdir()
        (folder / '001_bad.sql').write_text('CREATE TABLE probe(id INTEGER);\nINVALID SQL;\n')
        raw = sqlite3.connect(':memory:', isolation_level=None)
        with patch('vac.storage.session.files') as resource:
            resource.return_value.joinpath.return_value = folder
            with self.assertRaises(sqlite3.OperationalError):
                migrate(raw)
        self.assertEqual(raw.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [])
        raw.close()

    def test_upgrade_stage7_archive_preserves_existing_data(self):
        folder = Path(self.tmp.name) / 'stage7-schema'
        folder.mkdir()
        (folder / '001_initial.sql').write_bytes((ROOT / 'vac/storage/migrations/001_initial.sql').read_bytes())
        path = Path(self.tmp.name) / 'stage7.sqlite'
        raw = connect(path)
        with patch('vac.storage.session.files') as resource:
            resource.return_value.joinpath.return_value = folder
            migrate(raw)
        raw.execute("INSERT INTO well VALUES('existing','Старая скважина','{}')")
        raw.close()
        with Archive(path) as upgraded:
            self.assertEqual(upgraded.find_wells()[0]['id'], 'existing')
            self.assertEqual(upgraded._db.execute('SELECT count(*) FROM schema_migration').fetchone()[0], 2)
            study = upgraded.save_study('existing', sample())
            self.assertEqual(upgraded.open_study(study).sources, sample().sources)

    def test_roundtrip_duplicates_owned_arrays_and_reopen(self):
        study = self.db.save_study(self.well, sample())
        self.assertEqual(self.db.find_wells('Скважина')[0]['id'], self.well)
        other_well = self.db.create_well('Скважина 7')
        self.assertNotEqual(other_well, self.well)
        self.db.save_study(other_well, sample())
        match = self.db.find_sources(sample().sources[0])
        self.assertEqual(len(match), 2)
        self.assertTrue(all(m['same_content'] for m in match))
        changed = sample()
        changed.sources[0].content = b'changed'
        self.db.save_study(self.well, changed)
        matches = self.db.find_sources(changed.sources[0])
        self.assertEqual(sum(m['same_content'] for m in matches), 1)
        self.assertTrue(all(m['same_name'] for m in matches))
        channels = self.db.get_channels(study)
        part = self.db.read_interval(channels[0]['id'], .515, 1.03)
        np.testing.assert_array_equal(part.axis, sample().channels[0].axis[1:])
        self.db.close()
        self.db = Archive(self.path)
        restored = self.db.open_study(study)
        self.assertEqual(restored.sources, sample().sources)
        for actual, target in zip(restored.channels, sample().channels):
            self.assertEqual(actual.axis.dtype, target.axis.dtype)
            self.assertEqual(actual.axis.tobytes(), target.axis.tobytes())
            self.assertEqual(actual.values.tobytes(), target.values.tobytes())
            self.assertIsNone(actual.axis_units)
        restored.channels[0].values[:] = 0
        self.assertTrue(np.isnan(self.db.open_study(study).channels[0].values[0]))

    def test_atomic_study_and_run_failures(self):
        before = self.counts()
        bad = sample()
        bad.channels[1].values = np.ones(4)
        with self.assertRaises(ValueError):
            self.db.save_study(self.well, bad)
        self.assertEqual(before, self.counts())
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.save_study('missing-well', sample())
        self.assertEqual(before, self.counts())
        study = self.db.save_study(self.well, sample())
        other = self.db.save_study(self.well, sample())
        before = self.counts()
        bad_run = run_for(self.db, study)
        bad_run.channel_ids.append(self.db.get_channels(other)[0]['id'])
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.save_run(study, bad_run)
        self.assertEqual(before, self.counts())
        bad_run = run_for(self.db, study)
        bad_run.signals.append(Signal('bad', np.arange(2.), np.arange(3.)))
        with self.assertRaises(ValueError):
            self.db.save_run(study, bad_run)
        self.assertEqual(before, self.counts())
        with self.assertRaises(sqlite3.IntegrityError):
            self.db._db.execute('DELETE FROM well WHERE id=?', (self.well,))
        with self.assertRaises(sqlite3.IntegrityError):
            self.db._db.execute('DELETE FROM study WHERE id=?', (study,))
        self.assertEqual(self.db._db.execute('PRAGMA foreign_keys').fetchone()[0], 1)

    def test_versions_corrections_report_provenance_and_rollback(self):
        study = self.db.save_study(self.well, sample())
        original = self.db.open_study(study).channels[0].values.tobytes()
        spec = run_for(self.db, study)
        run = self.db.save_run(study, spec)
        spec.parameters['operations'].append('new-operation')
        second = self.db.save_run(study, spec)
        self.assertNotEqual(run, second)
        self.assertEqual(len(self.db.list_runs(study)), 2)
        opened = self.db.open_run(run)
        self.assertEqual(opened['parameters']['operations'], ['sum', 'quality'])
        self.assertTrue(np.isnan(opened['statistics'][spec.channel_ids[0]]['damping'][0]))
        signal = opened['signals'][0]['id']
        base = self.db.save_interpretation(run, Interpretation('legacy-class-1', ['good', 'absent'], {'cement': np.array([1., 0., 1.])}))
        correction = Correction(signal, 'set-class', {'from': 0., 'to': 1., 'class': 'absent'}, 'Иван')
        version = Interpretation('legacy-class-1', ['good', 'absent'], {'cement': np.array([0., 0., 1.])}, base,
                                 [correction], [Interval(signal, 0., 1., 'absent')])
        manual = self.db.save_interpretation(run, version)
        report_bytes = (ROOT / 'vac/resources/template.docx').read_bytes()
        report = self.db.save_report(manual, report_bytes, details={'скважина': '7'}, template_version='1', template_hash=hashlib.sha256((ROOT / 'vac/resources/template.docx').read_bytes()).hexdigest())
        before = self.counts()
        broken = copy.deepcopy(version)
        broken.intervals[0].label = 'unknown'
        with self.assertRaises(ValueError):
            self.db.save_interpretation(run, broken)
        self.assertEqual(before, self.counts())
        with self.assertRaises(ValueError):
            self.db.save_interpretation(second, version)
        self.assertEqual(before, self.counts())
        self.db.close()
        self.db = Archive(self.path)
        self.assertEqual(self.db.open_report(report)['payload'], report_bytes)
        self.assertEqual(self.db.open_report(report)['interpretation_id'], manual)
        self.assertEqual(self.db.list_reports(manual), [report])
        reopened = self.db.open_interpretation(manual)
        self.assertEqual(reopened['base_id'], base)
        self.assertEqual(reopened['corrections'][0]['parameters'], correction.parameters)
        self.assertEqual(len(self.db.list_interpretations(run)), 2)
        np.testing.assert_array_equal(self.db.open_interpretation(base)['arrays']['cement'], [1., 0., 1.])
        self.assertEqual(self.db.open_study(study).channels[0].values.tobytes(), original)
        failed = run_for(self.db, study)
        failed.signals, failed.statistics, failed.failure = [], {}, 'Расчёт не выполнен'
        failed_id = self.db.save_run(study, failed)
        self.assertEqual(self.db.open_run(failed_id)['status'], 'failed')
        with self.assertRaises(ValueError):
            self.db.save_interpretation(failed_id, Interpretation('v1', ['a'], {'a': np.ones(1)}))
        self.assertEqual(self.db._db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
        self.assertEqual(self.db._db.execute('PRAGMA foreign_key_check').fetchall(), [])

    def test_codec_invalid_metadata_objects_and_corruption(self):
        for array in [np.array([object()]), np.array(['text'])]:
            with self.assertRaises(ValueError):
                codec.encode(array)
        array, data = codec.encode(np.array([np.nan, np.inf, -np.inf, -0.], dtype='>f8'))
        empty, empty_data = codec.encode(np.empty((0, 3)))
        self.assertEqual(codec.decode(empty_data, empty.dtype.str, [0, 3], codec.digest(empty_data)).shape, (0, 3))
        result = codec.decode(data, array.dtype.str, [4], codec.digest(data))
        self.assertEqual(result.tobytes(), array.tobytes())
        with self.assertRaises(ValueError):
            codec.decode(data, array.dtype.str, [99], codec.digest(data))
        with self.assertRaises(ValueError):
            codec.decode(data[:-1], array.dtype.str, [4], codec.digest(data))
        study = self.db.save_study(self.well, sample())
        self.db._db.execute("UPDATE source_file SET sha256='bad' WHERE study_id=?", (study,))
        with self.assertRaises(ValueError):
            self.db.open_study(study)

    def test_real_lvm_and_twf_exact_roundtrip(self):
        for equipment, folder in [('new', 'новое оборудование'), ('old', 'старое оборудование')]:
            with self.subTest(equipment=equipment):
                root = ROOT / 'пример' / folder
                if equipment == 'new':
                    loaded, order = lvm.load_directory(root, 5)
                else:
                    loaded, order = twf.load_directory(root)
                paths = {p.name: p for p in root.rglob('*') if p.is_file()}
                sources = [Source(name, 'lvm' if equipment == 'new' else 'twf', paths[name].read_bytes()) for name in order]
                frame = loaded.original
                channels = []
                for column in frame:
                    if column == 'depth':
                        continue
                    index = (8 if equipment == 'new' else 0) if column == 'noise' else (int(column)-1 if equipment == 'new' else int(column))
                    channels.append(InputChannel(str(column), frame.depth.to_numpy(), frame[column].to_numpy(),
                        source_ordinal=index, role='noise' if column == 'noise' else 'signal',
                        transformation={'loader': equipment, 'mean': 5}))
                study = self.db.save_study(self.well, Study(equipment, 'stage4-loaders', sources, channels))
                self.db.close()
                self.db = Archive(self.path)
                restored = self.db.open_study(study)
                self.assertEqual(restored.sources, sources)
                for actual, expected in zip(restored.channels, channels):
                    self.assertEqual(actual.axis.tobytes(), expected.axis.tobytes())
                    self.assertEqual(actual.values.tobytes(), expected.values.tobytes())
                    self.assertEqual(actual.source_ordinal, expected.source_ordinal)
                    self.assertEqual(actual.role, expected.role)
                c = self.db.get_channels(study)[0]
                lo, hi = np.quantile(channels[0].axis, [.45, .55])
                part = self.db.read_interval(c['id'], lo, hi)
                mask = (channels[0].axis >= lo) & (channels[0].axis <= hi)
                np.testing.assert_array_equal(part.values, channels[0].values[mask])


if __name__ == '__main__':
    unittest.main()
