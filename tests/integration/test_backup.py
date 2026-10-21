"""Disaster recovery, WAL snapshots, migration failure and writer exclusion."""
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
import os
import tempfile
import unittest
from unittest.mock import patch

from test_storage import sample, run_for
from vac.storage import Archive, Interpretation, Correction
from vac.storage.backup import backup, restore, readonly, verify
from vac.storage.session import connect, migrate, transaction
import numpy as np


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='VAC восстановление ')
        self.root = Path(self.tmp.name)
        self.path = self.root / 'archive.sqlite'

    def tearDown(self):
        self.tmp.cleanup()

    def test_restore_all_records_in_clean_directory_and_reopen(self):
        with Archive(self.path) as db:
            well = db.create_well('7')
            study = db.save_study(well, sample())
            run = db.save_run(study, run_for(db, study))
            signal = db.open_run(run)['signals'][0]['id']
            base = db.save_interpretation(run, Interpretation('v1', ['good'], {'cement': np.zeros(3)}))
            manual = db.save_interpretation(run, Interpretation('v1', ['good'],
                {'cement': np.ones(3)}, base_id=base, corrections=[Correction(signal, 'manual', {'x': 1})]))
            report = db.save_report(manual, b'docx-content', details={'well': '7'},
                                    template_version='1', template_hash='template')
            attachment = db.save_legacy_attachment(study, 'settings.vac', b'legacy', {}, ['incomplete'])
            expected_dump = list(db._db.iterdump())
            counts = backup(self.path, self.root / 'backup.sqlite')
        self.path.unlink()  # Disaster: the working archive is gone.
        restored = self.root / 'чистый каталог' / 'restored.sqlite'
        self.assertEqual(restore(self.root / 'backup.sqlite', restored), counts)
        for _ in range(2):
            with Archive(restored) as db:
                self.assertEqual(db.open_study(study).sources, sample().sources)
                self.assertEqual(db.open_run(run)['parameters'], run_for(db, study).parameters)
                self.assertEqual(db.open_interpretation(manual)['corrections'][0]['parameters'], {'x': 1})
                self.assertEqual(db.open_report(report)['payload'], b'docx-content')
                self.assertEqual(db.open_legacy_attachment(attachment)['payload'], b'legacy')
                self.assertEqual(verify(db._db), counts)
                self.assertEqual(list(db._db.iterdump()), expected_dump)
        with self.assertRaises(FileExistsError):
            restore(self.root / 'backup.sqlite', restored)

    def test_wal_snapshot_excludes_uncommitted_changes(self):
        with Archive(self.path) as db:
            db._db.execute('PRAGMA journal_mode=WAL')
            committed = db.create_well('committed')
            with transaction(db._db):
                db.create_well('uncommitted')
                backup(self.path, self.root / 'copy.sqlite')
            with Archive(self.root / 'copy.sqlite') as copy:
                self.assertEqual([r['id'] for r in copy.find_wells()], [committed])
            self.assertEqual(len(db.find_wells()), 2)

    def test_corrupt_payload_never_publishes_destination(self):
        with Archive(self.path) as db:
            db.save_study(db.create_well('7'), sample())
            db._db.execute("UPDATE source_file SET sha256='broken'")
        target = self.root / 'copy.sqlite'
        with self.assertRaises(ValueError):
            backup(self.path, target)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob('.vac-snapshot-*')))
        with self.assertRaises(sqlite3.OperationalError):
            backup(self.root / 'missing.sqlite', target)
        self.assertFalse((self.root / 'missing.sqlite').exists())

    def test_unknown_or_modified_migration_is_rejected(self):
        with Archive(self.path) as db:
            db._db.execute("UPDATE schema_migration SET sha256='modified'")
        with self.assertRaisesRegex(ValueError, 'checksum'):
            backup(self.path, self.root / 'modified.sqlite')
        self.assertFalse((self.root / 'modified.sqlite').exists())
        with closing(connect(self.path)) as db:
            db.execute("UPDATE schema_migration SET version='999_future.sql' WHERE version='001_initial.sql'")
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            restore(self.path, self.root / 'future.sqlite')
        self.assertFalse((self.root / 'future.sqlite').exists())

    def test_upgrade_copy_and_failed_migration_preserve_backup(self):
        folder = self.root / 'migrations'
        folder.mkdir()
        initial = Path(__file__).resolve().parents[2] / 'vac/storage/migrations/001_initial.sql'
        (folder / initial.name).write_bytes(initial.read_bytes())
        with closing(connect(self.path)) as db:
            with patch('vac.storage.session.files') as resources:
                resources.return_value.joinpath.return_value = folder
                migrate(db)
            db.execute("INSERT INTO well VALUES('old','7','{}')")
        before = self.path.read_bytes()
        backup(self.path, self.root / 'stage7-copy.sqlite')
        self.assertEqual(self.path.read_bytes(), before)
        target = self.root / 'upgraded.sqlite'
        restore(self.root / 'stage7-copy.sqlite', target)
        with Archive(target) as db:
            self.assertEqual(db.find_wells()[0]['id'], 'old')
            self.assertEqual(len(db._db.execute('SELECT * FROM schema_migration').fetchall()), 2)
        (folder / '002_bad.sql').write_text('CREATE TABLE probe(id);\nINVALID SQL;\n')
        failed = self.root / 'failed.sqlite'
        with patch('vac.storage.session.files') as resources:
            resources.return_value.joinpath.return_value = folder
            with self.assertRaises(sqlite3.OperationalError):
                restore(self.root / 'stage7-copy.sqlite', failed)
        self.assertFalse(failed.exists())
        self.assertEqual(self.path.read_bytes(), before)
        with closing(readonly(self.root / 'stage7-copy.sqlite')) as db:
            self.assertEqual(len(db.execute('SELECT * FROM schema_migration').fetchall()), 1)

    def test_abrupt_process_exit_recovers_uncommitted_transaction(self):
        with Archive(self.path) as db:
            db.create_well('committed')
        process = subprocess.run([sys.executable, '-c',
            "import os,sys; from vac.storage import Archive; "
            "db=Archive(sys.argv[1]); db._db.execute('BEGIN IMMEDIATE'); "
            "db.create_well('interrupted'); os._exit(17)", str(self.path)],
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2])),
            timeout=20)
        self.assertEqual(process.returncode, 17)
        with Archive(self.path) as db:
            self.assertEqual([r['name'] for r in db.find_wells()], ['committed'])
            db.create_well('after restart')
            verify(db._db)

    def test_writer_lock_rollback_and_retry(self):
        with Archive(self.path) as first, Archive(self.path) as second:
            second._db.execute('PRAGMA busy_timeout=20')
            with self.assertRaises(RuntimeError):
                with transaction(first._db):
                    first.create_well('rolled back')
                    with self.assertRaisesRegex(sqlite3.OperationalError, 'locked'):
                        second.create_well('blocked')
                    raise RuntimeError('process operation failed')
            second.create_well('retry')
        with Archive(self.path) as reopened:
            self.assertEqual([r['name'] for r in reopened.find_wells()], ['retry'])


if __name__ == '__main__':
    unittest.main()
