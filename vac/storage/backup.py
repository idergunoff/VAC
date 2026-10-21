"""Verified SQLite snapshots. Never overwrite a destination or migrate the source."""
import argparse
from contextlib import closing
import json
import logging
import os
from pathlib import Path
import sqlite3
import tempfile

from . import codec
from .session import connect, migrate

log = logging.getLogger(__name__)


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True,
                           isolation_level=None, timeout=5)


def verify(db):
    """Check relational integrity and every embedded payload, without pickle."""
    db.row_factory = sqlite3.Row
    if [r[0] for r in db.execute('PRAGMA integrity_check')] != ['ok']:
        raise ValueError('Archive integrity check failed')
    if db.execute('PRAGMA foreign_key_check').fetchone() is not None:
        raise ValueError('Archive foreign key check failed')
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ('source_file', 'report', 'legacy_attachment'):
        if table not in tables:
            continue
        for row in db.execute(f'SELECT * FROM {table}'):
            if codec.digest(row['payload']) != row['sha256']:
                raise ValueError('Payload checksum mismatch: ' + table)
            if table == 'source_file' and row['size'] != len(row['payload']):
                raise ValueError('Source size mismatch')
    for row in db.execute('SELECT * FROM numeric_array'):
        if row['codec'] != 'npy-v1' or row['size'] != len(row['payload']):
            raise ValueError('Invalid array record')
        codec.decode(row['payload'], row['dtype'], json.loads(row['shape']), row['sha256'])
    return {table: db.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            for table in sorted(tables)}


def _snapshot(source, destination, upgrade=False):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.vac-snapshot-', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with closing(readonly(source)) as src, closing(connect(temporary)) as target:
            # A read transaction pins one consistent snapshot, including WAL data.
            src.execute('BEGIN')
            src.backup(target)
            src.rollback()
            verify(target)
            if upgrade:
                migrate(target)
            else:
                # Validate migration checksums/compatibility on this disposable copy.
                target.execute('BEGIN IMMEDIATE')
                try:
                    migrate(target)
                finally:
                    target.rollback()
            counts = verify(target)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        # Atomic publication that fails if another process created the target.
        os.link(temporary, destination)
        log.info('archive_snapshot_complete upgrade=%s tables=%d', upgrade, len(counts))
        return counts
    finally:
        temporary.unlink(missing_ok=True)


def backup(source, destination):
    """Copy all embedded originals, versions, corrections and reports."""
    return _snapshot(source, destination)


def restore(source, destination):
    """Restore and migrate a verified copy into a new path, preserving the backup."""
    return _snapshot(source, destination, upgrade=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('backup', 'restore', 'verify'))
    parser.add_argument('source')
    parser.add_argument('destination', nargs='?')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    if args.operation == 'verify':
        with closing(readonly(args.source)) as db:
            counts = verify(db)
    else:
        if args.destination is None:
            parser.error('destination is required')
        counts = globals()[args.operation](args.source, args.destination)
    print(json.dumps(counts, sort_keys=True))


if __name__ == '__main__':
    main()
