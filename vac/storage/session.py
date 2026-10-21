"""SQLite connection and checksummed, transactional schema migrations."""
from contextlib import contextmanager
from hashlib import sha256
from importlib.resources import files
import sqlite3


def connect(path):
    db = sqlite3.connect(path, isolation_level=None, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('PRAGMA busy_timeout=5000')
    return db


@contextmanager
def transaction(db):
    if db.in_transaction:
        raise RuntimeError('Nested archive transaction')
    db.execute('BEGIN IMMEDIATE')
    try:
        yield
        db.commit()
    except BaseException:
        db.rollback()
        raise


def migrate(db):
    """Never use executescript: it implicitly commits an active transaction."""
    migrations = sorted(files('vac.storage').joinpath('migrations').iterdir(), key=lambda p: p.name)
    with transaction(db):
        db.execute('CREATE TABLE IF NOT EXISTS schema_migration(version TEXT PRIMARY KEY, sha256 TEXT NOT NULL, applied_at TEXT NOT NULL)')
        known = {r['version']: r['sha256'] for r in db.execute('SELECT * FROM schema_migration')}
        available = {p.name for p in migrations if p.name.endswith('.sql')}
        if known.keys() - available:
            raise ValueError('Archive has unsupported migrations')
        for path in migrations:
            if path.name not in available:
                continue
            data = path.read_bytes()
            digest = sha256(data).hexdigest()
            if path.name in known:
                if known[path.name] != digest:
                    raise ValueError('Migration checksum mismatch: ' + path.name)
                continue
            statement = ''
            for line in data.decode('utf-8').splitlines(keepends=True):
                statement += line
                if sqlite3.complete_statement(statement):
                    db.execute(statement)
                    statement = ''
            if statement.strip():
                raise ValueError('Incomplete migration')
            db.execute("INSERT INTO schema_migration VALUES(?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))", (path.name, digest))
