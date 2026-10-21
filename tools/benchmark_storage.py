"""Stage 6 experiment, not the production storage layer. Run from any directory."""
import argparse
import hashlib
import io
import json
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vac.io import lvm, twf


def timed(call, repeats=7):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        values.append((time.perf_counter() - start) * 1000)
    return result, round(statistics.median(values), 3)


def encode(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def decode(data):
    return np.load(io.BytesIO(data), allow_pickle=False)


def compare_storage(signals, directory):
    """Same float64 pairs in every variant; middle 10% interval, all channels."""
    lo, hi = np.quantile(signals[0][:, 0], [0.45, 0.55])
    expected = [a[(a[:, 0] >= lo) & (a[:, 0] <= hi)] for a in signals]
    results = {}
    for variant in ('rows', 'whole_blob', 'blocks_1024', 'external_npy'):
        folder = directory / variant
        folder.mkdir()
        db = sqlite3.connect(folder / 'archive.sqlite')
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('CREATE TABLE files(channel INTEGER PRIMARY KEY, path TEXT)')
        db.execute('CREATE TABLE samples(channel INTEGER, sample INTEGER, depth REAL, value REAL, PRIMARY KEY(channel,sample))')
        db.execute('CREATE INDEX by_depth ON samples(channel,depth)')
        db.execute('CREATE TABLE blocks(channel INTEGER, start INTEGER, low REAL, high REAL, data BLOB, PRIMARY KEY(channel,start))')
        db.execute('CREATE INDEX block_bounds ON blocks(channel,low,high)')
        start = time.perf_counter()
        with db:
            for channel, array in enumerate(signals):
                if variant == 'rows':
                    db.executemany('INSERT INTO samples VALUES(?,?,?,?)',
                                   ((channel, i, float(d), float(v)) for i, (d, v) in enumerate(array)))
                elif variant == 'external_npy':
                    name = f'{channel}.npy'
                    np.save(folder / name, array, allow_pickle=False)
                    db.execute('INSERT INTO files VALUES(?,?)', (channel, name))
                else:
                    size = len(array) if variant == 'whole_blob' else 1024
                    for i in range(0, len(array), size):
                        block = array[i:i + size]
                        db.execute('INSERT INTO blocks VALUES(?,?,?,?,?)',
                                   (channel, i, float(block[:, 0].min()), float(block[:, 0].max()), encode(block)))
        write_ms = (time.perf_counter() - start) * 1000

        def read(interval=False):
            arrays = []
            for channel in range(len(signals)):
                if variant == 'rows':
                    query = 'SELECT depth,value FROM samples WHERE channel=?'
                    params = (channel,)
                    if interval:
                        query += ' AND depth>=? AND depth<=?'
                        params += (float(lo), float(hi))
                    array = np.asarray(db.execute(query + ' ORDER BY sample', params).fetchall(), dtype='<f8').reshape(-1, 2)
                elif variant == 'external_npy':
                    name = db.execute('SELECT path FROM files WHERE channel=?', (channel,)).fetchone()[0]
                    array = np.load(folder / name, allow_pickle=False)
                else:
                    query = 'SELECT data FROM blocks WHERE channel=?'
                    params = (channel,)
                    if interval:
                        query += ' AND high>=? AND low<=?'
                        params += (float(lo), float(hi))
                    chunks = [decode(row[0]) for row in db.execute(query + ' ORDER BY start', params)]
                    array = np.concatenate(chunks) if chunks else np.empty((0, 2))
                if interval:
                    array = array[(array[:, 0] >= lo) & (array[:, 0] <= hi)]
                arrays.append(array)
            return arrays

        full, full_ms = timed(read)
        part, interval_ms = timed(lambda: read(True))
        for actual, target in zip(full, signals):
            np.testing.assert_array_equal(actual, target)
        for actual, target in zip(part, expected):
            np.testing.assert_array_equal(actual, target)
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        db.close()
        results[variant] = dict(bytes=sum(p.stat().st_size for p in folder.iterdir()),
                                write_ms=round(write_ms, 3), full_read_ms=full_ms,
                                interval_read_ms=interval_ms, exact_roundtrip=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(python=platform.python_version(), platform=platform.platform(),
                  numpy=np.__version__, sqlite=sqlite3.sqlite_version,
                  timing='median of 7 warm reads; one write incl. commit; no cold-cache claim', datasets={})
    with tempfile.TemporaryDirectory(prefix='vac-storage-') as scratch:
        for device, name, loader in (
            ('new', 'новое оборудование', lambda p: lvm.load_directory(p, 5)),
            ('old', 'старое оборудование', twf.load_directory),
        ):
            source = ROOT / 'пример' / name
            files = sorted(p for p in source.rglob('*') if p.suffix.lower() in ('.lvm', '.twf'))
            measurements, load_ms = timed(lambda: loader(str(source)), 3)
            measurement, order = measurements
            original = measurement.original
            depth = original['depth'].to_numpy(dtype='<f8')
            signals = [np.column_stack((depth, original[c].to_numpy(dtype='<f8')))
                       for c in original if c != 'depth']
            folder = Path(scratch) / device
            folder.mkdir()
            stages = {}
            for stage in sorted((ROOT / 'tests/baselines/stage-1' / device).iterdir()):
                if not stage.is_dir():
                    continue
                size = 0
                for file in stage.glob('*.npz'):
                    with np.load(file, allow_pickle=False) as arrays:
                        size += sum(arrays[k].nbytes for k in arrays.files)
                stages[stage.name] = size
            report['datasets'][device] = dict(
                files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            raw_samples=(len(twf.read_twf(p)) if device == 'old' else len(lvm.read_lvm(p)))) for p in files],
                channel_order=order, channels_including_noise=len(signals),
                loaded_samples_per_channel=len(depth), raw_source_bytes=sum(p.stat().st_size for p in files),
                load_with_legacy_preprocessing_ms=load_ms,
                original_table_bytes=int(original.memory_usage(index=True, deep=True).sum()),
                baseline_numeric_bytes_by_stage=stages,
                interval_bounds=[float(x) for x in np.quantile(depth, [0.45, 0.55])],
                storage=compare_storage(signals, folder))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
