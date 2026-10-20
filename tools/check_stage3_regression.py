"""Check preserved scenarios and explicitly changed state lifetime against stage 1.

The strict stage-1 comparator remains unchanged: full comparison must report the
intentional changes at restore/reload. No baseline is rewritten by this tool.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from regression_compare import compare


def check(base, actual):
    with tempfile.TemporaryDirectory() as tmp:
        expected, candidate = Path(tmp) / 'expected', Path(tmp) / 'actual'
        for src, dst in [(base, expected), (actual, candidate)]:
            dst.mkdir()
            for path in src.iterdir():
                if path.is_dir() and path.name[:2].isdigit() and int(path.name[:2]) <= 10:
                    shutil.copytree(path, dst / path.name)
                elif path.name == 'report.json':
                    shutil.copy2(path, dst / path.name)
        compare(expected, candidate)
    bm = json.loads((base / 'manifest.json').read_text())
    am = json.loads((actual / 'manifest.json').read_text())
    for key in ['device', 'case', 'python', 'inputs', 'read_order', 'steps', 'qt_errors']:
        assert bm[key] == am[key], key
    # pip is a build tool, not a runtime dependency. All recorded runtime versions match.
    assert {k: v for k, v in bm['dependencies'].items() if k != 'pip'} == {
        k: v for k, v in am['dependencies'].items() if k != 'pip'}
    # Restore still produces identical signals and statistics. Only classifications
    # previously contaminated by the earlier manual interval are allowed to differ.
    before, after = base / '11-restore', actual / '11-restore'
    metadata = json.loads((before / 'tables.json').read_text())
    assert metadata == json.loads((after / 'tables.json').read_text())
    manual = json.loads((base / '10-save' / 'intervals.json').read_text())
    settings = json.loads((before / 'settings.json').read_text())
    for name, info in metadata.items():
        with np.load(before / (name + '.npz')) as x, np.load(after / (name + '.npz')) as y:
            assert x.files == y.files
            for key in x.files:
                col = str(info['columns'][int(key)])
                if name.startswith('all_signals') and ('qual' in col or col.startswith('match_def')):
                    assert not np.any(y[key] == 3), (name, col, 'manual interval leaked')
                    depth_index = str(info['columns'].index('depth'))
                    depth = x[depth_index]
                    start = int(np.argmin(abs(depth - settings['doubleSpinBox_int_min'])))
                    allowed = np.zeros(len(depth), dtype=bool)
                    for intervals in manual.values():
                        for lo, hi in intervals:
                            # Legacy removes runs shorter than 17 samples, affecting adjacent edges.
                            allowed[max(0, start + lo - 17):start + hi + 17] = True
                    np.testing.assert_array_equal(x[key][~allowed], y[key][~allowed])
                else:
                    np.testing.assert_allclose(x[key], y[key], atol=1e-12, rtol=1e-12, equal_nan=True)
    for path in before.glob('*'):
        if path.suffix == '.json' and path.name not in ('classes.json',):
            assert json.loads(path.read_text()) == json.loads((after / path.name).read_text()), path.name
        elif path.suffix == '.npz' and path.stem not in metadata:
            with np.load(path) as x, np.load(after / path.name) as y:
                assert x.files == y.files
                for key in x.files:
                    np.testing.assert_allclose(x[key], y[key], atol=1e-12, rtol=1e-12, equal_nan=True)
    reload = actual / '12-reload'
    tables = json.loads((reload / 'tables.json').read_text())
    assert 'cement_sig' not in tables, 'old cementogram survived reload'
    assert all(not v for v in json.loads((reload / 'intervals.json').read_text()).values())
    suffix = '' if am['device'] == 'new' else '_old'
    statistics = tables['all_stat' + suffix]
    name_index = statistics['columns'].index('name')
    sum_start = 9 if am['device'] == 'new' else 13
    for index, name in zip(statistics['index'], statistics['data'][str(name_index)]):
        if index >= sum_start:
            assert name is None, 'old sum statistics survived reload'
    print('PASS stage 3:', am['device'], 'steps 01–10 + report unchanged; restore/reload lifetime verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    args = parser.parse_args()
    check(args.baseline, args.actual)
