"""Comparison of recorded legacy runs; never rewrites references."""
import json
from pathlib import Path
import numpy as np


def compare(base, actual):
    # Exact comparisons for structure, labels, boundaries, nonfinite masks and metadata.
    # Floating results allow only 1e-12 absolute + relative roundoff on this pinned stack.
    bfiles = {str(p.relative_to(base)) for p in base.rglob('*') if p.suffix in ['.json', '.npz']}
    afiles = {str(p.relative_to(actual)) for p in actual.rglob('*') if p.suffix in ['.json', '.npz']}
    assert bfiles == afiles, ('files', bfiles ^ afiles)
    for path in sorted(bfiles):
        if path == 'manifest.json':
            b, a = [json.loads((d / path).read_text()) for d in [base, actual]]
            for key in ['upstream', 'compatibility', 'device', 'case', 'python', 'dependencies', 'inputs', 'read_order', 'steps', 'qt_errors']:
                assert b[key] == a[key], (path, key)
        elif path.endswith('.json'):
            assert json.loads((base / path).read_text()) == json.loads((actual / path).read_text()), path
        else:
            with np.load(base / path, allow_pickle=False) as b, np.load(actual / path, allow_pickle=False) as a:
                assert b.files == a.files, path
                for key in b.files:
                    x, y = b[key], a[key]
                    assert x.shape == y.shape and x.dtype == y.dtype, (path, key, 'structure')
                    for mask in [np.isnan, np.isposinf, np.isneginf]:
                        assert np.array_equal(mask(x), mask(y)), (path, key, 'nonfinite')
                    table_path = base / Path(path).with_name('tables.json')
                    column = ''
                    if table_path.exists() and Path(path).stem != 'curves' and Path(path).stem != 'matplotlib':
                        table_info = json.loads(table_path.read_text()).get(Path(path).stem)
                        if table_info:
                            column = str(table_info['columns'][int(key)])
                    if 'qual' in column or column == 'depth':
                        np.testing.assert_array_equal(x, y)
                    else:
                        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=path + ':' + key)
    print('PASS regression', actual.name)
