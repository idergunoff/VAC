"""Mutation checks: a changed class/shape/nonfinite mask must fail comparison."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from regression_compare import compare


class ComparisonTests(unittest.TestCase):
    def test_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            a.mkdir(); b.mkdir()
            for d in [a, b]:
                (d / 'tables.json').write_text(json.dumps({'signals': {'columns': ['quality', 'depth', 'signal']}}))
                np.savez(d / 'signals.npz', **{'0': np.array([0., 1.]), '1': np.array([1., 2.]), '2': np.array([np.nan, 3.])})
            compare(a, b)
            for key, value in [('0', [0., 1.0000000000001]), ('1', [1., 2.0000000000001]), ('2', [0., 3.]), ('2', [np.nan, np.inf]), ('2', [np.nan])]:
                arrays = {'0': np.array([0., 1.]), '1': np.array([1., 2.]), '2': np.array([np.nan, 3.])}
                arrays[key] = np.array(value)
                np.savez(b / 'signals.npz', **arrays)
                with self.assertRaises(AssertionError):
                    compare(a, b)


if __name__ == '__main__':
    unittest.main()
