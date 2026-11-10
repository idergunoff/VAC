"""Local edits preserve data outside their support and never mutate the source."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from vac.processing.corrections import local_shift, local_shift_candidates


class LocalShiftTests(unittest.TestCase):
    def setUp(self):
        self.source = pd.DataFrame({'depth': [-1., 0., 1., 2., 3., 4., 5.],
            'corr_sig': [2., 1., 3., 5., 2., 1., 7.], 'first_sig': 42., 'corr_coeff': 1.},
            index=[10, 20, 30, 40, 50, 60, 70])

    def test_peak_and_trough_preserve_bounds_outside_and_source(self):
        before = self.source.copy(deep=True)
        for delta in [4., -8.]:
            result = local_shift(self.source, 0., 4., 2., delta)
            np.testing.assert_allclose(result.corr_sig - before.corr_sig,
                                       [0., 0., delta / 2, delta, delta / 2, 0., 0.])
            pd.testing.assert_frame_equal(result.drop(columns='corr_sig'), before.drop(columns='corr_sig'))
        pd.testing.assert_frame_equal(self.source, before)

    def test_irregular_descending_depth_uses_distance_not_row_number(self):
        source = pd.DataFrame({'depth': [8., 4., 3., 1., 0.], 'corr_sig': [2., 1., 3., 5., 4.]})
        result = local_shift(source, 0., 8., 4., 10.)
        np.testing.assert_allclose(result.corr_sig - source.corr_sig, [0., 10., 8.4375, 1.5625, 0.])

    def test_repeated_edits_accumulate_and_reverse(self):
        first = local_shift(self.source, 0., 4., 2., 3.)
        second = local_shift(first, 0., 4., 2., 3.)
        self.assertEqual(second.corr_sig.iloc[3], 11.)
        pd.testing.assert_frame_equal(local_shift(second, 0., 4., 2., -6.), self.source)

    def test_missing_samples_are_preserved(self):
        source = self.source.copy()
        source.loc[30, 'corr_sig'] = np.nan
        source.loc[50, 'depth'] = np.nan
        result = local_shift(source, 0., 4., 2., 2.)
        self.assertTrue(np.isnan(result.loc[30, 'corr_sig']))
        self.assertEqual(result.loc[50, 'corr_sig'], source.loc[50, 'corr_sig'])
        self.assertEqual(result.loc[40, 'corr_sig'], 7.)

    def test_empty_and_zero_change(self):
        pd.testing.assert_frame_equal(local_shift(self.source, 0., 4., 2., 0.), self.source)
        pd.testing.assert_frame_equal(local_shift(self.source.iloc[:0], 0., 4., 2., 1.), self.source.iloc[:0])

    def test_invalid_bounds_anchor_and_nonfinite_parameters(self):
        for args in [(0, 0, 0, 1), (4, 0, 2, 1), (0, 4, 0, 1), (0, 4, 4, 1),
                     (0, 4, -1, 1), (0, np.inf, 2, 1), (0, 4, 2, np.nan)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                local_shift(self.source, *args)

    def test_both_endpoints_have_one_sided_falloff_without_division_by_zero(self):
        for source in (self.source, self.source.iloc[::-1]):
            for bounds, anchor, delta, expected in [
                ((-1., 3.), -1., 4., [4., 3.375, 2., .625, 0., 0., 0.]),
                ((1., 5.), 5., -4., [0., 0., 0., -.625, -2., -3.375, -4.]),
            ]:
                with self.subTest(anchor=anchor, descending=source.depth.iloc[0] == 5.):
                    before = source.copy(deep=True)
                    with np.errstate(divide='raise', invalid='raise'):
                        result = local_shift(source, *bounds, anchor, delta)
                    change = (result.corr_sig - source.corr_sig).sort_index()
                    np.testing.assert_allclose(change, expected)
                    pd.testing.assert_frame_equal(source, before)
                    pd.testing.assert_frame_equal(result.drop(columns='corr_sig'), source.drop(columns='corr_sig'))

    def test_candidates_only_release_true_endpoints_and_require_nonzero_width(self):
        np.testing.assert_array_equal(local_shift_candidates(self.source, -1., 5.), np.arange(7))
        np.testing.assert_array_equal(local_shift_candidates(self.source, 0., 4.), [2, 3, 4])
        np.testing.assert_array_equal(local_shift_candidates(self.source, -1., -.5), [0])
        for bounds in [(-1., -1.), (5., 5.), (5., -1.)]:
            self.assertEqual(len(local_shift_candidates(self.source, *bounds)), 0)
        missing = self.source.copy()
        missing.iloc[0, missing.columns.get_loc('corr_sig')] = np.nan
        np.testing.assert_array_equal(local_shift_candidates(missing, -1., 5.), np.arange(1, 7))
        with self.assertRaises(ValueError):
            local_shift(missing, -1., 5., -1., 2.)

    def test_endpoint_with_no_interior_sample_and_repeated_edits(self):
        raised = local_shift(self.source, -1., -.5, -1., 2.)
        np.testing.assert_allclose(raised.corr_sig - self.source.corr_sig, [2., 0., 0., 0., 0., 0., 0.])
        pd.testing.assert_frame_equal(local_shift(raised, -1., -.5, -1., -2.), self.source)


if __name__ == '__main__':
    unittest.main()
