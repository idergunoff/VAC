import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from vac.domain.state import ApplicationState
from vac.domain.models import (Interpretation, ManualCorrection, ProcessingParameters,
                               ProcessingResult, SignalIdentity)


class StateTests(unittest.TestCase):
    def test_ownership_and_reset(self):
        state, other = ApplicationState(), ApplicationState()
        state.new.signals = pd.DataFrame({'depth': [1., 2.], 1: [3., 4.]})
        state.new.capture_original()
        state.new.signals.at[0, 1] = 99
        original = state.new.original
        self.assertEqual(original.at[0, 1], 3)
        original.at[0, 1] = 42
        self.assertEqual(state.new.original.at[0, 1], 3)
        identity = SignalIdentity('new', 0, '1')
        state.interpretation.signal = identity
        state.interpretation.corrections.append(ManualCorrection(identity, 'shift', ()))
        old = state.old
        state.reset_measurements('new')
        self.assertIs(state.old, old)
        self.assertEqual(state.new.revision, 1)
        self.assertTrue(state.new.statistics.isna().all().all())
        self.assertIsNone(state.interpretation.table)
        self.assertEqual(state.interpretation.corrections, [])
        self.assertEqual(other.new.revision, 0)

    def test_absent_empty_and_stale(self):
        state = ApplicationState()
        self.assertIsNone(state.new.result)
        self.assertIsNone(state.interpretation.table)
        state.interpretation = Interpretation(table=pd.DataFrame())
        self.assertIsNotNone(state.interpretation.table)
        state.new.result = ProcessingResult(ProcessingParameters(0, 1, 0, 0, 1, 1), 0)
        state.invalidate('new')
        self.assertTrue(state.new.result.stale)
        self.assertIsNone(state.interpretation.table)


if __name__ == '__main__':
    unittest.main()
