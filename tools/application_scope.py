"""Read-only view for regression tools written against the legacy module."""
from collections.abc import Mapping

_FIELDS = {
    'all_signals': 'new.signals', 'all_signals_old': 'old.signals',
    'all_stat': 'new.statistics', 'all_stat_old': 'old.statistics',
    'int_min': 'new.interval_min', 'int_max': 'new.interval_max',
    'int_min_old': 'old.interval_min', 'int_max_old': 'old.interval_max',
    'k_oldtonew': 'old.scale', 'cement_sig': 'interpretation.table',
    'int_undefine_cement': 'interpretation.intervals.undefined',
    'int_0_cement': 'interpretation.intervals.good',
    'int_1_cement': 'interpretation.intervals.partial',
    'int_2_cement': 'interpretation.intervals.absent',
    'F': 'interpretation.frequency', 'k_damp': 'interpretation.damping',
    'options': 'interpretation.options',
}


class ApplicationScope(Mapping):
    def __init__(self, controller):
        self.controller = controller

    def __getitem__(self, key):
        if key in _FIELDS:
            value = self.controller.state
            for field in _FIELDS[key].split('.'):
                value = getattr(value, field)
            return value
        if key == 'MainWindow':
            return self.controller.window
        try:
            return getattr(self.controller, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __iter__(self):
        return iter([*_FIELDS, 'ui', 'MainWindow', *(
            name for name in dir(self.controller) if not name.startswith('_') and callable(getattr(self.controller, name)))])

    def __len__(self):
        return sum(1 for _ in self)
