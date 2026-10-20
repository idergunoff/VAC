"""State lifetime rules, independent of windows and persistence."""
from dataclasses import dataclass, field
from .models import MeasurementSet, Interpretation


@dataclass
class ApplicationState:
    new: MeasurementSet = field(default_factory=lambda: MeasurementSet('new'))
    old: MeasurementSet = field(default_factory=lambda: MeasurementSet('old'))
    interpretation: Interpretation = field(default_factory=Interpretation)

    def reset_measurements(self, equipment):
        previous = getattr(self, equipment)
        replacement = MeasurementSet(equipment)
        replacement.revision = previous.revision + 1
        setattr(self, equipment, replacement)
        self.clear_interpretation()
        return replacement

    def clear_interpretation(self):
        self.interpretation = Interpretation()

    def invalidate(self, equipment):
        getattr(self, equipment).invalidate()
        self.clear_interpretation()
