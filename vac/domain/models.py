"""Minimal legacy-compatible entities. Technical axis units are not inferred."""
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Equipment = Literal['new', 'old']
STAT_COLUMNS = ['name', 'useful_depth', 'max_min', 'coeff_A', 'coeff_B', 'k_damp', 'max_h']


@dataclass(frozen=True)
class Channel:
    number: int
    source: str


@dataclass(frozen=True)
class ProcessingParameters:
    interval_min: float
    interval_max: float
    normalization: float
    function_shift: float
    difference_scale: int
    mean_window: int


@dataclass
class ProcessingResult:
    parameters: ProcessingParameters
    revision: int
    stale: bool = False


@dataclass
class MeasurementSet:
    equipment: Equipment
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    statistics: pd.DataFrame = field(init=False)
    channels: list[Channel] = field(default_factory=list)
    _original: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    interval_min: int | None = None
    interval_max: int | None = None
    scale: float = 1
    revision: int = 0
    result: ProcessingResult | None = None

    def __post_init__(self):
        self.statistics = pd.DataFrame(index=range(1, 12 if self.equipment == 'new' else 16), columns=STAT_COLUMNS)

    @property
    def original(self):
        """Return an owned copy: processing and callers cannot overwrite the input."""
        return self._original.copy(deep=True)

    def capture_original(self):
        columns = [c for c in self.signals if c in ('depth', 'noise') or isinstance(c, int)]
        self._original = self.signals.loc[:, columns].copy(deep=True)
        self.channels = [Channel(int(i), str(self.statistics.at[i, 'name'])) for i in self.statistics.index
                         if pd.notna(self.statistics.at[i, 'name'])]

    def invalidate(self):
        if self.result is not None:
            self.result.stale = True


@dataclass
class QualityIntervals:
    undefined: list = field(default_factory=list)
    good: list = field(default_factory=list)
    partial: list = field(default_factory=list)
    absent: list = field(default_factory=list)


@dataclass(frozen=True)
class SignalIdentity:
    equipment: Equipment
    revision: int
    signal: str


@dataclass(frozen=True)
class ManualCorrection:
    signal: SignalIdentity
    operation: str
    parameters: tuple


@dataclass(frozen=True)
class ReportDetails:
    fields: tuple[tuple[str, str], ...] = ()


@dataclass
class Interpretation:
    # None means no interpretation; an empty DataFrame is a valid empty result.
    table: pd.DataFrame | None = None
    signal: SignalIdentity | None = None
    intervals: QualityIntervals = field(default_factory=QualityIntervals)
    corrections: list[ManualCorrection] = field(default_factory=list)
    frequency: str = ''
    damping: float | None = None
    options: dict | None = None
    report: ReportDetails = field(default_factory=ReportDetails)
