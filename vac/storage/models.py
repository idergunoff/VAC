"""Archive transfer objects. No database handles enter domain calculations."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Source:
    name: str
    format: str
    content: bytes


@dataclass
class Signal:
    name: str
    axis: np.ndarray
    values: np.ndarray
    units: str | None = None
    axis_units: str | None = None
    axis_meaning: str | None = None


@dataclass
class InputChannel(Signal):
    source_ordinal: int = 0
    role: str = 'signal'
    transformation: dict = field(default_factory=dict)


@dataclass
class Study:
    equipment: str
    loader_version: str
    sources: list[Source]
    channels: list[InputChannel]
    description: str = ''
    observed_at: str | None = None


@dataclass
class Run:
    channel_ids: list[str]
    parameters: dict
    parameter_schema: str
    algorithm_version: str
    reference_hash: str
    program_version: str
    signals: list[Signal]
    # channel ID -> metric -> (numeric value, unit or None)
    statistics: dict = field(default_factory=dict)
    failure: str | None = None


@dataclass
class Correction:
    signal_id: str
    operation: str
    parameters: dict
    author: str | None = None


@dataclass
class Interval:
    signal_id: str
    start: float
    end: float
    label: str
    boundary: str = 'closed'


@dataclass
class Interpretation:
    class_system: str
    classes: list[str]
    arrays: dict[str, np.ndarray]
    base_id: str | None = None
    corrections: list[Correction] = field(default_factory=list)
    intervals: list[Interval] = field(default_factory=list)
