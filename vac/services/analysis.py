"""Headless analysis. Own the working copy; never write files or initialize Qt."""
from copy import deepcopy
from dataclasses import dataclass, replace

from ..calculation import calculate_new, calculate_old
from ..domain.models import MeasurementSet, ProcessingParameters, ProcessingResult
from ..processing import preprocessing, interpretation


@dataclass
class AnalysisResult:
    measurements: MeasurementSet
    thresholds: tuple[float, float]
    classification_errors: list


def analyze(source: MeasurementSet, parameters: ProcessingParameters, *, automatic_window=True,
            absolute=False, progress=lambda value: None) -> AnalysisResult:
    """Process an owned copy. Legacy partial classifications carry explicit errors.

    automatic_window reproduces calc_to_int; False uses the supplied mean window.
    Physical axis units and legacy threshold semantics are intentionally unchanged.
    """
    measurements = deepcopy(source)
    window = preprocessing.prepare_interval(measurements, parameters.interval_min, parameters.interval_max)
    if automatic_window:
        parameters = replace(parameters, mean_window=window)
    limit = 9 if measurements.equipment == 'new' else 13
    for i in range(limit, limit + 3):
        key = str(i) + '_envelope' if measurements.equipment == 'new' else i
        if key not in measurements.signals:
            measurements.signals[key] = 0
    measurements.capture_original()
    calculate = calculate_new if measurements.equipment == 'new' else calculate_old
    # In the GUI, setting the recommended window emits change_mean_win before
    # the full pass. Retain its column insertion order for compatible tables.
    change_mean = preprocessing.change_mean_win if measurements.equipment == 'new' else preprocessing.change_mean_win_old
    change_mean(measurements, parameters)
    calculate(measurements, parameters, progress)
    channels = [channel.number for channel in measurements.channels]
    errors = []
    thresholds = interpretation.individual_thresholds(
        measurements.signals, channels, measurements.interval_min, measurements.interval_max,
        measurements.equipment, absolute, errors)
    interpretation.common_thresholds(measurements.signals, channels, measurements.interval_min,
                                     measurements.interval_max, *thresholds, absolute, errors)
    interpretation.match_defect(measurements.signals, [c for c in channels if c < limit])
    measurements.result = ProcessingResult(parameters, measurements.revision)
    return AnalysisResult(measurements, thresholds, errors)
