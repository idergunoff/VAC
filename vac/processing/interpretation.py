"""Classification thresholds and cementogram data, without plotting or exports."""
import numpy as np
import pandas as pd
from ..calculation import classify, ClassificationError
from ..domain.models import QualityIntervals

def classify_channel(signal, first, second, absolute, errors=None):
    try:
        return classify(signal, first, second, absolute, QualityIntervals())
    except ClassificationError as exc:
        if errors is None:
            raise
        errors.append(exc)
        return exc.quality


def individual_thresholds(device, c_sig, int_min_self, int_max_self, equipment, absolute, errors=None):
    list_def1, list_def2 = [], []
    for n_sig in c_sig:
        min_value = device[str(n_sig) + '_diff_result'].iloc[int_min_self:int_max_self].min()
        max_value = device[str(n_sig) + '_diff_result'].iloc[int_min_self:int_max_self].max()
        def1 = min_value + (max_value - min_value) / 3
        def2 = min_value + (2 * (max_value - min_value) / 3)
        device[str(n_sig) + '_quality'] = [0]*int_min_self + \
            classify_channel(device[str(n_sig) + '_diff_result'].iloc[int_min_self:int_max_self], def1, def2, absolute, errors) + \
                                          [0]*(len(device['depth']) - int_max_self)
        if equipment == 'new':
            if n_sig not in [9, 10, 11]:
                list_def1.append(def1)
                list_def2.append(def2)
        else:
            if n_sig not in [13, 14, 15]:
                list_def1.append(def1)
                list_def2.append(def2)
    mean_def1 = np.mean(list_def1)
    mean_def2 = np.mean(list_def2)
    return mean_def1, mean_def2


def common_thresholds(device, c_sig, int_min_self, int_max_self, mean_def1, mean_def2, absolute, errors=None):
    for n_sig in c_sig:
        device[str(n_sig) + '_qual_mean'] = [0]*int_min_self + \
            classify_channel(device[str(n_sig) + '_diff_result'].iloc[int_min_self:int_max_self], mean_def1, mean_def2, absolute, errors) + \
                                          [0]*(len(device['depth']) - int_max_self)


def match_defect(device, list_sig):
    list_match_uniq = []
    list_match_mean = []
    for row in range(len(device['depth'])):
        sel_mean = 0
        sel_uniq = 0
        for n_izm in list_sig:
            if device[str(n_izm) + '_quality'][row] > 0:
                sel_uniq += 1
            if device[str(n_izm) + '_qual_mean'][row] > 0:
                sel_mean += 1
        list_match_uniq.append((sel_uniq/len(list_sig))*10-5)
        list_match_mean.append((sel_mean / len(list_sig))*10-5)
    device['match_def_uniq'] = list_match_uniq
    device['match_def_mean'] = list_match_mean


def damping_class(value):
    if value < 0.3233:
        return 'red'
    if 0.3233 <= value < 0.6806:
        return 'yellow'
    if value >= 0.6806:
        return 'green'


def prepare_cement(measurements, signal, maximum=None):
    """Positional [max(86,min):max) slice; output has an independent zero-based index."""
    prefix = str(signal)
    maximum = measurements.interval_max if maximum is None else maximum
    interval = slice(max(86, measurements.interval_min), maximum)
    signals = measurements.signals
    table = pd.DataFrame(signals['depth'].iloc[interval])
    table['first_sig'] = signals[prefix + '_diff_result'].iloc[interval]
    table['func'] = signals[prefix + '_func'].iloc[interval]
    table = table.reset_index(drop=True)
    table['corr_sig'] = table['first_sig'].copy()
    table['corr_coeff'] = 1
    return table


def frequency(equipment, number):
    if number in ([1, 2, 5, 6] if equipment == 'new' else [1, 2, 5, 6, 9, 10]):
        return '5'
    if number in ([3, 4, 7, 8] if equipment == 'new' else [3, 4, 7, 8, 11, 12]):
        return '2'
    return ''
