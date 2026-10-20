"""Legacy preprocessing; functions modify only the explicitly supplied working tables."""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ..calculation import func1, get_nearest_value

def envelope(depth, signal, window, bounds_error=True):
    """Linear interpolation of local maxima, then centred rolling mean."""
    n_max = ((np.diff(np.sign(np.diff(signal))) < 0).nonzero()[0] + 1).tolist()
    n_max.insert(0, 0)
    n_max.append(int(len(signal)) - 1)
    curve = interp1d(depth.loc[n_max], signal.loc[n_max], kind='linear', bounds_error=bounds_error)
    return pd.Series(curve(depth), index=depth.index).rolling(window, min_periods=1, center=True).mean()


def twf_depth(length):
    return np.arange(0, length * 0.515, 0.05839)


def interpolate_twf(signal, depth, bounds_error=True):
    return interp1d(np.linspace(0, len(signal) * 0.515, len(signal), endpoint=True),
                    signal, kind='linear', bounds_error=bounds_error)(depth)


def prepare_interval(measurements, minimum, maximum):
    depth = measurements.signals['depth']
    measurements.interval_min = depth.index[depth == get_nearest_value(depth, minimum)].tolist()[0]
    measurements.interval_max = depth.index[depth == get_nearest_value(depth, maximum)].tolist()[0]
    width = measurements.interval_max - measurements.interval_min
    # QSpinBox's legacy range is part of the headless contract too.
    return max(1, min(1000, int(width / (1 + 3.322 * np.log(width)) / 2.302585)))


def change_mean_win(measurements, parameters):
    signals, statistics = measurements.signals, measurements.statistics
    interval_max = measurements.interval_max
    mean_win = parameters.mean_window
    coeff_norm = parameters.normalization
    coeff_dif_res = parameters.difference_scale
    for i in range(1, 12):
        if (str(i) + '_envelope') in signals:
            if signals[str(i) + '_envelope'][1] != 0:
                signals[str(i) + '_mean'] = signals[str(i) + '_envelope'].rolling(mean_win, min_periods=1,
                                                                                          center=True).mean()
                signals[str(i) + '_norm'] = signals[str(i) + '_envelope'] / (signals[str(i) + '_mean'] + coeff_norm)
                signals[str(i) + '_norm'] = signals[str(i) + '_norm'].diff().rolling(50, min_periods=1, center=True).mean()  # todo эксперимент
                signals[str(i) + '_mean-1'] = signals[str(i) + '_mean'].shift(-1)
                signals[str(i) + '_diff_norm'] = (signals[str(i) + '_mean-1'] - signals[str(i) + '_mean']) / \
                                                     signals[str(i) + '_mean']
                signals[str(i) + '_diff_norm-1'] = signals[str(i) + '_diff_norm'].shift(-1)
                signals[str(i) + '_diff_result'] = (signals[str(i) + '_diff_norm-1'] + signals[
                    str(i) + '_diff_norm']) * coeff_dif_res
                signals.at[interval_max, str(i) + '_diff_result'] = signals.at[interval_max - 2, str(i) + '_diff_result']
                signals.at[interval_max - 1, str(i) + '_diff_result'] = signals.at[
                    interval_max - 2, str(i) + '_diff_result']


def change_mean_win_old(measurements, parameters):
    signals, statistics = measurements.signals, measurements.statistics
    interval_max = measurements.interval_max
    mean_win = parameters.mean_window
    coeff_norm = parameters.normalization
    coeff_dif_res = parameters.difference_scale
    for i in range(1, 16):
        if i in signals.columns:
            if signals[i][1] != 0:
                signals[str(i) + '_mean'] = signals[i].rolling(mean_win, min_periods=1, center=True).mean()
                signals[str(i) + '_norm'] = signals[i] / (signals[str(i) + '_mean'] + coeff_norm)
                signals[str(i) + '_norm'] = signals[str(i) + '_norm'].diff().rolling(50, min_periods=1, center=True).mean()  # todo эксперимент
                signals[str(i) + '_mean-1'] = signals[str(i) + '_mean'].shift(-1)  # сдвиг усредненного сигнала
                signals[str(i) + '_diff_norm'] = (signals[str(i) + '_mean-1'] -  # нормированная производная
                                                          signals[str(i) + '_mean']) / signals[str(i) + '_mean']
                signals[str(i) + '_diff_norm-1'] = signals[str(i) + '_diff_norm'].shift(-1)  # сдвиг
                signals[str(i) + '_diff_result'] = (signals[str(i) + '_diff_norm-1'] + signals[
                    str(i) + '_diff_norm']) * coeff_dif_res  # расчет цементограммы по производной (как в щелкуне)
                signals.at[interval_max, str(i) + '_diff_result'] = signals.at[
                    interval_max - 2, str(i) + '_diff_result']
                signals.at[interval_max - 1, str(i) + '_diff_result'] = signals.at[
                    interval_max - 2, str(i) + '_diff_result']



def change_func(measurements, parameters, kA, kB):
    signals, statistics = measurements.signals, measurements.statistics
    interval_max = measurements.interval_max
    coeff_func = parameters.function_shift
    for i in range(1, 12):
        if (str(i) + '_envelope') in signals:
            if signals[str(i) + '_envelope'][1] != 0:
                signals[str(i) + '_func'] = func1(signals['depth'], statistics['coeff_A'][i]+kA, statistics['coeff_B'][i]+kB) + coeff_func
                signals[str(i) + '_rel_ampl'] = (signals[str(i) + '_envelope-1'] - signals[str(i) + '_func']) / \
                                                    signals[str(i) + '_envelope-1']
                signals[str(i) + '_rel_ampl'] = signals[str(i) + '_rel_ampl'].diff().rolling(50, min_periods=1, center=True).mean()  # todo эксперимент


def change_func_old(measurements, parameters, kA, kB):
    signals, statistics = measurements.signals, measurements.statistics
    interval_max = measurements.interval_max
    coeff_func = parameters.function_shift
    for i in range(1, 16):
        if i in signals.columns:
            if signals[i][1] != 0:
                signals[str(i) + '_func'] = func1(signals['depth'], statistics['coeff_A'][i]+kA, statistics['coeff_B'][i]+kB) + coeff_func
                signals[str(i) + '_rel_ampl'] = (signals[str(i) + '-1'] - signals[str(i) + '_func']) / \
                                                        signals[str(i) + '-1']  # расчет цементограммы по функции
                signals[str(i) + '_rel_ampl'] = signals[str(i) + '_rel_ampl'].diff().rolling(50, min_periods=1, center=True).mean()  # todo эксперимент



def extend_lvm_axis(measurements, signal):
    signals = measurements.signals
    if 'depth' in signals:
        if signals['depth'].max() >= signal[1].max():
            return False
        extension = pd.DataFrame(columns=signals.columns, index=range(len(signals['depth']), len(signal[1])))
        measurements.signals = pd.concat([signals, extension])
    measurements.signals['depth'] = signal[1]
    return True


def extend_twf_axis(measurements, length):
    maximum = length * 0.515
    signals = measurements.signals
    if 'depth' in signals:
        if signals['depth'].max() + 0.515 >= maximum:
            return False
        extension = pd.DataFrame(columns=signals.columns, index=range(len(signals['depth']), int(maximum / 0.05839) + 1))
        measurements.signals = pd.concat([signals, extension])
    measurements.signals['depth'] = twf_depth(length)
    return True


def scale_single_twf(measurements, channel):
    if measurements.scale == 1:
        measurements.scale = measurements.signals[channel].max() / 5
    measurements.signals[channel] = measurements.signals[channel] / measurements.scale
