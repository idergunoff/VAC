"""Sum calculations intentionally differ from the normal per-channel calculation."""
import numpy as np
from scipy.optimize import curve_fit
from ..calculation import func1, get_nearest_value, calc_max_int

class SignalLengthError(ValueError):
    pass


def sum_signals(measurements, parameters, channels, n_sum):
    """Average selected channels in supplied order, then fit using legacy sum formulas."""
    signals, statistics = measurements.signals, measurements.statistics
    interval_min = measurements.interval_min
    coeff_norm = parameters.normalization
    coeff_func = parameters.function_shift
    coeff_dif_res = parameters.difference_scale
    mean_win = parameters.mean_window
    signals[str(n_sum) + '_envelope'] = 0
    max_h = 0
    n_signals = 0
    for n_izm in channels:
        if max_h == 0:
            max_h = statistics['max_h'][n_izm]
        elif max_h != statistics['max_h'][n_izm]:
            raise SignalLengthError('Ошибка сложения! Нельзя сложить сигналы разной длины.')
        signals[str(n_sum) + '_envelope'] = signals[str(n_sum) + '_envelope'] + signals[str(n_izm) + '_envelope']
        n_signals += 1
    if n_signals == 0:
        return False
    signals[str(n_sum) + '_envelope'] = signals[str(n_sum) + '_envelope'] / n_signals
    statistics['max_h'][n_sum] = max_h
    int_max = calc_max_int(signals, statistics, n_sum, parameters.interval_max)
    max_min = signals[str(n_sum) + '_envelope'].iloc[interval_min:int_max].max() / \
              signals[str(n_sum) + '_envelope'].iloc[interval_min:int_max].min()
    statistics['max_min'][n_sum] = max_min  # расчет отношения макс к мин и заносим значение в таблицу статистики
    signals[str(n_sum) + '_envelope-1'] = signals[str(n_sum) + '_envelope'].shift(-1)  # сдвиг огибающ сигнала на 1
    signals[str(n_sum) + '_mean'] = signals[str(n_sum) + '_envelope'].rolling(mean_win, min_periods=1,
                                                                              center=True).mean()  # усреднение
    signals[str(n_sum) + '_norm'] = signals[str(n_sum) + '_envelope'] / (signals[str(n_sum) + '_mean'] + coeff_norm)
    # расчет цементограммы по усредняющей с коэффициентом
    popt, pcov = curve_fit(func1, signals['depth'].iloc[interval_min:int_max],  # расчет коэф для функции
                           signals[str(n_sum) + '_envelope'].iloc[interval_min:int_max])
    statistics['coeff_A'][n_sum] = popt[0]
    statistics['coeff_B'][n_sum] = popt[1]
    signals[str(n_sum) + '_func'] = func1(signals['depth'], popt[0], popt[1]) + coeff_func  # построение
    # фунции в интервале сигнала со сдвигом по Y на coeff_func

    """Расчет коэффициента затухания"""
    signals[str(n_sum) + '_func_kdamp'] = signals[str(n_sum) + '_func'][0]/signals[str(n_sum) + '_func']
    i_kdamp = signals.index[signals[str(n_sum) + '_func_kdamp'] == get_nearest_value(signals[str(n_sum)
                                                                         +'_func_kdamp'], np.exp(1))].tolist()[0]
    # all_stat['k_damp'][n_sum] = 0.2029*np.log(np.exp(1)/all_signals['depth'][i_kdamp])+0.9127
    statistics['k_damp'][n_sum] = np.exp(1) / signals['depth'][i_kdamp]

    signals[str(n_sum) + '_rel_ampl'] = (signals[str(n_sum) + '_envelope-1'] - signals[str(n_sum) + '_func']) / \
                                        signals[str(n_sum) + '_envelope-1']  # расчет цементограммы по функции
    signals[str(n_sum) + '_mean-1'] = signals[str(n_sum) + '_mean'].shift(-1)  # сдвиг усредненного сигнала на 1
    signals[str(n_sum) + '_diff_norm'] = (signals[str(n_sum) + '_mean-1'] - signals[str(n_sum) + '_mean']) / \
                                         signals[str(n_sum) + '_mean']  # нормированная производная
    signals[str(n_sum) + '_diff_norm-1'] = signals[str(n_sum) + '_diff_norm'].shift(-1)  # сдвиг
    signals[str(n_sum) + '_diff_result'] = (signals[str(n_sum) + '_diff_norm-1'] + signals[
        str(n_sum) + '_diff_norm']) * coeff_dif_res  # расчет цементограммы по производной (как в щелкуне)
    signals.at[int_max, str(n_sum) + '_diff_result'] = signals.at[int_max - 2, str(n_sum) + '_diff_result']
    signals.at[int_max - 1, str(n_sum) + '_diff_result'] = signals.at[int_max - 2, str(n_sum) + '_diff_result']
    std_25 = signals[str(n_sum) + '_envelope'].iloc[-430:].std()  # стандартное отклонение по последним 25 метрам
    for n, k in enumerate(signals[str(n_sum) + '_mean'].iloc[interval_min:int_max]):
        if k <= std_25 + signals[str(n_sum) + '_mean'].iloc[interval_min:int_max].min():
            useful_depth = signals['depth'][n]
            statistics['useful_depth'][n_sum] = useful_depth
            break
    return True


def sum_signals_old(measurements, parameters, channels, n_sum):
    """Average selected channels in supplied order, then fit using legacy sum formulas."""
    signals, statistics = measurements.signals, measurements.statistics
    interval_min = measurements.interval_min
    coeff_norm = parameters.normalization
    coeff_func = parameters.function_shift
    coeff_dif_res = parameters.difference_scale
    mean_win = parameters.mean_window
    signals[n_sum] = 0
    max_h = 0
    n_signals = 0
    for n_izm in channels:
        if max_h == 0:
            max_h = statistics['max_h'][n_izm]
        elif max_h != statistics['max_h'][n_izm]:
            raise SignalLengthError('Ошибка сложения! Нельзя сложить сигналы разной длины.')
        signals[n_sum] = signals[n_sum] + signals[n_izm]
        n_signals += 1
    if n_signals == 0:
        return False
    signals[n_sum] = signals[n_sum] / n_signals
    statistics['max_h'][n_sum] = max_h
    int_max_old = calc_max_int(signals, statistics, n_sum, parameters.interval_max)
    max_min = signals[n_sum].iloc[interval_min:int_max_old].max() / \
              signals[n_sum].iloc[interval_min:int_max_old].min()
    statistics['max_min'][n_sum] = max_min  # расчет отношения макс к мин и заносим значение в таблицу статистики
    signals[str(n_sum) + '-1'] = signals[n_sum].shift(-1)  # сдвиг огибающ сигнала на 1
    signals[str(n_sum) + '_mean'] = signals[n_sum].rolling(mean_win, min_periods=1, center=True).mean()
    signals[str(n_sum) + '_norm'] = signals[n_sum] / (signals[str(n_sum) + '_mean'] + coeff_norm)
    # расчет цементограммы по усредняющей с коэффициентом
    popt, pcov = curve_fit(func1, signals['depth'].iloc[interval_min:int_max_old],  # расчет коэф для функции
                           signals[n_sum].iloc[interval_min:int_max_old])
    statistics['coeff_A'][n_sum] = popt[0]
    statistics['coeff_B'][n_sum] = popt[1]
    signals[str(n_sum) + '_func'] = func1(signals['depth'], popt[0], popt[1]) + coeff_func  # построение
    # фунции в интервале сигнала со сдвигом по Y на coeff_func

    """Расчет коэффициента затухания"""
    signals[str(n_sum) + '_func_kdamp'] = signals[str(n_sum) + '_func'][0] / signals[
        str(n_sum) + '_func']
    i_kdamp = signals.index[
        signals[str(n_sum) + '_func_kdamp'] == get_nearest_value(signals[str(n_sum) + '_func_kdamp'],
                                                                     np.exp(1))].tolist()[0]
    # all_stat_old['k_damp'][n_sum] = 0.2029 * np.log(np.exp(1) / all_signals_old['depth'][i_kdamp]) + 0.9127
    statistics['k_damp'][n_sum] = np.exp(1) / signals['depth'][i_kdamp]

    signals[str(n_sum) + '_rel_ampl'] = (signals[str(n_sum) + '-1'] - signals[str(n_sum) + '_func']) / \
                                            signals[str(n_sum) + '-1']  # расчет цементограммы по функции
    signals[str(n_sum) + '_mean-1'] = signals[str(n_sum) + '_mean'].shift(-1)  # сдвиг усредненного сигнала
    signals[str(n_sum) + '_diff_norm'] = (signals[str(n_sum) + '_mean-1'] -  # нормированная производная
                                              signals[str(n_sum) + '_mean']) / signals[str(n_sum) + '_mean']
    signals[str(n_sum) + '_diff_norm-1'] = signals[str(n_sum) + '_diff_norm'].shift(-1)  # сдвиг
    signals[str(n_sum) + '_diff_result'] = (signals[str(n_sum) + '_diff_norm-1'] + signals[
        str(n_sum) + '_diff_norm']) * coeff_dif_res  # расчет цементограммы по производной (как в щелкуне)
    signals.at[int_max_old, str(n_sum) + '_diff_result'] = signals.at[
        int_max_old - 2, str(n_sum) + '_diff_result']
    signals.at[int_max_old - 1, str(n_sum) + '_diff_result'] = signals.at[
        int_max_old - 2, str(n_sum) + '_diff_result']

    std_25 = signals[n_sum].iloc[-50:].std()  # стандартное отклонение по последним 25 метрам
    for n, k in enumerate(signals[str(n_sum) + '_mean'].iloc[interval_min:int_max_old]):
        if k <= std_25 + signals[str(n_sum) + '_mean'].iloc[interval_min:int_max_old].min():
            useful_depth = signals['depth'][n]
            statistics['useful_depth'][n_sum] = useful_depth
            break
    return True
