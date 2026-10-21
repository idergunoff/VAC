"""First calculation boundary: explicit state and parameter snapshots, no Qt."""
import logging

log = logging.getLogger(__name__)

import numpy as np
from scipy.optimize import curve_fit
from .domain.models import ProcessingParameters


def func1( x, a, b):
    """ функция затухания """
    return a * b ** x
    # return a*(x-b)**c



def get_nearest_value(iterable, value):
    return min(iterable, key=lambda x: abs(x - value))


def calc_max_int(tab1, tab2, i, maximum):
    target = tab2['max_h'][i] if tab2['max_h'][i] < maximum else maximum
    return tab1.index[tab1['depth'] == get_nearest_value(tab1['depth'], target)].tolist()[0]


def calculate_new(measurements, parameters, progress=lambda value: None):
    log.info('calculation_started equipment=new')
    signals, statistics = measurements.signals, measurements.statistics
    interval_min, interval_max = measurements.interval_min, measurements.interval_max
    coeff_norm = parameters.normalization
    coeff_func = parameters.function_shift
    coeff_dif_res = parameters.difference_scale
    mean_win = parameters.mean_window
    for i in range(1, 12):
        if (str(i) + '_envelope') in signals:
            if signals[str(i) + '_envelope'][1] != 0:
                interval_max = calc_max_int(signals, statistics, i, parameters.interval_max)
                max_min = signals[str(i) + '_envelope'].iloc[interval_min:interval_max].max() / \
                          signals[str(i) + '_envelope'].iloc[interval_min:interval_max].min()
                statistics['max_min'][i] = max_min  # расчет отношения макс к мин и заносим значение в таблицу статистики
                signals[str(i) + '_envelope-1'] = signals[str(i) + '_envelope'].shift(-1)  # сдвиг огибающ сигнала на 1
                signals[str(i) + '_mean'] = signals[str(i) + '_envelope'].rolling(mean_win, min_periods=1,
                                                                                          center=True).mean()  # усреднение
                signals[str(i) + '_norm'] = signals[str(i) + '_envelope'] / (signals[str(i) + '_mean'] + coeff_norm)
                # расчет цементограммы по усредняющей с коэффициентом
                signals[str(i) + '_norm'] = signals[str(i) + '_norm'].diff().rolling(50, min_periods=1, center=True).mean()    #todo эксперимент
                popt, pcov = curve_fit(func1, signals['depth'].iloc[interval_min:interval_max],  # расчет коэф для функции
                                       signals[str(i) + '_envelope'].iloc[interval_min:interval_max])
                statistics['coeff_A'][i] = popt[0]
                statistics['coeff_B'][i] = popt[1]
                signals[str(i) + '_func'] = func1(signals['depth'], popt[0], popt[1]) + coeff_func  # построение
                # фунции в интервале сигнала со сдвигом по Y на coeff_func
                """Расчет коэффициента затухания"""
                signals[str(i) + '_func_kdamp'] = signals[str(i) + '_func'][0]/signals[str(i) + '_func']
                i_kdamp = signals.index[signals[str(i) + '_func_kdamp'] == get_nearest_value(signals[str(i) +
                                                                                     '_func_kdamp'], np.exp(1))].tolist()[0]
                # all_stat['k_damp'][i] = 0.2029*np.log(np.exp(1)/all_signals['depth'][i_kdamp])+0.9127
                statistics['k_damp'][i] = np.exp(1) / signals['depth'][i_kdamp]
                # расчет
                # коэффициента затухания и пресчет в размерность коэффициента качества
                signals[str(i) + '_rel_ampl'] = (signals[str(i) + '_envelope-1'] - signals[str(i) + '_func']) /\
                                                    signals[str(i) + '_envelope-1']  # расчет цементограммы по функции
                signals[str(i) + '_rel_ampl'] = signals[str(i) + '_rel_ampl'].diff().rolling(50, min_periods=1, center=True).mean()  # todo эксперимент
                signals[str(i) + '_mean-1'] = signals[str(i) + '_mean'].shift(-1)  # сдвиг усредненного сигнала на 1
                signals[str(i) + '_diff_norm'] = (signals[str(i) + '_mean-1'] - signals[str(i) + '_mean']) / \
                                                     signals[str(i) + '_mean']  # нормированная производная
                signals[str(i) + '_diff_norm-1'] = signals[str(i) + '_diff_norm'].shift(-1)  # сдвиг
                signals[str(i) + '_diff_result'] = (signals[str(i) + '_diff_norm-1'] + signals[
                    str(i) + '_diff_norm']) * coeff_dif_res  # расчет цементограммы по производной (как в щелкуне)
                signals.at[interval_max, str(i) + '_diff_result'] = signals.at[interval_max-2, str(i) + '_diff_result']
                signals.at[interval_max-1, str(i) + '_diff_result'] = signals.at[interval_max-2, str(i) + '_diff_result']

                std_25 = signals[str(i) + '_envelope'].iloc[-430:].std()  #стандартное отклонение по последним 25 метрам
                for n, k in enumerate(signals[str(i) + '_mean'].iloc[interval_min:interval_max]):
                    if k <= std_25 + signals[str(i) + '_mean'].iloc[interval_min:interval_max].min():
                        useful_depth = signals['depth'][n]
                        statistics['useful_depth'][i] = useful_depth
                        break
                progress(i)

    measurements.interval_max = interval_max
    log.info('calculation_complete equipment=%s', measurements.equipment)

def calculate_old(measurements, parameters, progress=lambda value: None):
    log.info('calculation_started equipment=old')
    signals, statistics = measurements.signals, measurements.statistics
    interval_min, interval_max = measurements.interval_min, measurements.interval_max
    coeff_norm = parameters.normalization
    coeff_func = parameters.function_shift
    coeff_dif_res = parameters.difference_scale
    mean_win = parameters.mean_window
    for i in range(1, 16):
        if i in signals.columns:
            if signals[i][1] != 0:
                interval_max = calc_max_int(signals, statistics, i, parameters.interval_max)
                max_min = signals[i].iloc[interval_min:interval_max].max() / \
                          signals[i].iloc[interval_min:interval_max].min()
                statistics['max_min'][i] = max_min  # расчет отношения макс к мин и заносим значение в таблицу статистики
                signals[str(i) + '-1'] = signals[i].shift(-1)  # сдвиг огибающ сигнала на 1
                signals[str(i) + '_mean'] = signals[i].rolling(mean_win, min_periods=1, center=True).mean()
                signals[str(i) + '_norm'] = signals[i] / (signals[str(i) + '_mean'] + coeff_norm)
                signals[str(i) + '_norm'] = signals[str(i) + '_norm'].diff().rolling(50, min_periods=1, center=True).mean()    #todo эксперимент
                # расчет цементограммы по усредняющей с коэффициентом
                popt, pcov = curve_fit(func1, signals['depth'].iloc[interval_min:interval_max],
                                   signals[i].iloc[interval_min:interval_max])    # расчет коэф для функции
                statistics['coeff_A'][i] = popt[0]
                statistics['coeff_B'][i] = popt[1]
                signals[str(i) + '_func'] = func1(signals['depth'], popt[0], popt[1]) + coeff_func
                # построение фунции в интервале сигнала со сдвигом по Y на coeff_func
                """Расчет коэффициента затухания"""
                signals[str(i) + '_func_kdamp'] = signals[str(i) + '_func'][0] / \
                                                          signals[str(i) + '_func']
                i_kdamp = signals.index[
                    signals[str(i) + '_func_kdamp'] == get_nearest_value(signals[str(i) + '_func_kdamp'],
                                                                             np.exp(1))].tolist()[0]
                statistics['k_damp'][i] = 0.2029 * np.log(np.exp(1) / signals['depth'][i_kdamp]) + 0.9127
                statistics['k_damp'][i] = np.exp(1) / signals['depth'][i_kdamp]
                signals[str(i) + '_rel_ampl'] = (signals[str(i) + '-1'] -  # расчет цементограммы по функции
                                                         signals[str(i) + '_func']) / signals[str(i) + '-1']
                signals[str(i) + '_rel_ampl'] = signals[str(i) + '_rel_ampl'].diff().rolling(50, min_periods=1, center=True).mean()    #todo эксперимент
                signals[str(i) + '_mean-1'] = signals[str(i) + '_mean'].shift(-1)
                                                                                            # сдвиг усредненного сигнала
                signals[str(i) + '_diff_norm'] = (signals[str(i) + '_mean-1'] -  # нормированная производная
                                                      signals[str(i) + '_mean']) / signals[str(i) + '_mean']
                signals[str(i) + '_diff_norm-1'] = signals[str(i) + '_diff_norm'].shift(-1)  # сдвиг
                signals[str(i) + '_diff_result'] = (signals[str(i) + '_diff_norm-1'] + signals[
                    str(i) + '_diff_norm']) * coeff_dif_res  # расчет цементограммы по производной (как в щелкуне)
                signals.at[interval_max, str(i) + '_diff_result'] = signals.at[interval_max - 2, str(i) + '_diff_result']
                signals.at[interval_max - 1, str(i) + '_diff_result'] = signals.at[
                    interval_max - 2, str(i) + '_diff_result']

                std_25 = signals[i].iloc[-50:].std()  # стандартное отклонение по последним 25 метрам
                for n, k in enumerate(signals[str(i) + '_mean'].iloc[interval_min:interval_max]):
                    if k <= std_25 + signals[str(i) + '_mean'].iloc[interval_min:interval_max].min():
                        useful_depth = signals['depth'][n]
                        statistics['useful_depth'][i] = useful_depth
                        break
                progress(i)
    # расчет рекомендуемого окна усреднения по формуле из Щелкуна, установка значения в спинбок
    measurements.interval_max = interval_max
    log.info('calculation_complete equipment=%s', measurements.equipment)

class ClassificationError(ValueError):
    def __init__(self, quality):
        super().__init__("Invalid classification interval")
        self.quality = quality


def classify(input_list_1, defect1, defect2, absolute, intervals):
    """ Функция расчета интервалов дефектов. Принимает сигнал и 2 уровня девектов """
    input_list = input_list_1.tolist()
    l_list = len(input_list)
    quality = [0]*l_list

    try:
        if absolute:
            for n, i in enumerate(input_list):
                if defect1 <= i < defect2:
                    quality[n] = 1
                elif i >= defect2:
                    quality[n] = 2
        else:
            begin = []
            end = []
            if input_list[0] > defect1:
                begin.append(0)
            for n, i in enumerate(input_list):
                if 0 < n < (l_list - 1):
                    if i >= defect1:
                        if input_list[n - 1] < defect1:
                            begin.append(n)
                        if input_list[n + 1] < defect1:
                            end.append(n + 1)
            if input_list[l_list - 1] >= defect1:
                end.append(l_list)
            for i in range(len(begin)):
                if np.max(input_list[begin[i]:end[i]]) >= defect2:
                    quality[begin[i]:end[i]] = [2]*len(quality[begin[i]:end[i]])
                else:
                    quality[begin[i]:end[i]] = [1]*len(quality[begin[i]:end[i]])
        for i in intervals.good:
            quality[i[0]:i[1]] = [0]*len(quality[i[0]:i[1]])
        for i in intervals.partial:
            quality[i[0]:i[1]] = [1]*len(quality[i[0]:i[1]])
        for i in intervals.absent:
            quality[i[0]:i[1]] = [2]*len(quality[i[0]:i[1]])
        for i in intervals.undefined:
            quality[i[0]:i[1]] = [3]*len(quality[i[0]:i[1]])

        """ удаляем дефекты мощностью менее 1 метра """
        iq = quality[0]
        start_q = 0
        n_def = 0
        for n, q in enumerate(quality):
            if q != iq:
                n_def += 1
                if n - start_q < 17:
                    if n_def != 1:
                        quality[start_q:n] = [quality[start_q-1]] * len(quality[start_q:n])
                    else:
                        quality[start_q:n] = [q] * len(quality[start_q:n])
                iq = q
                start_q = n

    except IndexError:
        raise ClassificationError(quality) from None
    return quality

