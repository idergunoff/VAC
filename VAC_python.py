import os

import pandas as pd

import numpy as np

import struct

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QFileDialog

import pyqtgraph as pg

Form, Window = uic.loadUiType("VAC_dialog.ui")

app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)
window.show()

all_signals = pd.DataFrame()
all_signals_old = pd.DataFrame()
columns = ['name', 'useful_depth', 'max_min', 'coeff_A', 'coeff_B', 'k_damp', 'gl_kd']
all_stat = pd.DataFrame(index=list(range(1, 12)), columns=columns)
all_stat_old = pd.DataFrame(index=list(range(1, 16)), columns=columns)
form.graphicsView.showGrid(x=True, y=True)  # грид-сетка


def open_dir():
    """ функция выбора директории с файлами измерений в формате *.lvm """
    global all_signals
    form.progressBar.reset()
    all_signals = pd.DataFrame()  # при открытии новой директории таблица all_signals очищается
    dir_name = QFileDialog.getExistingDirectory()  # окно выбора папки
    all_files = list()
    n = 1
    for dir_i in os.walk(dir_name):  # перебор директорий
        for file in sorted(dir_i[-1], reverse=True):  # перебор файлов в директории с обратной сортировкой
            if file.endswith('_5150m_c.lvm'):  # выбраем и заносим в список только файлы с расширением *.lvm
                all_files.append(file)
                signal = pd.read_table(os.path.join(dir_i[0], file), delimiter='\t', header=None)  # считываем таблицу
                all_signals['depth'] = signal[1]
                if n == 9:  # если записаны 8 файлов, последний файл записываем в шум и создаем колонку с глубиной
                    all_signals['noise'] = np.abs(signal[2])
                else:
                    all_signals[n] = np.abs(signal[2])
                    n_max = ((np.diff(np.sign(np.diff(all_signals[n]))) < 0).nonzero()[
                                 0] + 1).tolist()  # находим максимумы
                    n_max.insert(0, 0)  # добавляем начальную и конечную точки                              # сигнала
                    # n_max = n_max[::3]
                    n_max.append(int(len(all_signals[n])) - 1)
                    max_x = all_signals.loc[n_max, 'depth']  # выбираем x и y максимумов
                    max_y = all_signals.loc[n_max, n]
                    envelope = interp1d(max_x, max_y,
                                        kind='linear')  # выполняем интерполяцию по максимуму, получая огибающую
                    all_signals[str(n) + '_envelope'] = envelope(all_signals['depth'])
                    all_signals[str(n) + '_envelope'] = all_signals[str(n) + '_envelope'].rolling(80, min_periods=1,
                                                                                                  center=True).mean()  # усреднение
                    all_stat['name'][n] = file  # записываем названия файлов в таблицу статистики
                n += 1
    print(all_signals)

    form.label_direct.setText("<b>" + dir_name + ":</b>")
    form.checkBox_signal1.setText(all_files[0][0:20])  # присваиваем чекбоксам названия файлов
    form.checkBox_signal2.setText(all_files[1][0:20])
    form.checkBox_signal3.setText(all_files[2][0:20])
    form.checkBox_signal4.setText(all_files[3][0:20])
    form.checkBox_signal5.setText(all_files[4][0:20])  # присваиваем чекбоксам названия файлов
    form.checkBox_signal6.setText(all_files[5][0:20])
    form.checkBox_signal7.setText(all_files[6][0:20])
    form.checkBox_signal8.setText(all_files[7][0:20])
    form.checkBox_noise.setText(all_files[8][0:20])

    form.checkBox_signal1.setEnabled(True)
    form.checkBox_signal2.setEnabled(True)
    form.checkBox_signal3.setEnabled(True)
    form.checkBox_signal4.setEnabled(True)
    form.checkBox_signal5.setEnabled(True)
    form.checkBox_signal6.setEnabled(True)
    form.checkBox_signal7.setEnabled(True)
    form.checkBox_signal8.setEnabled(True)
    form.checkBox_noise.setEnabled(True)
    form.pushButton_sum1.setEnabled(True)
    form.pushButton_sum2.setEnabled(True)
    form.pushButton_sum3.setEnabled(True)

    form.doubleSpinBox_int_max.setMaximum(all_signals['depth'].max())  # устанавливаем в спинбоксы интервала сигнала
    form.doubleSpinBox_int_min.setMaximum(all_signals['depth'].max())  # максимально допустимые значения из файла
    form.doubleSpinBox_int_max.setValue(all_signals['depth'].max())  # устанавливаем в спинбоксы значения мин и макс
    form.doubleSpinBox_int_min.setValue(all_signals['depth'].min())

    for i in range(9, 12):      # пустые столбцы для суммы сигналов
        all_signals[str(i) + '_envelope'] = 0

    calc_to_int()  # запускаем функцию обработки сигнала в интервале


def open_dir_old():
    """ функция выбора директории с файлами измерений в формате *.TWF """
    global all_signals_old
    form.progressBar_2.reset()
    all_signals_old = pd.DataFrame()  # при открытии новой директории таблица all_signals_old очищается
    dir_name = QFileDialog.getExistingDirectory()  # окно выбора папки
    all_files = list()
    n = 1
    for file in os.listdir(dir_name):  # перебор файлов в папке
        if file.endswith('.TWF'):  # выбраем и заносим в список только файлы с расширением *.TWF
            all_files.append(file)
            f = open(os.path.join(dir_name, file), 'rb')    # открываем бинарный файл для чтения
            signal_b = f.read()[130:]                       # считываем байты кроме первых 130
            len_signal = int(len(signal_b) / 4)             # количество чисел
            format_b = '<' + str(len_signal) + 'i'          # формат файла для чтения
            signal = struct.unpack(format_b, signal_b)      # пересчет байтов в кортеж сигнала
            f.close()                                       # закрываем файл
            if n == 1:  # первый файл записываем в шум и создаем колонку с глубиной
                all_signals_old['noise'] = signal
                all_signals_old['depth'] = np.arange(0, len_signal * 0.515, 0.515)  # шкала глубин через 0.515 метров
            else:
                all_signals_old[n-1] = signal
                all_stat_old['name'][n-1] = file  # записываем названия файлов в таблицу статистики
            n += 1
            if n == 14:     # если считано 13 файлов выходим из цикла
                break
    print(all_signals_old)
    col_max = all_signals_old.columns[all_signals_old.loc[0] == all_signals_old.loc[0].max()].tolist()[0]
    k_oldtonew = all_signals_old[col_max].max()/5
    all_signals_old['noise'] = all_signals_old['noise']/k_oldtonew
    for i in range(1, len(all_signals_old.loc[0])-1):
        all_signals_old[i] = all_signals_old[i]/k_oldtonew
    print(all_signals_old)

    form.label_direct_old.setText("<b>" + dir_name + ":</b>")
    form.checkBox_signal_old_1.setText(all_files[1][0:7])  # присваиваем чекбоксам названия файлов - первые 7 символов
    form.checkBox_signal_old_2.setText(all_files[2][0:7])
    form.checkBox_signal_old_3.setText(all_files[3][0:7])
    form.checkBox_signal_old_4.setText(all_files[4][0:7])
    form.checkBox_signal_old_5.setText(all_files[5][0:7])  # присваиваем чекбоксам названия файлов
    form.checkBox_signal_old_6.setText(all_files[6][0:7])
    form.checkBox_signal_old_7.setText(all_files[7][0:7])
    form.checkBox_signal_old_8.setText(all_files[8][0:7])
    form.checkBox_signal_old_9.setText(all_files[9][0:7])  # присваиваем чекбоксам названия файлов
    form.checkBox_signal_old_10.setText(all_files[10][0:7])
    form.checkBox_signal_old_11.setText(all_files[11][0:7])
    form.checkBox_signal_old_12.setText(all_files[12][0:7])
    form.checkBox_noise_old.setText(all_files[0][0:7])

    form.checkBox_signal_old_1.setEnabled(True)
    form.checkBox_signal_old_2.setEnabled(True)
    form.checkBox_signal_old_3.setEnabled(True)
    form.checkBox_signal_old_4.setEnabled(True)
    form.checkBox_signal_old_5.setEnabled(True)
    form.checkBox_signal_old_6.setEnabled(True)
    form.checkBox_signal_old_7.setEnabled(True)
    form.checkBox_signal_old_8.setEnabled(True)
    form.checkBox_signal_old_9.setEnabled(True)
    form.checkBox_signal_old_10.setEnabled(True)
    form.checkBox_signal_old_11.setEnabled(True)
    form.checkBox_signal_old_12.setEnabled(True)
    form.checkBox_noise_old.setEnabled(True)
    form.pushButton_sum1_old.setEnabled(True)
    form.pushButton_sum2_old.setEnabled(True)
    form.pushButton_sum3_old.setEnabled(True)

    form.doubleSpinBox_int_max.setMaximum(all_signals_old['depth'].max())  # устанавливаем в спинбоксы интервала сигнала
    form.doubleSpinBox_int_min.setMaximum(all_signals_old['depth'].max())  # максимально допустимые значения из файла
    form.doubleSpinBox_int_max.setValue(all_signals_old['depth'].max())  # устанавливаем в спинбоксы значения мин и макс
    form.doubleSpinBox_int_min.setValue(all_signals_old['depth'].min())

    for i in range(13, 16):      # пустые столбцы для суммы сигналов
        all_signals_old[i] = 0

    calc_to_int_old()  # запускаем функцию обработки сигнала в интервале


def calc_to_int():
    global int_min, int_max

    coeff_norm = form.doubleSpinBox_coeff_norm.value()  # считываем значения коэффициентов из спинбоксов
    coeff_func = form.doubleSpinBox_coeff_func.value()
    coeff_dif_res = form.spinBox_coeff_dif_res.value()
    mean_win = form.spinBox_mean_win.value()
    level_defect = form.doubleSpinBox_level_defect.value()
    all_signals['defect'] = level_defect  # записываем таблицу уровень дефектов для построения

    int_min = all_signals.index[all_signals['depth'] ==  # определяем интервалы индексов min - max
                                get_nearest_value(all_signals['depth'], form.doubleSpinBox_int_min.value())].tolist()[0]
    int_max = all_signals.index[all_signals['depth'] ==
                                get_nearest_value(all_signals['depth'], form.doubleSpinBox_int_max.value())].tolist()[0]

    for i in range(1, 12):
        if all_signals[str(i) + '_envelope'][1] != 0:
            max_min = all_signals[str(i) + '_envelope'].iloc[int_min:int_max].max() / \
                      all_signals[str(i) + '_envelope'].iloc[int_min:int_max].min()
            all_stat['max_min'][i] = max_min  # расчет отношения макс к мин и заносим значение в таблицу статистики
            all_signals[str(i) + '_envelope-1'] = all_signals[str(i) + '_envelope'].shift(-1)  # сдвиг огибающ сигнала на 1
            all_signals[str(i) + '_mean'] = all_signals[str(i) + '_envelope'].rolling(mean_win, min_periods=1,
                                                                                      center=True).mean()  # усреднение
            all_signals[str(i) + '_norm'] = all_signals[str(i) + '_envelope'] / (all_signals[str(i) + '_mean'] + coeff_norm)
            # расчет цементограммы по усредняющей с коэффициентом
            popt, pcov = curve_fit(func1, all_signals['depth'].iloc[int_min:int_max],  # расчет коэф для функции
                                   all_signals[str(i) + '_envelope'].iloc[int_min:int_max])
            all_stat['coeff_A'][i] = popt[0]
            all_stat['coeff_B'][i] = popt[1]
            all_signals[str(i) + '_func'] = func1(all_signals['depth'], popt[0], popt[1]) + coeff_func  # построение
            # фунции в интервале сигнала со сдвигом по Y на coeff_func
            """Расчет коэффициента затухания"""
            all_signals[str(i) + '_func_kdamp'] = all_signals[str(i) + '_func'][0]/all_signals[str(i) + '_func']
            i_kdamp = all_signals.index[all_signals[str(i) + '_func_kdamp'] == get_nearest_value(all_signals[str(i) +
                                                                                 '_func_kdamp'], np.exp(1))].tolist()[0]
            all_stat['k_damp'][i] = np.exp(1)/all_signals['depth'][i_kdamp]


            all_signals[str(i) + '_rel_ampl'] = (all_signals[str(i) + '_envelope-1'] - all_signals[str(i) + '_func']) / \
                                                all_signals[str(i) + '_envelope-1']  # расчет цементограммы по функции
            all_signals[str(i) + '_mean-1'] = all_signals[str(i) + '_mean'].shift(-1)  # сдвиг усредненного сигнала на 1
            all_signals[str(i) + '_diff_norm'] = (all_signals[str(i) + '_mean-1'] - all_signals[str(i) + '_mean']) / \
                                                 all_signals[str(i) + '_mean']  # нормированная производная
            all_signals[str(i) + '_diff_norm-1'] = all_signals[str(i) + '_diff_norm'].shift(-1)  # сдвиг
            all_signals[str(i) + '_diff_result'] = (all_signals[str(i) + '_diff_norm-1'] + all_signals[
                str(i) + '_diff_norm']) * coeff_dif_res  # расчет цементограммы по производной (как в щелкуне)

            std_25 = all_signals[str(i) + '_envelope'].iloc[-430:].std()    # стандартное отклонение по последним 25 метрам
            for n, k in enumerate(all_signals[str(i) + '_mean'].iloc[int_min:int_max]):
                if k <= std_25 + all_signals[str(i) + '_mean'].iloc[int_min:int_max].min():
                    useful_depth = all_signals['depth'][n]
                    all_stat['useful_depth'][i] = useful_depth
                    break
            form.progressBar.setValue(i)
    # расчет рекомендуемого окна усреднения по формуле из Щелкуна, установка значения в спинбок
    auto_mean_win = (int_max - int_min) / (1 + 3.322 * np.log(int_max - int_min)) / 2.302585
    form.spinBox_mean_win.setValue(auto_mean_win)

    print(all_stat)

    form.checkBox_signal1.setToolTip('Коэффициент затухания - '+str(round(all_stat['k_damp'][1], 3))+
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][1], 2))+
                                     ' м\nМакс/Мин - '+str(round(all_stat['max_min'][1], 3)))
    form.checkBox_signal2.setToolTip('Коэффициент затухания - '+str(round(all_stat['k_damp'][2], 3))+
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][2], 2))+
                                     ' м\nМакс/Мин - '+str(round(all_stat['max_min'][2], 3)))
    form.checkBox_signal3.setToolTip('Коэффициент затухания - '+str(round(all_stat['k_damp'][3], 3))+
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][3], 2))+
                                     ' м\nМакс/Мин - '+str(round(all_stat['max_min'][3], 3)))
    form.checkBox_signal4.setToolTip('Коэффициент затухания - '+str(round(all_stat['k_damp'][4], 3))+
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][4], 2))+
                                     ' м\nМакс/Мин - '+str(round(all_stat['max_min'][4], 3)))
    form.checkBox_signal5.setToolTip('Коэффициент затухания - '+str(round(all_stat['k_damp'][5], 3))+
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][5], 2))+
                                     ' м\nМакс/Мин - '+str(round(all_stat['max_min'][5], 3)))
    form.checkBox_signal6.setToolTip('Коэффициент затухания - '+str(round(all_stat['k_damp'][6], 3))+
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][6], 2))+
                                     ' м\nМакс/Мин - '+str(round(all_stat['max_min'][6], 3)))
    form.checkBox_signal7.setToolTip('Коэффициент затухания - '+str(round(all_stat['k_damp'][7], 3))+
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][7], 2))+
                                     ' м\nМакс/Мин - '+str(round(all_stat['max_min'][7], 3)))
    form.checkBox_signal8.setToolTip('Коэффициент затухания - '+str(round(all_stat['k_damp'][8], 3))+
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][8], 2))+
                                     ' м\nМакс/Мин - '+str(round(all_stat['max_min'][8], 3)))

    all_signals.to_csv('result.txt', sep='\t')
    choice_signal()


def calc_to_int_old():
    global int_min_old, int_max_old

    coeff_norm = form.doubleSpinBox_coeff_norm_old.value()  # считываем значения коэффициентов из спинбоксов
    coeff_func = form.doubleSpinBox_coeff_func_old.value()
    coeff_dif_res = form.spinBox_coeff_dif_res_old.value()
    mean_win = form.spinBox_mean_win_old.value()


    int_min_old = all_signals_old.index[all_signals_old['depth'] ==  # определяем интервалы индексов min - max
                            get_nearest_value(all_signals_old['depth'], form.doubleSpinBox_int_min.value())].tolist()[0]
    int_max_old = all_signals_old.index[all_signals_old['depth'] ==
                            get_nearest_value(all_signals_old['depth'], form.doubleSpinBox_int_max.value())].tolist()[0]

    for i in range(1, 16):
        if all_signals_old[i][1] != 0:
            max_min = all_signals_old[i].iloc[int_min_old:int_max_old].max() / \
                      all_signals_old[i].iloc[int_min_old:int_max_old].min()
            all_stat_old['max_min'][i] = max_min  # расчет отношения макс к мин и заносим значение в таблицу статистики
            all_signals_old[str(i) + '-1'] = all_signals_old[i].shift(-1)  # сдвиг огибающ сигнала на 1
            all_signals_old[str(i) + '_mean'] = all_signals_old[i].rolling(mean_win, min_periods=1, center=True).mean()
            all_signals_old[str(i) + '_norm'] = all_signals_old[i] / (all_signals_old[str(i) + '_mean'] + coeff_norm)
            # расчет цементограммы по усредняющей с коэффициентом
            popt, pcov = curve_fit(func1, all_signals_old['depth'].iloc[int_min_old:int_max_old],  # расчет коэф для функции
                                   all_signals_old[i].iloc[int_min_old:int_max_old])
            all_stat_old['coeff_A'][i] = popt[0]
            all_stat_old['coeff_B'][i] = popt[1]
            all_signals_old[str(i) + '_func'] = func1(all_signals_old['depth'], popt[0], popt[1]) + coeff_func  # построение
            # фунции в интервале сигнала со сдвигом по Y на coeff_func

            """Расчет коэффициента затухания"""
            all_signals_old[str(i) + '_func_kdamp'] = all_signals_old[str(i) + '_func'][0] / all_signals_old[str(i) + '_func']
            i_kdamp = all_signals_old.index[
                all_signals_old[str(i) + '_func_kdamp'] == get_nearest_value(all_signals_old[str(i) + '_func_kdamp'],
                                                                         np.exp(1))].tolist()[0]
            all_stat_old['k_damp'][i] = np.exp(1) / all_signals_old['depth'][i_kdamp]
            all_stat_old['gl_kd'][i] = all_signals_old['depth'][i_kdamp]

            all_signals_old[str(i) + '_rel_ampl'] = (all_signals_old[str(i) + '-1'] - all_signals_old[str(i) + '_func']) / \
                                                all_signals_old[str(i) + '-1']  # расчет цементограммы по функции
            all_signals_old[str(i) + '_mean-1'] = all_signals_old[str(i) + '_mean'].shift(-1)  # сдвиг усредненного сигнала
            all_signals_old[str(i) + '_diff_norm'] = (all_signals_old[str(i) + '_mean-1'] -  # нормированная производная
                                                      all_signals_old[str(i) + '_mean']) / all_signals_old[str(i) + '_mean']
            all_signals_old[str(i) + '_diff_norm-1'] = all_signals_old[str(i) + '_diff_norm'].shift(-1)  # сдвиг
            all_signals_old[str(i) + '_diff_result'] = (all_signals_old[str(i) + '_diff_norm-1'] + all_signals_old[
                str(i) + '_diff_norm']) * coeff_dif_res  # расчет цементограммы по производной (как в щелкуне)

            std_25 = all_signals_old[i].iloc[-50:].std()  # стандартное отклонение по последним 25 метрам
            for n, k in enumerate(all_signals_old[str(i) + '_mean'].iloc[int_min_old:int_max_old]):
                if k <= std_25 + all_signals_old[str(i) + '_mean'].iloc[int_min_old:int_max_old].min():
                    useful_depth = all_signals_old['depth'][n]
                    all_stat_old['useful_depth'][i] = useful_depth
                    break
            form.progressBar_2.setValue(i)
    # расчет рекомендуемого окна усреднения по формуле из Щелкуна, установка значения в спинбок
    auto_mean_win = (int_max_old - int_min_old) / (1 + 3.322 * np.log(int_max_old - int_min_old)) / 2.302585
    form.spinBox_mean_win_old.setValue(auto_mean_win)
    print(all_stat_old)
    print(all_signals_old)

    all_signals_old.to_csv('result_old.txt', sep='\t')
    choice_signal()


def calc():
    if form.checkBox_signal1.isEnabled():
        calc_to_int()
    if form.checkBox_signal_old_1.isEnabled():
        calc_to_int_old()


def choice_signal():
    """
    выбор сигналов для построения
    """
    form.graphicsView.clear()
    if form.checkBox_signal1.checkState() == 2 and form.checkBox_signal1.isEnabled():
        plot_graph(1, 1.5, [1])
    if form.checkBox_signal2.checkState() == 2 and form.checkBox_signal2.isEnabled():
        plot_graph(2, 1.5, [3, 3])
    if form.checkBox_signal3.checkState() == 2 and form.checkBox_signal3.isEnabled():
        plot_graph(3, 1, [1])
    if form.checkBox_signal4.checkState() == 2 and form.checkBox_signal4.isEnabled():
        plot_graph(4, 1, [3, 3])
    if form.checkBox_signal5.checkState() == 2 and form.checkBox_signal5.isEnabled():
        plot_graph(5, 1.5, [1])
    if form.checkBox_signal6.checkState() == 2 and form.checkBox_signal6.isEnabled():
        plot_graph(6, 1.5, [3, 3])
    if form.checkBox_signal7.checkState() == 2 and form.checkBox_signal7.isEnabled():
        plot_graph(7, 1, [1])
    if form.checkBox_signal8.checkState() == 2 and form.checkBox_signal8.isEnabled():
        plot_graph(8, 1, [3, 3])
    if form.checkBox_sum1.checkState() == 2 and form.checkBox_sum1.isEnabled():
        plot_graph(9, 2.5, [3, 3])
    if form.checkBox_sum2.checkState() == 2 and form.checkBox_sum2.isEnabled():
        plot_graph(10, 2.5, [3, 3])
    if form.checkBox_sum3.checkState() == 2 and form.checkBox_sum3.isEnabled():
        plot_graph(11, 2.5, [3, 3])

    if form.checkBox_signal_old_1.checkState() == 2 and form.checkBox_signal_old_1.isEnabled():
        plot_graph_old(1, 1.5, [1])
    if form.checkBox_signal_old_2.checkState() == 2 and form.checkBox_signal_old_2.isEnabled():
        plot_graph_old(2, 1.5, [3, 3])
    if form.checkBox_signal_old_3.checkState() == 2 and form.checkBox_signal_old_3.isEnabled():
        plot_graph_old(3, 1, [1])
    if form.checkBox_signal_old_4.checkState() == 2 and form.checkBox_signal_old_4.isEnabled():
        plot_graph_old(4, 1, [3, 3])
    if form.checkBox_signal_old_5.checkState() == 2 and form.checkBox_signal_old_5.isEnabled():
        plot_graph_old(5, 1.5, [1])
    if form.checkBox_signal_old_6.checkState() == 2 and form.checkBox_signal_old_6.isEnabled():
        plot_graph_old(6, 1.5, [3, 3])
    if form.checkBox_signal_old_7.checkState() == 2 and form.checkBox_signal_old_7.isEnabled():
        plot_graph_old(7, 1, [1])
    if form.checkBox_signal_old_8.checkState() == 2 and form.checkBox_signal_old_8.isEnabled():
        plot_graph_old(8, 1, [3, 3])
    if form.checkBox_signal_old_9.checkState() == 2 and form.checkBox_signal_old_9.isEnabled():
        plot_graph_old(9, 1.5, [1])
    if form.checkBox_signal_old_10.checkState() == 2 and form.checkBox_signal_old_10.isEnabled():
        plot_graph_old(10, 1.5, [3, 3])
    if form.checkBox_signal_old_11.checkState() == 2 and form.checkBox_signal_old_11.isEnabled():
        plot_graph_old(11, 1, [1])
    if form.checkBox_signal_old_12.checkState() == 2 and form.checkBox_signal_old_12.isEnabled():
        plot_graph_old(12, 1, [3, 3])
    if form.checkBox_sum1_old.checkState() == 2 and form.checkBox_sum1_old.isEnabled():
        plot_graph_old(13, 2.5, [3, 3])
    if form.checkBox_sum2_old.checkState() == 2 and form.checkBox_sum2_old.isEnabled():
        plot_graph_old(14, 2.5, [3, 3])
    if form.checkBox_sum3_old.checkState() == 2 and form.checkBox_sum3_old.isEnabled():
        plot_graph_old(15, 2.5, [3, 3])

    if form.checkBox_noise.checkState() == 2 and form.checkBox_noise.isEnabled():
        form.graphicsView.plot(x=all_signals['depth'].iloc[int_min:int_max].tolist(),
                               y=all_signals['noise'].iloc[int_min:int_max].tolist(),
                               pen=pg.mkPen(color='c', dash=[2, 1, 4], width=1))
    if form.checkBox_noise_old.checkState() == 2 and form.checkBox_noise_old.isEnabled():
        form.graphicsView.plot(x=all_signals_old['depth'].iloc[int_min_old:int_max_old].tolist(),
                               y=all_signals_old['noise'].iloc[int_min_old:int_max_old].tolist(),
                               pen=pg.mkPen(color='c', dash=[2, 1, 4], width=1))
    if form.checkBox_defect.checkState() == 2 and form.checkBox_defect.isEnabled():
        form.graphicsView.plot(x=all_signals['depth'].iloc[int_min:int_max].tolist(),
                               y=all_signals['defect'].iloc[int_min:int_max].tolist(),
                               pen=pg.mkPen(color='c', dash=[2, 2], width=1))


def plot_graph(n_sig, sig_width, sig_dash):
    """
    выбор и построение кривых для выбранных сигналов
    принимает параметры линии для построения
    """
    if form.checkBox_origin_sig.checkState() == 2:
        form.graphicsView.plot(x=all_signals['depth'].iloc[int_min:int_max].tolist(),
                               y=all_signals[n_sig].iloc[int_min:int_max].tolist(),
                               pen=pg.mkPen(width=sig_width, dash=sig_dash))
    if form.checkBox_envelop.checkState() == 2:
        form.graphicsView.plot(x=all_signals['depth'].iloc[int_min:int_max].tolist(),
                               y=all_signals[str(n_sig) + '_envelope'].iloc[int_min:int_max].tolist(),
                               pen=pg.mkPen(color='r', width=sig_width, dash=sig_dash))
    if form.checkBox_func.checkState() == 2:
        form.graphicsView.plot(x=all_signals['depth'].iloc[int_min:int_max].tolist(),
                               y=all_signals[str(n_sig) + '_func'].iloc[int_min:int_max].tolist(),
                               pen=pg.mkPen(color='g', width=sig_width, dash=sig_dash))
    if form.checkBox_mean.checkState() == 2:
        form.graphicsView.plot(x=all_signals['depth'].iloc[int_min:int_max].tolist(),
                               y=all_signals[str(n_sig) + '_mean'].iloc[int_min:int_max].tolist(),
                               pen=pg.mkPen(color='b', width=sig_width, dash=sig_dash))
    if form.checkBox_norm.checkState() == 2:
        form.graphicsView.plot(x=all_signals['depth'].iloc[int_min:int_max].tolist(),
                               y=all_signals[str(n_sig) + '_norm'].iloc[int_min:int_max].tolist(),
                               pen=pg.mkPen(color='y', width=sig_width, dash=sig_dash))
    if form.checkBox_rel_ampl.checkState() == 2:
        form.graphicsView.plot(x=all_signals['depth'].iloc[int_min:int_max].tolist(),
                               y=all_signals[str(n_sig) + '_rel_ampl'].iloc[int_min:int_max].tolist(),
                               pen=pg.mkPen(color='c', width=sig_width, dash=sig_dash))
    if form.checkBox_diff_result.checkState() == 2:
        form.graphicsView.plot(x=all_signals['depth'].iloc[int_min:int_max].tolist(),
                               y=all_signals[str(n_sig) + '_diff_result'].iloc[int_min:int_max].tolist(),
                               pen=pg.mkPen(color='y', width=sig_width, dash=sig_dash))


def plot_graph_old(n_sig, sig_width, sig_dash):
    """
    выбор и построение кривых для выбранных сигналов
    принимает параметры линии для построения
    """
    if form.checkBox_envelop_old.checkState() == 2:
        form.graphicsView.plot(x=all_signals_old['depth'].iloc[int_min_old:int_max_old].tolist(),
                               y=all_signals_old[n_sig].iloc[int_min_old:int_max_old].tolist(),
                               pen=pg.mkPen(color='b', width=sig_width, dash=sig_dash))
    if form.checkBox_func_old.checkState() == 2:
        form.graphicsView.plot(x=all_signals_old['depth'].iloc[int_min_old:int_max_old].tolist(),
                               y=all_signals_old[str(n_sig) + '_func'].iloc[int_min_old:int_max_old].tolist(),
                               pen=pg.mkPen(color='g', width=sig_width, dash=sig_dash))
    if form.checkBox_mean_old.checkState() == 2:
        form.graphicsView.plot(x=all_signals_old['depth'].iloc[int_min_old:int_max_old].tolist(),
                               y=all_signals_old[str(n_sig) + '_mean'].iloc[int_min_old:int_max_old].tolist(),
                               pen=pg.mkPen(color='r', width=sig_width, dash=sig_dash))
    if form.checkBox_norm_old.checkState() == 2:
        form.graphicsView.plot(x=all_signals_old['depth'].iloc[int_min_old:int_max_old].tolist(),
                               y=all_signals_old[str(n_sig) + '_norm'].iloc[int_min_old:int_max_old].tolist(),
                               pen=pg.mkPen(color='y', width=sig_width, dash=sig_dash))
    if form.checkBox_rel_ampl_old.checkState() == 2:
        form.graphicsView.plot(x=all_signals_old['depth'].iloc[int_min_old:int_max_old].tolist(),
                               y=all_signals_old[str(n_sig) + '_rel_ampl'].iloc[int_min_old:int_max_old].tolist(),
                               pen=pg.mkPen(color='c', width=sig_width, dash=sig_dash))
    if form.checkBox_diff_result_old.checkState() == 2:
        form.graphicsView.plot(x=all_signals_old['depth'].iloc[int_min_old:int_max_old].tolist(),
                               y=all_signals_old[str(n_sig) + '_diff_result'].iloc[int_min_old:int_max_old].tolist(),
                               pen=pg.mkPen(color='y', width=sig_width, dash=sig_dash))


def change_mean_win():
    """
    пересчёт кривых при изменении коэффициентов
    """
    mean_win = form.spinBox_mean_win.value()
    coeff_norm = form.doubleSpinBox_coeff_norm.value()
    coeff_dif_res = form.spinBox_coeff_dif_res.value()
    for i in range(1, 12):
        if all_signals[str(i) + '_envelope'][1] != 0:
            all_signals[str(i) + '_mean'] = all_signals[str(i) + '_envelope'].rolling(mean_win, min_periods=1,
                                                                                      center=True).mean()
            all_signals[str(i) + '_norm'] = all_signals[str(i) + '_envelope'] / (all_signals[str(i) + '_mean'] + coeff_norm)
            all_signals[str(i) + '_mean-1'] = all_signals[str(i) + '_mean'].shift(-1)
            all_signals[str(i) + '_diff_norm'] = (all_signals[str(i) + '_mean-1'] - all_signals[str(i) + '_mean']) / \
                                                 all_signals[str(i) + '_mean']
            all_signals[str(i) + '_diff_norm-1'] = all_signals[str(i) + '_diff_norm'].shift(-1)
            all_signals[str(i) + '_diff_result'] = (all_signals[str(i) + '_diff_norm-1'] + all_signals[
                str(i) + '_diff_norm']) * coeff_dif_res

    all_signals.to_csv('result.txt', sep='\t')
    choice_signal()


def change_mean_win_old():
    """
    пересчёт кривых при изменении коэффициентов
    """
    coeff_norm = form.doubleSpinBox_coeff_norm_old.value()  # считываем значения коэффициентов из спинбоксов
    coeff_dif_res = form.spinBox_coeff_dif_res_old.value()
    mean_win = form.spinBox_mean_win_old.value()

    for i in range(1, 16):
        if all_signals_old[i][1] != 0:
            all_signals_old[str(i) + '_mean'] = all_signals_old[i].rolling(mean_win, min_periods=1, center=True).mean()
            all_signals_old[str(i) + '_norm'] = all_signals_old[i] / (all_signals_old[str(i) + '_mean'] + coeff_norm)


            all_signals_old[str(i) + '_mean-1'] = all_signals_old[str(i) + '_mean'].shift(-1)  # сдвиг усредненного сигнала
            all_signals_old[str(i) + '_diff_norm'] = (all_signals_old[str(i) + '_mean-1'] -  # нормированная производная
                                                      all_signals_old[str(i) + '_mean']) / all_signals_old[str(i) + '_mean']
            all_signals_old[str(i) + '_diff_norm-1'] = all_signals_old[str(i) + '_diff_norm'].shift(-1)  # сдвиг
            all_signals_old[str(i) + '_diff_result'] = (all_signals_old[str(i) + '_diff_norm-1'] + all_signals_old[
                str(i) + '_diff_norm']) * coeff_dif_res  # расчет цементограммы по производной (как в щелкуне)

    all_signals_old.to_csv('result_old.txt', sep='\t')
    choice_signal()


def change_level_defect():
    """
    изменение уровня дефектов
    """
    level_defect = form.doubleSpinBox_level_defect.value()
    all_signals['defect'] = level_defect
    choice_signal()


# def func1(x, a, b):
#     if form.comboBox_func.currentText() == 'y = a*b^x':
#         return a * b ** x
#     if form.comboBox_func.currentText() == 'y = a*x + b':
#         return a * x + b

def func1(x, a, b):
    """ функция затухания """
    return a * b ** x
    # return a*(x-b)**c


def change_func():
    """
    сдвиг функции затухания
    """
    coeff_func = form.doubleSpinBox_coeff_func.value()
    kA = form.doubleSpinBox_kA.value()
    kB = form.doubleSpinBox_kB.value()
    for i in range(1, 12):
        if all_signals[str(i) + '_envelope'][1] != 0:
            all_signals[str(i) + '_func'] = func1(all_signals['depth'], all_stat['coeff_A'][i]+kA, all_stat['coeff_B'][i]+kB) + coeff_func
            all_signals[str(i) + '_rel_ampl'] = (all_signals[str(i) + '_envelope-1'] - all_signals[str(i) + '_func']) / \
                                                all_signals[str(i) + '_envelope-1']
    choice_signal()


def change_func_old():
    """
    сдвиг функции затухания
    """
    coeff_func = form.doubleSpinBox_coeff_func_old.value()
    for i in range(1, 16):
        if all_signals_old[i][1] != 0:
            popt, pcov = curve_fit(func1, all_signals_old['depth'].iloc[int_min_old:int_max_old],  # расчет коэф для функции
                                   all_signals_old[i].iloc[int_min_old:int_max_old])
            all_stat_old['coeff_A'][i] = popt[0]
            all_stat_old['coeff_B'][i] = popt[1]
            all_signals_old[str(i) + '_func'] = func1(all_signals_old['depth'], popt[0], popt[1]) + coeff_func  # построение
            # фунции в интервале сигнала со сдвигом по Y на coeff_func
            all_signals_old[str(i) + '_rel_ampl'] = (all_signals_old[str(i) + '-1'] - all_signals_old[str(i) + '_func']) / \
                                                    all_signals_old[str(i) + '-1']  # расчет цементограммы по функции
    choice_signal()


def check_coeff_int():
    """ проверка интервала """
    form.doubleSpinBox_int_min.setMaximum(form.doubleSpinBox_int_max.value())
    form.doubleSpinBox_int_max.setMinimum(form.doubleSpinBox_int_min.value())


def get_nearest_value(iterable, value):
    """ функция поиска ближайщего значения """
    return min(iterable, key=lambda x: abs(x - value))


def sum_signals(n_sum):
    n_signals = 0
    all_signals[str(n_sum) + '_envelope'] = 0
    if form.checkBox_signal1.checkState() == 2:
        all_signals[str(n_sum) + '_envelope'] = all_signals[str(n_sum) + '_envelope'] + all_signals['1_envelope']
        n_signals += 1
    if form.checkBox_signal2.checkState() == 2:
        all_signals[str(n_sum) + '_envelope'] = all_signals[str(n_sum) + '_envelope'] + all_signals['2_envelope']
        n_signals += 1
    if form.checkBox_signal3.checkState() == 2:
        all_signals[str(n_sum) + '_envelope'] = all_signals[str(n_sum) + '_envelope'] + all_signals['3_envelope']
        n_signals += 1
    if form.checkBox_signal4.checkState() == 2:
        all_signals[str(n_sum) + '_envelope'] = all_signals[str(n_sum) + '_envelope'] + all_signals['4_envelope']
        n_signals += 1
    if form.checkBox_signal5.checkState() == 2:
        all_signals[str(n_sum) + '_envelope'] = all_signals[str(n_sum) + '_envelope'] + all_signals['5_envelope']
        n_signals += 1
    if form.checkBox_signal6.checkState() == 2:
        all_signals[str(n_sum) + '_envelope'] = all_signals[str(n_sum) + '_envelope'] + all_signals['6_envelope']
        n_signals += 1
    if form.checkBox_signal7.checkState() == 2:
        all_signals[str(n_sum) + '_envelope'] = all_signals[str(n_sum) + '_envelope'] + all_signals['7_envelope']
        n_signals += 1
    if form.checkBox_signal8.checkState() == 2:
        all_signals[str(n_sum) + '_envelope'] = all_signals[str(n_sum) + '_envelope'] + all_signals['8_envelope']
        n_signals += 1
    if n_signals != 0:
        all_signals[str(n_sum) + '_envelope'] = all_signals[str(n_sum) + '_envelope']/n_signals
        coeff_norm = form.doubleSpinBox_coeff_norm.value()  # считываем значения коэффициентов из спинбоксов
        coeff_func = form.doubleSpinBox_coeff_func.value()
        coeff_dif_res = form.spinBox_coeff_dif_res.value()
        mean_win = form.spinBox_mean_win.value()
        max_min = all_signals[str(n_sum) + '_envelope'].iloc[int_min:int_max].max() / \
                  all_signals[str(n_sum) + '_envelope'].iloc[int_min:int_max].min()
        all_stat['max_min'][n_sum] = max_min  # расчет отношения макс к мин и заносим значение в таблицу статистики
        all_signals[str(n_sum) + '_envelope-1'] = all_signals[str(n_sum) + '_envelope'].shift(-1)  # сдвиг огибающ сигнала на 1
        all_signals[str(n_sum) + '_mean'] = all_signals[str(n_sum) + '_envelope'].rolling(mean_win, min_periods=1,
                                                                                  center=True).mean()  # усреднение
        all_signals[str(n_sum) + '_norm'] = all_signals[str(n_sum) + '_envelope'] / (all_signals[str(n_sum) + '_mean'] + coeff_norm)
        # расчет цементограммы по усредняющей с коэффициентом
        popt, pcov = curve_fit(func1, all_signals['depth'].iloc[int_min:int_max],  # расчет коэф для функции
                               all_signals[str(n_sum) + '_envelope'].iloc[int_min:int_max])
        all_stat['coeff_A'][n_sum] = popt[0]
        all_stat['coeff_B'][n_sum] = popt[1]
        all_signals[str(n_sum) + '_func'] = func1(all_signals['depth'], popt[0], popt[1]) + coeff_func  # построение
        # фунции в интервале сигнала со сдвигом по Y на coeff_func

        """Расчет коэффициента затухания"""
        all_signals[str(n_sum) + '_func_kdamp'] = all_signals[str(n_sum) + '_func'][0]/all_signals[str(n_sum) + '_func']
        i_kdamp = all_signals.index[all_signals[str(n_sum) + '_func_kdamp'] == get_nearest_value(all_signals[str(n_sum)
                                                                             +'_func_kdamp'], np.exp(1))].tolist()[0]
        all_stat['k_damp'][n_sum] = np.exp(1) / all_signals['depth'][i_kdamp]

        all_signals[str(n_sum) + '_rel_ampl'] = (all_signals[str(n_sum) + '_envelope-1'] - all_signals[str(n_sum) + '_func']) / \
                                            all_signals[str(n_sum) + '_envelope-1']  # расчет цементограммы по функции
        all_signals[str(n_sum) + '_mean-1'] = all_signals[str(n_sum) + '_mean'].shift(-1)  # сдвиг усредненного сигнала на 1
        all_signals[str(n_sum) + '_diff_norm'] = (all_signals[str(n_sum) + '_mean-1'] - all_signals[str(n_sum) + '_mean']) / \
                                             all_signals[str(n_sum) + '_mean']  # нормированная производная
        all_signals[str(n_sum) + '_diff_norm-1'] = all_signals[str(n_sum) + '_diff_norm'].shift(-1)  # сдвиг
        all_signals[str(n_sum) + '_diff_result'] = (all_signals[str(n_sum) + '_diff_norm-1'] + all_signals[
            str(n_sum) + '_diff_norm']) * coeff_dif_res  # расчет цементограммы по производной (как в щелкуне)
        std_25 = all_signals[str(n_sum) + '_envelope'].iloc[-430:].std()  # стандартное отклонение по последним 25 метрам
        for n, k in enumerate(all_signals[str(n_sum) + '_mean'].iloc[int_min:int_max]):
            if k <= std_25 + all_signals[str(n_sum) + '_mean'].iloc[int_min:int_max].min():
                useful_depth = all_signals['depth'][n]
                all_stat['useful_depth'][n_sum] = useful_depth
                break
        print(all_stat)
        choice_signal()


def sum_signals_old(n_sum):
    n_signals = 0
    all_signals_old[n_sum] = 0
    if form.checkBox_signal_old_1.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[1]
        n_signals += 1
    if form.checkBox_signal_old_2.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[2]
        n_signals += 1
    if form.checkBox_signal_old_3.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[3]
        n_signals += 1
    if form.checkBox_signal_old_4.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[4]
        n_signals += 1
    if form.checkBox_signal_old_5.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[5]
        n_signals += 1
    if form.checkBox_signal_old_6.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[6]
        n_signals += 1
    if form.checkBox_signal_old_7.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[7]
        n_signals += 1
    if form.checkBox_signal_old_8.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[8]
        n_signals += 1
    if form.checkBox_signal_old_9.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[8]
        n_signals += 1
    if form.checkBox_signal_old_10.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[10]
        n_signals += 1
    if form.checkBox_signal_old_11.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[11]
        n_signals += 1
    if form.checkBox_signal_old_12.checkState() == 2:
        all_signals_old[n_sum] = all_signals_old[n_sum] + all_signals_old[12]
        n_signals += 1
    if n_signals != 0:
        all_signals_old[n_sum] = all_signals_old[n_sum]/n_signals
        coeff_norm = form.doubleSpinBox_coeff_norm_old.value()  # считываем значения коэффициентов из спинбоксов
        coeff_func = form.doubleSpinBox_coeff_func_old.value()
        coeff_dif_res = form.spinBox_coeff_dif_res_old.value()
        mean_win = form.spinBox_mean_win_old.value()

        max_min = all_signals_old[n_sum].iloc[int_min_old:int_max_old].max() / \
                  all_signals_old[n_sum].iloc[int_min_old:int_max_old].min()
        all_stat_old['max_min'][n_sum] = max_min  # расчет отношения макс к мин и заносим значение в таблицу статистики
        all_signals_old[str(n_sum) + '-1'] = all_signals_old[n_sum].shift(-1)  # сдвиг огибающ сигнала на 1
        all_signals_old[str(n_sum) + '_mean'] = all_signals_old[n_sum].rolling(mean_win, min_periods=1, center=True).mean()
        all_signals_old[str(n_sum) + '_norm'] = all_signals_old[n_sum] / (all_signals_old[str(n_sum) + '_mean'] + coeff_norm)
        # расчет цементограммы по усредняющей с коэффициентом
        popt, pcov = curve_fit(func1, all_signals_old['depth'].iloc[int_min_old:int_max_old],  # расчет коэф для функции
                               all_signals_old[n_sum].iloc[int_min_old:int_max_old])
        all_stat_old['coeff_A'][n_sum] = popt[0]
        all_stat_old['coeff_B'][n_sum] = popt[1]
        all_signals_old[str(n_sum) + '_func'] = func1(all_signals_old['depth'], popt[0], popt[1]) + coeff_func  # построение
        # фунции в интервале сигнала со сдвигом по Y на coeff_func
        all_signals_old[str(n_sum) + '_rel_ampl'] = (all_signals_old[str(n_sum) + '-1'] - all_signals_old[str(n_sum) + '_func']) / \
                                                all_signals_old[str(n_sum) + '-1']  # расчет цементограммы по функции
        all_signals_old[str(n_sum) + '_mean-1'] = all_signals_old[str(n_sum) + '_mean'].shift(-1)  # сдвиг усредненного сигнала
        all_signals_old[str(n_sum) + '_diff_norm'] = (all_signals_old[str(n_sum) + '_mean-1'] -  # нормированная производная
                                                  all_signals_old[str(n_sum) + '_mean']) / all_signals_old[str(n_sum) + '_mean']
        all_signals_old[str(n_sum) + '_diff_norm-1'] = all_signals_old[str(n_sum) + '_diff_norm'].shift(-1)  # сдвиг
        all_signals_old[str(n_sum) + '_diff_result'] = (all_signals_old[str(n_sum) + '_diff_norm-1'] + all_signals_old[
            str(n_sum) + '_diff_norm']) * coeff_dif_res  # расчет цементограммы по производной (как в щелкуне)

        std_25 = all_signals_old[n_sum].iloc[-50:].std()  # стандартное отклонение по последним 25 метрам
        for n, k in enumerate(all_signals_old[str(n_sum) + '_mean'].iloc[int_min_old:int_max_old]):
            if k <= std_25 + all_signals_old[str(n_sum) + '_mean'].iloc[int_min_old:int_max_old].min():
                useful_depth = all_signals_old['depth'][n]
                all_stat_old['useful_depth'][n_sum] = useful_depth
                break
        print(all_stat)
        choice_signal()


def sum1():
    all_stat['name'][9] = text_sum()
    sum_signals(9)
    form.checkBox_sum1.setText(text_sum())
    form.checkBox_sum1.setToolTip('Коэффициент затухания - ' + str(round(all_stat['k_damp'][9], 3)) +
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][9], 2)) +
                                     ' м\nМакс/Мин - ' + str(round(all_stat['max_min'][9], 3)))
    form.checkBox_sum1.setEnabled(True)


def sum2():
    all_stat['name'][10] = text_sum()
    sum_signals(10)
    form.checkBox_sum2.setText(text_sum())
    form.checkBox_sum2.setToolTip('Коэффициент затухания - ' + str(round(all_stat['k_damp'][10], 3)) +
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][10], 2)) +
                                     ' м\nМакс/Мин - ' + str(round(all_stat['max_min'][10], 3)))
    form.checkBox_sum2.setEnabled(True)


def sum3():
    all_stat['name'][11] = text_sum()
    sum_signals(11)
    form.checkBox_sum3.setText(text_sum())
    form.checkBox_sum3.setToolTip('Коэффициент затухания - ' + str(round(all_stat['k_damp'][11], 3)) +
                                     '\nЭффективная глубина - ' + str(round(all_stat['useful_depth'][11], 2)) +
                                     ' м\nМакс/Мин - ' + str(round(all_stat['max_min'][11], 3)))
    form.checkBox_sum3.setEnabled(True)


def sum1_old():
    all_stat_old['name'][13] = text_sum_old()
    sum_signals_old(13)
    form.checkBox_sum1_old.setText(text_sum_old())
    form.checkBox_sum1_old.setEnabled(True)


def sum2_old():
    all_stat_old['name'][14] = text_sum_old()
    sum_signals_old(14)
    form.checkBox_sum2_old.setText(text_sum_old())
    form.checkBox_sum2_old.setEnabled(True)


def sum3_old():
    all_stat_old['name'][15] = text_sum_old()
    sum_signals_old(15)
    form.checkBox_sum3_old.setText(text_sum_old())
    form.checkBox_sum3_old.setEnabled(True)


def text_sum():
    text = 'Сумма: '
    if form.checkBox_signal1.checkState() == 2:
        text = text + '1 '
    if form.checkBox_signal2.checkState() == 2:
        text = text + '2 '
    if form.checkBox_signal3.checkState() == 2:
        text = text + '3 '
    if form.checkBox_signal4.checkState() == 2:
        text = text + '4 '
    if form.checkBox_signal5.checkState() == 2:
        text = text + '5 '
    if form.checkBox_signal6.checkState() == 2:
        text = text + '6 '
    if form.checkBox_signal7.checkState() == 2:
        text = text + '7 '
    if form.checkBox_signal8.checkState() == 2:
        text = text + '8 '
    return text


def text_sum_old():
    text = 'Сумма: '
    if form.checkBox_signal_old_1.checkState() == 2:
        text = text + '1 '
    if form.checkBox_signal_old_2.checkState() == 2:
        text = text + '2 '
    if form.checkBox_signal_old_3.checkState() == 2:
        text = text + '3 '
    if form.checkBox_signal_old_4.checkState() == 2:
        text = text + '4 '
    if form.checkBox_signal_old_5.checkState() == 2:
        text = text + '5 '
    if form.checkBox_signal_old_6.checkState() == 2:
        text = text + '6 '
    if form.checkBox_signal_old_7.checkState() == 2:
        text = text + '7 '
    if form.checkBox_signal_old_8.checkState() == 2:
        text = text + '8 '
    if form.checkBox_signal_old_9.checkState() == 2:
        text = text + '9 '
    if form.checkBox_signal_old_10.checkState() == 2:
        text = text + '10 '
    if form.checkBox_signal_old_11.checkState() == 2:
        text = text + '11 '
    if form.checkBox_signal_old_12.checkState() == 2:
        text = text + '12 '
    return text





form.Button_direct.clicked.connect(open_dir)
form.Button_direct_old.clicked.connect(open_dir_old)

form.checkBox_signal1.stateChanged.connect(choice_signal)
form.checkBox_signal2.stateChanged.connect(choice_signal)
form.checkBox_signal3.stateChanged.connect(choice_signal)
form.checkBox_signal4.stateChanged.connect(choice_signal)
form.checkBox_signal5.stateChanged.connect(choice_signal)
form.checkBox_signal6.stateChanged.connect(choice_signal)
form.checkBox_signal7.stateChanged.connect(choice_signal)
form.checkBox_signal8.stateChanged.connect(choice_signal)
form.checkBox_noise.stateChanged.connect(choice_signal)
form.checkBox_origin_sig.stateChanged.connect(choice_signal)
form.checkBox_envelop.stateChanged.connect(choice_signal)
form.checkBox_func.stateChanged.connect(choice_signal)
form.checkBox_mean.stateChanged.connect(choice_signal)
form.checkBox_norm.stateChanged.connect(choice_signal)
form.checkBox_defect.stateChanged.connect(choice_signal)
form.checkBox_rel_ampl.stateChanged.connect(choice_signal)
form.checkBox_diff_result.stateChanged.connect(choice_signal)

form.checkBox_signal_old_1.stateChanged.connect(choice_signal)
form.checkBox_signal_old_2.stateChanged.connect(choice_signal)
form.checkBox_signal_old_3.stateChanged.connect(choice_signal)
form.checkBox_signal_old_4.stateChanged.connect(choice_signal)
form.checkBox_signal_old_5.stateChanged.connect(choice_signal)
form.checkBox_signal_old_6.stateChanged.connect(choice_signal)
form.checkBox_signal_old_7.stateChanged.connect(choice_signal)
form.checkBox_signal_old_8.stateChanged.connect(choice_signal)
form.checkBox_signal_old_9.stateChanged.connect(choice_signal)
form.checkBox_signal_old_10.stateChanged.connect(choice_signal)
form.checkBox_signal_old_11.stateChanged.connect(choice_signal)
form.checkBox_signal_old_12.stateChanged.connect(choice_signal)
form.checkBox_noise_old.stateChanged.connect(choice_signal)
form.checkBox_envelop_old.stateChanged.connect(choice_signal)
form.checkBox_func_old.stateChanged.connect(choice_signal)
form.checkBox_mean_old.stateChanged.connect(choice_signal)
form.checkBox_norm_old.stateChanged.connect(choice_signal)
form.checkBox_rel_ampl_old.stateChanged.connect(choice_signal)
form.checkBox_diff_result_old.stateChanged.connect(choice_signal)

form.spinBox_mean_win.valueChanged.connect(change_mean_win)
form.doubleSpinBox_coeff_norm.valueChanged.connect(change_mean_win)
form.spinBox_coeff_dif_res.valueChanged.connect(change_mean_win)

form.spinBox_mean_win_old.valueChanged.connect(change_mean_win_old)
form.doubleSpinBox_coeff_norm_old.valueChanged.connect(change_mean_win_old)
form.spinBox_coeff_dif_res_old.valueChanged.connect(change_mean_win_old)

form.doubleSpinBox_coeff_func.valueChanged.connect(change_func)
form.doubleSpinBox_kA.valueChanged.connect(change_func)
form.doubleSpinBox_kB.valueChanged.connect(change_func)
form.doubleSpinBox_coeff_func_old.valueChanged.connect(change_func_old)

form.doubleSpinBox_int_max.valueChanged.connect(check_coeff_int)
form.doubleSpinBox_int_max.valueChanged.connect(check_coeff_int)
form.pushButton_int.clicked.connect(calc)
form.doubleSpinBox_level_defect.valueChanged.connect(change_level_defect)

form.pushButton_sum1.clicked.connect(sum1)
form.pushButton_sum2.clicked.connect(sum2)
form.pushButton_sum3.clicked.connect(sum3)
form.checkBox_sum1.stateChanged.connect(choice_signal)
form.checkBox_sum2.stateChanged.connect(choice_signal)
form.checkBox_sum3.stateChanged.connect(choice_signal)

form.pushButton_sum1_old.clicked.connect(sum1_old)
form.pushButton_sum2_old.clicked.connect(sum2_old)
form.pushButton_sum3_old.clicked.connect(sum3_old)
form.checkBox_sum1_old.stateChanged.connect(choice_signal)
form.checkBox_sum2_old.stateChanged.connect(choice_signal)
form.checkBox_sum3_old.stateChanged.connect(choice_signal)
app.exec_()
