import datetime

import pandas as pd
import numpy as np
from scipy import stats as st
from PyQt5.QtWidgets import QFileDialog, QCheckBox
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from .resources import template_path
from vac.domain.state import ApplicationState
from vac.domain.models import ProcessingParameters, ProcessingResult, SignalIdentity, ManualCorrection, ReportDetails
from vac.calculation import calculate_new, calculate_old, classify, ClassificationError, func1, get_nearest_value, calc_max_int
from vac.domain.models import QualityIntervals
from .io import lvm, twf, exports, legacy_options
from .gui import plotting
from .reporting import figures, word
from .processing import preprocessing, sums, pdf, interpretation, corrections

TEMPLATE_PATH = template_path()

class Controller:
    """One window owns one state; legacy formulas are retained during extraction."""

    def __init__(self, window, ui, state=None):
        self.window = window
        self.ui = ui
        self.state = state if state is not None else ApplicationState()
        self.ui.dateEdit.setDate(datetime.datetime.today() - datetime.timedelta(days=1))
        self.bind_signals()
        from .gui.archive import ArchiveMenu
        self.archive_menu = ArchiveMenu(self)
        from .gui.workspace import WorkspaceActions
        self.workspace_actions = WorkspaceActions(self)

    def read_display_values(self, names):
        """Snapshot scalar display settings; widgets never enter rendering modules."""
        return {name: getattr(getattr(self.ui, name.split('.')[0]), name.split('.')[1])()
                for name in names}

    def read_parameters(self, equipment, mean_window):
        suffix = '' if equipment == 'new' else '_old'
        return ProcessingParameters(
            self.ui.doubleSpinBox_int_min.value(), self.ui.doubleSpinBox_int_max.value(),
            getattr(self.ui, 'doubleSpinBox_coeff_norm' + suffix).value(),
            getattr(self.ui, 'doubleSpinBox_coeff_func' + suffix).value(),
            getattr(self.ui, 'spinBox_coeff_dif_res' + suffix).value(), mean_window)

    def clear_interpretation(self):
        self.state.clear_interpretation()
        self.ui.label_int_undefine.setText('Интервалы неопр. цемента:')
        self.ui.name_cement.setText('измерение цементограммы')
        self.ui.graphicsView.clear()
        if hasattr(self, 'workspace_actions'):
            self.workspace_actions.reset()

    def invalidate(self, equipment):
        self.state.invalidate(equipment)
        self.clear_interpretation()

    def begin_load(self, equipment):
        self.archive_menu.study_id = None
        self.state.reset_measurements(equipment)
        self.clear_interpretation()
        group = self.ui.new_device if equipment == 'new' else self.ui.old_device
        for checkbox in group.findChildren(QCheckBox):
            blocker = QtCore.QSignalBlocker(checkbox)
            checkbox.setChecked(False)
            checkbox.setEnabled(False)
            del blocker

    def bind_interpretation(self, equipment, signal):
        identity = SignalIdentity(equipment, getattr(self.state, equipment).revision, signal)
        self.state.interpretation.signal = identity
        self.state.interpretation.corrections.clear()
        self.state.interpretation.options = None
        self.state.interpretation.frequency = ''
        self.state.interpretation.damping = None
        if hasattr(self, 'workspace_actions'):
            self.workspace_actions.reset()

    def record_correction(self, operation):
        interpretation = self.state.interpretation
        if interpretation.signal is not None:
            values = tuple((name, getattr(self.ui, name).value()) for name in (
                'doubleSpinBox_x1_line', 'doubleSpinBox_x2_line', 'doubleSpinBox_y_line',
                'doubleSpinBox_undefine_min', 'doubleSpinBox_undefine_max'))
            flags = tuple((name, getattr(self.ui, name).isChecked()) for name in (
                'checkBox_shift_corr', 'checkBox_lin_cor', 'radioButton_KC',
                'radioButton_ChC', 'radioButton_OC', 'radioButton_NC'))
            interpretation.corrections.append(ManualCorrection(interpretation.signal, operation, values + flags))

    def bind_signals(self):
        self.ui.spinBox_mean_win.valueChanged.connect(lambda value: self.invalidate('new'))
        self.ui.doubleSpinBox_coeff_norm.valueChanged.connect(lambda value: self.invalidate('new'))
        self.ui.spinBox_coeff_dif_res.valueChanged.connect(lambda value: self.invalidate('new'))
        self.ui.doubleSpinBox_coeff_func.valueChanged.connect(lambda value: self.invalidate('new'))
        self.ui.doubleSpinBox_kA.valueChanged.connect(lambda value: self.invalidate('new'))
        self.ui.doubleSpinBox_kB.valueChanged.connect(lambda value: self.invalidate('new'))
        self.ui.spinBox_mean_win_old.valueChanged.connect(lambda value: self.invalidate('old'))
        self.ui.doubleSpinBox_coeff_norm_old.valueChanged.connect(lambda value: self.invalidate('old'))
        self.ui.spinBox_coeff_dif_res_old.valueChanged.connect(lambda value: self.invalidate('old'))
        self.ui.doubleSpinBox_coeff_func_old.valueChanged.connect(lambda value: self.invalidate('old'))
        self.ui.doubleSpinBox_kA_old.valueChanged.connect(lambda value: self.invalidate('old'))
        self.ui.doubleSpinBox_kB_old.valueChanged.connect(lambda value: self.invalidate('old'))
        self.ui.doubleSpinBox_int_min.valueChanged.connect(lambda value: (self.invalidate('new'), self.invalidate('old')))
        self.ui.doubleSpinBox_int_max.valueChanged.connect(lambda value: (self.invalidate('new'), self.invalidate('old')))
        self.ui.spinBox_mean.valueChanged.connect(lambda value: (self.invalidate('new'), self.invalidate('old')))
        self.proxy = pg.SignalProxy(self.ui.graphicsView.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

        self.ui.Button_direct.clicked.connect(self.open_dir)
        self.ui.Button_direct_old.clicked.connect(self.open_dir_old)
        self.ui.Button_open_sig.clicked.connect(self.open_sig)
        self.ui.Button_open_sig_old.clicked.connect(self.open_sig_old)
        self.ui.pushButton_mean.clicked.connect(self.calc_mean)

        self.ui.checkBox_signal1.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal2.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal3.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal4.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal5.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal6.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal7.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal8.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_noise.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_origin_sig.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_envelop.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_func.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_mean.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_norm.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_rel_ampl.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_diff_result.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_all_new.stateChanged.connect(self.check_all_new)

        self.ui.checkBox_signal_old_1.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_2.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_3.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_4.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_5.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_6.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_7.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_8.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_9.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_10.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_11.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_signal_old_12.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_noise_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_envelop_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_func_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_mean_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_norm_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_rel_ampl_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_diff_result_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_all_old.stateChanged.connect(self.check_all_old)

        self.ui.spinBox_mean_win.valueChanged.connect(self.change_mean_win)
        self.ui.doubleSpinBox_coeff_norm.valueChanged.connect(self.change_mean_win)
        self.ui.spinBox_coeff_dif_res.valueChanged.connect(self.change_mean_win)

        self.ui.spinBox_mean_win_old.valueChanged.connect(self.change_mean_win_old)
        self.ui.doubleSpinBox_coeff_norm_old.valueChanged.connect(self.change_mean_win_old)
        self.ui.spinBox_coeff_dif_res_old.valueChanged.connect(self.change_mean_win_old)

        self.ui.doubleSpinBox_coeff_func.valueChanged.connect(self.change_func)
        self.ui.doubleSpinBox_kA.valueChanged.connect(self.change_func)
        self.ui.doubleSpinBox_kB.valueChanged.connect(self.change_func)
        self.ui.doubleSpinBox_coeff_func_old.valueChanged.connect(self.change_func_old)
        self.ui.doubleSpinBox_kA_old.valueChanged.connect(self.change_func_old)
        self.ui.doubleSpinBox_kB_old.valueChanged.connect(self.change_func_old)

        self.ui.doubleSpinBox_int_max.valueChanged.connect(self.check_int_sig)
        self.ui.doubleSpinBox_int_max.valueChanged.connect(self.check_int_sig)
        self.ui.pushButton_int.clicked.connect(self.calc)
        self.ui.checkBox_abs_def.stateChanged.connect(self.calc)

        self.ui.pushButton_sum1.clicked.connect(self.sum1)
        self.ui.pushButton_sum2.clicked.connect(self.sum2)
        self.ui.pushButton_sum3.clicked.connect(self.sum3)
        self.ui.checkBox_sum1.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_sum2.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_sum3.stateChanged.connect(self.choice_signal)

        self.ui.pushButton_pdf.clicked.connect(self.calc_pdf)
        self.ui.checkBox_pdf.stateChanged.connect(self.choice_signal)
        self.ui.pushButton_pdf_old.clicked.connect(self.calc_pdf_old)
        self.ui.checkBox_pdf_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_diff_res_pdf_new.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_diff_res_pdf_old.stateChanged.connect(self.choice_signal)
        self.ui.pushButton_cem_pdf_new.clicked.connect(self.cement_from_pdf_new)
        self.ui.pushButton_cem_pdf_old.clicked.connect(self.cement_from_pdf_old)

        self.ui.checkBox_match_def_new.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_match_def_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_gen_def_new.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_gen_def_old.stateChanged.connect(self.choice_signal)

        self.ui.pushButton_sum1_old.clicked.connect(self.sum1_old)
        self.ui.pushButton_sum2_old.clicked.connect(self.sum2_old)
        self.ui.pushButton_sum3_old.clicked.connect(self.sum3_old)
        self.ui.checkBox_sum1_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_sum2_old.stateChanged.connect(self.choice_signal)
        self.ui.checkBox_sum3_old.stateChanged.connect(self.choice_signal)

        self.ui.pushButton_int_cement.clicked.connect(self.calc_int_cement)
        self.ui.doubleSpinBox_defect1.valueChanged.connect(self.check_int_defect)
        self.ui.doubleSpinBox_defect2.valueChanged.connect(self.check_int_defect)
        self.ui.pushButton_minmax_cem.clicked.connect(self.calc_maxmin_cem)
        self.ui.doubleSpinBox_x1_line.valueChanged.connect(self.check_int_corr)
        self.ui.doubleSpinBox_x2_line.valueChanged.connect(self.check_int_corr)
        self.ui.doubleSpinBox_y_line.valueChanged.connect(self.draw_cement)
        self.ui.pushButton_corr_sig.clicked.connect(self.corr_sig)
        self.ui.pushButton_corr_sig_bottom.clicked.connect(self.corr_sig_bottom)
        self.ui.pushButton_fix_corr.clicked.connect(self.fix_corr)
        self.ui.pushButton_plus_func.clicked.connect(self.plus_func)

        self.ui.doubleSpinBox_undefine_min.valueChanged.connect(self.check_int_undefine)
        self.ui.doubleSpinBox_undefine_max.valueChanged.connect(self.check_int_undefine)
        self.ui.pushButton_add_undefine.clicked.connect(self.add_undefine)

        self.ui.pushButton_cement.clicked.connect(self.cementogramma)

        self.ui.checkBox_corr_mode.stateChanged.connect(self.draw_cement)
        self.ui.pushButton_from_temp.clicked.connect(self.cement_temp)
        self.ui.pushButton_save_cement.clicked.connect(self.export_report)
        self.ui.pushButton_open_options.clicked.connect(self.open_options)

        self.ui.pushButton_all_cement.clicked.connect(self.all_cement)
        self.ui.pushButton_all_cement_old.clicked.connect(self.all_cement_old)
        self.ui.pushButton_choice_cement.clicked.connect(self.choice_cement)

        self.ui.pushButton_all_graphics.clicked.connect(self.draw_graphics)
        self.ui.pushButton_all_graphics_old.clicked.connect(self.draw_graphics_old)
        self.ui.pushButton_corr_matrix.clicked.connect(self.corr_matrix)

        self.ui.doubleSpinBox_defect1_new.valueChanged.connect(self.check_int_defect_general)
        self.ui.doubleSpinBox_defect2_new.valueChanged.connect(self.check_int_defect_general)
        self.ui.doubleSpinBox_defect1_old.valueChanged.connect(self.check_int_defect_general)
        self.ui.doubleSpinBox_defect2_old.valueChanged.connect(self.check_int_defect_general)
        self.ui.pushButton_recalc_qual_new.clicked.connect(self.recalc_qual_new)
        self.ui.pushButton_recalc_qual_old.clicked.connect(self.recalc_qual_old)

        self.ui.checkBox_background_white.stateChanged.connect(self.background_white)

    def open_dir(self):
        """ функция выбора директории с файлами измерений в формате *.lvm """
        dir_name = QFileDialog.getExistingDirectory()
        if not dir_name:
            return
        try:
            loaded, all_files = lvm.load_directory(dir_name, self.ui.spinBox_mean.value())
        except (ValueError, OSError, KeyError, IndexError) as exc:
            self.ui.label_info.setText(str(exc))
            self.ui.label_info.setStyleSheet('color:red')
            return
        self.begin_load('new')
        loaded.revision = self.state.new.revision
        self.state.new = loaded
        self.ui.progressBar.reset()
        self.ui.label_info.setText('Загрузка данных нового оборудования...')
        self.ui.label_info.setStyleSheet('color:blue')
        self.ui.label_direct.setText("<b>" + dir_name + ":</b>")
        self.ui.label_info.setText('Загружены данные нового оборудования!')
        self.ui.label_info.setStyleSheet('color:green')
        checkboxes = self.ui.new_device.findChildren(QCheckBox)
        for item in checkboxes:
            n_izm = int(item.text()[0:2])
            if n_izm < 9:
                item.setText(str(n_izm)+' '+all_files[n_izm-1][0:20])  # присваиваем чекбоксам названия файлов
                item.setEnabled(True)

        self.ui.checkBox_noise.setText(all_files[8][0:20])
        number_skv = all_files[0].split('_')[0]
        self.ui.lineEdit_number_skv.setText(number_skv)

        self.ui.checkBox_noise.setEnabled(True)
        self.ui.pushButton_sum1.setEnabled(True)
        self.ui.pushButton_sum2.setEnabled(True)
        self.ui.pushButton_sum3.setEnabled(True)

        self.set_int(self.state.new.signals)

        for i in range(9, 12):      # пустые столбцы для суммы сигналов
            self.state.new.signals[str(i) + '_envelope'] = 0

        self.calc_to_int()  # запускаем функцию обработки сигнала в интервале


    def open_dir_old(self):
        """ функция выбора директории с файлами измерений в формате *.TWF """
        dir_name = QFileDialog.getExistingDirectory()
        if not dir_name:
            return
        try:
            loaded, all_files = twf.load_directory(dir_name)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            self.ui.label_info.setText(str(exc))
            self.ui.label_info.setStyleSheet('color:red')
            return
        self.begin_load('old')
        loaded.revision = self.state.old.revision
        self.state.old = loaded
        self.ui.progressBar.reset()
        self.ui.label_info.setText('Загрузка данных старого оборудования...')
        self.ui.label_info.setStyleSheet('color:blue')
        self.ui.label_direct_old.setText("<b>" + dir_name + ":</b>")
        self.ui.label_info.setText('Загружены данные старого оборудования!')
        self.ui.label_info.setStyleSheet('color:green')
        checkboxes = self.ui.old_device.findChildren(QCheckBox)
        for item in checkboxes:
            n_izm = int(item.text()[0:2])
            if n_izm < 13:
                item.setText(str(n_izm)+' '+all_files[n_izm][0:7])   # присваиваем чекбоксам названия файлов - первые 7 символов
                item.setEnabled(True)
        self.ui.checkBox_noise_old.setText(all_files[0][0:7])
        self.ui.checkBox_noise_old.setEnabled(True)
        self.ui.pushButton_sum1_old.setEnabled(True)
        self.ui.pushButton_sum2_old.setEnabled(True)
        self.ui.pushButton_sum3_old.setEnabled(True)

        self.set_int(self.state.old.signals)

        for i in range(13, 16):      # пустые столбцы для суммы сигналов
            self.state.old.signals[i] = 0

        self.calc_to_int_old()  # запускаем функцию обработки сигнала в интервале


    def open_sig(self):
        """ функция выбора файлов измерений в формате *.lvm по одному """
        file_name = QFileDialog.getOpenFileName(filter='*.lvm')
        if not file_name[0]:
            return
        try:
            signal = lvm.read_lvm(file_name[0])
        except (ValueError, OSError) as exc:
            self.ui.label_info.setText(str(exc))
            self.ui.label_info.setStyleSheet('color:red')
            return
        self.invalidate('new')
        checkboxes = self.ui.new_device.findChildren(QCheckBox)
        n_chek = 0
        for item in checkboxes:
            if item.isEnabled() and int(item.text()[0:2]) not in [9, 10, 11]:
                n_chek += 1
        if n_chek == 8:
            self.begin_load('new')
            for item in checkboxes:
                item.setEnabled(False)
                item.setCheckState(0)
                item.setStyleSheet('background: #f0f0f0')
        if n_chek > 0:
            self.ui.pushButton_sum1.setEnabled(True)
            self.ui.pushButton_sum2.setEnabled(True)
            self.ui.pushButton_sum3.setEnabled(True)
        if preprocessing.extend_lvm_axis(self.state.new, signal):
            self.set_int(self.state.new.signals)
        mean = self.ui.spinBox_mean.value()
        for item in checkboxes:
            if not item.isEnabled() and int(item.text()[0:2]) not in [9, 10, 11]:
                n_sig = int(item.text()[0:2])
                item.setText(str(n_sig) + ' ' + file_name[0].split('/')[-1][:20])
                self.state.new.signals[n_sig] = np.abs(signal[2])
                self.state.new.signals[str(n_sig) + '_envelope'] = preprocessing.envelope(
                    self.state.new.signals['depth'], self.state.new.signals[n_sig], mean, bounds_error=False)
                self.state.new.statistics['name'][n_sig] = item.text()
                self.state.new.statistics['max_h'][n_sig] = signal[1].max()
                item.setEnabled(True)
                self.calc_to_int()
                item.setCheckState(2)
                break


    def calc_mean(self):
        self.invalidate('new')
        """ Пересчет усреднения исходного сигнала для нового оборудования """
        checkboxes = self.ui.new_device.findChildren(QCheckBox)
        mean = self.ui.spinBox_mean.value()
        for item in checkboxes:
            if item.isEnabled() and int(item.text()[0:2]) not in [9, 10, 11]:
                n_sig = int(item.text()[0:2])
                self.state.new.signals[str(n_sig) + '_envelope'] = preprocessing.envelope(
                    self.state.new.signals['depth'], self.state.new.signals[n_sig], mean, bounds_error=False)
        self.calc_to_int()


    def open_sig_old(self):
        """ функция выбора файлов измерений в формате *.TWF по одному """
        file_name = QFileDialog.getOpenFileName(filter='*.TWF')
        if not file_name[0]:
            return
        try:
            signal = twf.read_twf(file_name[0])
        except (ValueError, OSError) as exc:
            self.ui.label_info.setText(str(exc))
            self.ui.label_info.setStyleSheet('color:red')
            return
        self.invalidate('old')
        checkboxes = self.ui.old_device.findChildren(QCheckBox)
        n_chek = 0
        for item in checkboxes:
            if item.isEnabled() and int(item.text()[0:2]) not in [13, 14, 15]:
                n_chek += 1
        if n_chek == 12:
            self.begin_load('old')
            for item in checkboxes:
                item.setEnabled(False)
                item.setCheckState(0)
                item.setStyleSheet('background: #f0f0f0')
        if n_chek > 0:
            self.ui.pushButton_sum1_old.setEnabled(True)
            self.ui.pushButton_sum2_old.setEnabled(True)
            self.ui.pushButton_sum3_old.setEnabled(True)
        len_signal = len(signal)
        max_gl = len_signal * 0.515

        if preprocessing.extend_twf_axis(self.state.old, len_signal):
            self.set_int(self.state.old.signals)
        list_boxes = []
        for item in checkboxes:
            if not item.isEnabled() and int(item.text()[0:2]) not in [13, 14, 15]:
                list_boxes.append(int(item.text()[0:2]))
        list_boxes.sort()
        n_sig = list_boxes[0]
        self.state.old.signals[n_sig] = preprocessing.interpolate_twf(signal, self.state.old.signals['depth'], bounds_error=False)
        self.state.old.statistics['max_h'][n_sig] = max_gl
        for item in checkboxes:
            if int(item.text()[0:2]) == n_sig:
                item.setText(str(n_sig) + ' ' + file_name[0].split('/')[-1])
                self.state.old.statistics['name'][n_sig] = item.text()
                item.setEnabled(True)
                item.setCheckState(2)
                break
        preprocessing.scale_single_twf(self.state.old, n_sig)
        self.calc_to_int_old()


    def calc_to_int(self):
        if self.state.new.signals.empty:
            return
        self.archive_menu.run_id = None
        """ Функция пересчета исходного сигнала, расчет цементограмм """


        auto_mean_win = preprocessing.prepare_interval(self.state.new, self.ui.doubleSpinBox_int_min.value(), self.ui.doubleSpinBox_int_max.value())
        self.ui.spinBox_mean_win.setValue(int(auto_mean_win))
        mean_win = self.ui.spinBox_mean_win.value()

        self.ui.progressBar.setMaximum(8)
        self.ui.label_info.setText('Расчет кривых нового оборудования...')
        self.ui.label_info.setStyleSheet('color:blue')

        parameters = self.read_parameters('new', mean_win)
        self.state.new.capture_original()
        calculate_new(self.state.new, parameters, self.ui.progressBar.setValue)
        self.state.new.result = ProcessingResult(parameters, self.state.new.revision)
        checkboxes = self.ui.new_device.findChildren(QCheckBox)
        for item in checkboxes:
            if item.isEnabled():
                n_izm = int(item.text()[0:2])
                item.setToolTip('Коэффициент затухания - '+str(round(self.state.new.statistics['k_damp'][n_izm], 3))+
                                 '\nЭффективная глубина - ' + str(round(self.state.new.statistics['useful_depth'][n_izm], 2))+
                                 ' м\nМакс/Мин - '+str(round(self.state.new.statistics['max_min'][n_izm], 3)))
                item.setStyleSheet(self.k_damp_detect('new', n_izm))
        self.calc_defect_to_tab(self.state.new.signals, True)
        exports.write_table(self.state.new.signals, 'result.txt', sep='\t')
        exports.write_table(self.state.new.statistics, 'all_stat.txt', sep='\t')
        self.ui.label_info.setText('Готово!')
        self.ui.label_info.setStyleSheet('color:green')
        self.choice_signal()
        exports.write_table(self.state.new.signals, 'all_signals.txt', sep='\t')


    def calc_to_int_old(self):
        if self.state.old.signals.empty:
            return
        self.archive_menu.run_id = None
        """ Функция пересчета исходного сигнала, расчет цементограмм для старого оборудования"""


        auto_mean_win = preprocessing.prepare_interval(self.state.old, self.ui.doubleSpinBox_int_min.value(), self.ui.doubleSpinBox_int_max.value())
        self.ui.spinBox_mean_win_old.setValue(int(auto_mean_win))
        mean_win = self.ui.spinBox_mean_win_old.value()
        self.ui.progressBar.setMaximum(12)
        self.ui.label_info.setText('Расчет кривых старого оборудования...')
        self.ui.label_info.setStyleSheet('color:blue')

        parameters = self.read_parameters('old', mean_win)
        self.state.old.capture_original()
        calculate_old(self.state.old, parameters, self.ui.progressBar.setValue)
        self.state.old.result = ProcessingResult(parameters, self.state.old.revision)
        checkboxes = self.ui.old_device.findChildren(QCheckBox)
        for item in checkboxes:
            if item.isEnabled():
                n_izm = int(item.text()[0:2])
                item.setToolTip('Коэффициент затухания - ' + str(round(self.state.old.statistics['k_damp'][n_izm], 3)) +
                                 '\nЭффективная глубина - ' + str(round(self.state.old.statistics['useful_depth'][n_izm], 2)) +
                                 ' м\nМакс/Мин - ' + str(round(self.state.old.statistics['max_min'][n_izm], 3)))
                item.setStyleSheet(self.k_damp_detect('old', n_izm))
        self.calc_defect_to_tab(self.state.old.signals, True)
        self.ui.label_info.setText('Готово!')
        self.ui.label_info.setStyleSheet('color:green')
        exports.write_table(self.state.old.signals, 'result_old.txt', sep='\t')
        exports.write_table(self.state.old.statistics, 'all_stat_old.txt', sep='\t')
        self.choice_signal()
        exports.write_table(self.state.old.signals, 'all_signals_old.txt', sep='\t')


    def calc(self):
        """ Запуск пересчета параметров сигнала для существующих измерений """
        if self.ui.checkBox_signal1.isEnabled():
            self.calc_to_int()
        if self.ui.checkBox_signal_old_1.isEnabled():
            self.calc_to_int_old()


    def calc_defect_to_tab(self, device, uniq_qual):
        """ Сохранение интервалов дефектов и процентов совпадения дефектов в таблице """
        if device is self.state.old.signals:
            int_min_self = self.state.old.interval_min
            int_max_self = self.state.old.interval_max
            c_sig = self.create_list_sig('old')
        else:
            int_min_self = self.state.new.interval_min
            int_max_self = self.state.new.interval_max
            c_sig = self.create_list_sig('new')
        classification_errors = []
        if uniq_qual:
            mean_def1, mean_def2 = interpretation.individual_thresholds(
                device, c_sig, int_min_self, int_max_self,
                'new' if device is self.state.new.signals else 'old', self.ui.checkBox_abs_def.isChecked(), classification_errors)
            self.show_classification_errors(classification_errors)
            if device is self.state.new.signals:
                self.ui.doubleSpinBox_defect1_new.setValue(mean_def1)
                self.ui.doubleSpinBox_defect2_new.setValue(mean_def2)
            else:
                self.ui.doubleSpinBox_defect1_old.setValue(mean_def1)
                self.ui.doubleSpinBox_defect2_old.setValue(mean_def2)
        else:
            if device is self.state.new.signals:
                mean_def1 = self.ui.doubleSpinBox_defect1_new.value()
                mean_def2 = self.ui.doubleSpinBox_defect2_new.value()
            else:
                mean_def1 = self.ui.doubleSpinBox_defect1_old.value()
                mean_def2 = self.ui.doubleSpinBox_defect2_old.value()
        interpretation.common_thresholds(device, c_sig, int_min_self, int_max_self,
                                         mean_def1, mean_def2, self.ui.checkBox_abs_def.isChecked(), classification_errors)
        self.show_classification_errors(classification_errors)
        self.match_defect(device)
        self.choice_signal()


    def recalc_qual_new(self):
        """ Пересчет интервалов дефектов при изменении общих уровней дефектов для нового оборудования   """
        self.calc_defect_to_tab(self.state.new.signals, False)


    def recalc_qual_old(self):
        """ Пересчет интервалов дефектов при изменении общих уровней дефектов для старого оборудования   """
        self.calc_defect_to_tab(self.state.old.signals, False)


    def show_classification_errors(self, errors):
        if errors:
            self.ui.label_info.setText('Ошибка расчета интервалов дефектов. Поробуйте уменьшить максимальную глубину. '
                                       'Например на 1 метр.')
            self.ui.label_info.setStyleSheet('color:red')

    def calc_defect(self, input_list_1, defect1, defect2, intervals=None):
        try:
            return classify(input_list_1, defect1, defect2,
                            self.ui.checkBox_abs_def.isChecked(), intervals or QualityIntervals())
        except ClassificationError as exc:
            self.ui.label_info.setText('Ошибка расчета интервалов дефектов. Поробуйте уменьшить максимальную глубину. '
                                       'Например на 1 метр.')
            self.ui.label_info.setStyleSheet('color:red')
            return exc.quality


    def match_defect(self, device):
        """ Расчет процента совпадений дефектов """
        list_match_uniq = []
        list_match_mean = []
        if device is self.state.new.signals:
            list_sig = self.create_list_sig('new')
        else:
            list_sig = self.create_list_sig('old')
        self.remove_sum(device, list_sig)
        interpretation.match_defect(device, list_sig)


    def choice_signal(self):
        if hasattr(self, 'workspace_actions'):
            self.workspace_actions.refresh_tooltips()
        plotting.choice_signal(self.state, self.read_display_values((
            'checkBox_diff_res_pdf_new.checkState',
            'checkBox_diff_res_pdf_old.checkState',
            'checkBox_diff_result.checkState',
            'checkBox_diff_result_old.checkState',
            'checkBox_envelop.checkState',
            'checkBox_envelop_old.checkState',
            'checkBox_func.checkState',
            'checkBox_func_old.checkState',
            'checkBox_gen_def_new.checkState',
            'checkBox_gen_def_old.checkState',
            'checkBox_match_def_new.checkState',
            'checkBox_match_def_old.checkState',
            'checkBox_mean.checkState',
            'checkBox_mean_old.checkState',
            'checkBox_noise.checkState',
            'checkBox_noise.isEnabled',
            'checkBox_noise_old.checkState',
            'checkBox_noise_old.isEnabled',
            'checkBox_norm.checkState',
            'checkBox_norm_old.checkState',
            'checkBox_origin_sig.checkState',
            'checkBox_pdf.checkState',
            'checkBox_pdf_old.checkState',
            'checkBox_rel_ampl.checkState',
            'checkBox_rel_ampl_old.checkState',
            'checkBox_signal1.checkState',
            'checkBox_signal1.isEnabled',
            'checkBox_signal2.checkState',
            'checkBox_signal2.isEnabled',
            'checkBox_signal3.checkState',
            'checkBox_signal3.isEnabled',
            'checkBox_signal4.checkState',
            'checkBox_signal4.isEnabled',
            'checkBox_signal5.checkState',
            'checkBox_signal5.isEnabled',
            'checkBox_signal6.checkState',
            'checkBox_signal6.isEnabled',
            'checkBox_signal7.checkState',
            'checkBox_signal7.isEnabled',
            'checkBox_signal8.checkState',
            'checkBox_signal8.isEnabled',
            'checkBox_signal_old_1.checkState',
            'checkBox_signal_old_1.isEnabled',
            'checkBox_signal_old_10.checkState',
            'checkBox_signal_old_10.isEnabled',
            'checkBox_signal_old_11.checkState',
            'checkBox_signal_old_11.isEnabled',
            'checkBox_signal_old_12.checkState',
            'checkBox_signal_old_12.isEnabled',
            'checkBox_signal_old_2.checkState',
            'checkBox_signal_old_2.isEnabled',
            'checkBox_signal_old_3.checkState',
            'checkBox_signal_old_3.isEnabled',
            'checkBox_signal_old_4.checkState',
            'checkBox_signal_old_4.isEnabled',
            'checkBox_signal_old_5.checkState',
            'checkBox_signal_old_5.isEnabled',
            'checkBox_signal_old_6.checkState',
            'checkBox_signal_old_6.isEnabled',
            'checkBox_signal_old_7.checkState',
            'checkBox_signal_old_7.isEnabled',
            'checkBox_signal_old_8.checkState',
            'checkBox_signal_old_8.isEnabled',
            'checkBox_signal_old_9.checkState',
            'checkBox_signal_old_9.isEnabled',
            'checkBox_sum1.checkState',
            'checkBox_sum1.isEnabled',
            'checkBox_sum1_old.checkState',
            'checkBox_sum1_old.isEnabled',
            'checkBox_sum2.checkState',
            'checkBox_sum2.isEnabled',
            'checkBox_sum2_old.checkState',
            'checkBox_sum2_old.isEnabled',
            'checkBox_sum3.checkState',
            'checkBox_sum3.isEnabled',
            'checkBox_sum3_old.checkState',
            'checkBox_sum3_old.isEnabled',
            'doubleSpinBox_defect1_new.value',
            'doubleSpinBox_defect1_old.value',
            'doubleSpinBox_defect2_new.value',
            'doubleSpinBox_defect2_old.value',
            'doubleSpinBox_int_max.value',
        )), self.ui.graphicsView)


    def plot_graph(self, n_sig, sig_width, sig_dash, color):
        plotting.plot_graph(self.state, self.read_display_values((
            'checkBox_diff_result.checkState',
            'checkBox_envelop.checkState',
            'checkBox_func.checkState',
            'checkBox_mean.checkState',
            'checkBox_norm.checkState',
            'checkBox_origin_sig.checkState',
            'checkBox_rel_ampl.checkState',
            'doubleSpinBox_int_max.value',
        )), self.ui.graphicsView, n_sig, sig_width, sig_dash, color)


    def plot_graph_old(self, n_sig, sig_width, sig_dash, color):
        plotting.plot_graph_old(self.state, self.read_display_values((
            'checkBox_diff_result_old.checkState',
            'checkBox_envelop_old.checkState',
            'checkBox_func_old.checkState',
            'checkBox_mean_old.checkState',
            'checkBox_norm_old.checkState',
            'checkBox_rel_ampl_old.checkState',
            'doubleSpinBox_int_max.value',
        )), self.ui.graphicsView, n_sig, sig_width, sig_dash, color)


    def change_mean_win(self):
        """ пересчёт кривых при изменении коэффициентов """
        preprocessing.change_mean_win(self.state.new, self.read_parameters('new', self.ui.spinBox_mean_win.value()))
        exports.write_table(self.state.new.signals, 'result.txt', sep='\t')
        self.choice_signal()


    def change_mean_win_old(self):
        """ пересчёт кривых при изменении коэффициентов """
        preprocessing.change_mean_win_old(self.state.old, self.read_parameters('old', self.ui.spinBox_mean_win_old.value()))
        exports.write_table(self.state.old.signals, 'result_old.txt', sep='\t')
        self.choice_signal()


    def func1(self, x, a, b):
        return func1(x, a, b)


    def change_func(self):
        """ пересчет для сдвига функции затухания """
        preprocessing.change_func(self.state.new, self.read_parameters('new', self.ui.spinBox_mean_win.value()), self.ui.doubleSpinBox_kA.value(), self.ui.doubleSpinBox_kB.value())
        self.choice_signal()


    def change_func_old(self):
        """ пересчет для сдвига функции затухания """
        preprocessing.change_func_old(self.state.old, self.read_parameters('old', self.ui.spinBox_mean_win_old.value()), self.ui.doubleSpinBox_kA_old.value(), self.ui.doubleSpinBox_kB_old.value())
        self.choice_signal()


    def check_int_sig(self):
        """ проверка максимума и минимума интервала сигнала """
        self.ui.doubleSpinBox_int_min.setMaximum(self.ui.doubleSpinBox_int_max.value())
        self.ui.doubleSpinBox_int_max.setMinimum(self.ui.doubleSpinBox_int_min.value())


    def get_nearest_value(self, iterable, value):
        return get_nearest_value(iterable, value)


    def sum_signals(self, n_sum):
        self.invalidate('new')
        channels = [int(item.text()[0:2]) for item in self.ui.new_device.findChildren(QCheckBox)
                    if item.isEnabled() and int(item.text()[0:2]) < 9 and item.checkState() == 2]
        parameters = self.read_parameters('new', self.ui.spinBox_mean_win.value())
        try:
            result_sum = sums.sum_signals(self.state.new, parameters, channels, n_sum)
        except sums.SignalLengthError as exc:
            self.ui.label_info.setText(str(exc))
            self.ui.label_info.setStyleSheet('color:red')
            return False
        if result_sum:
            self.calc_defect_to_tab(self.state.new.signals, True)
            exports.write_table(self.state.new.signals, 'all_signals.txt', sep='\t')
            exports.write_table(self.state.new.statistics, 'all_stat.txt', sep='\t')
            self.choice_signal()
            self.ui.label_info.setText('Выполнено сложение сигналов.')
            self.ui.label_info.setStyleSheet('color:green')
            result_sum = True
        return result_sum


    def sum_signals_old(self, n_sum):
        self.invalidate('old')
        channels = [int(item.text()[0:2]) for item in self.ui.old_device.findChildren(QCheckBox)
                    if item.isEnabled() and int(item.text()[0:2]) < 13 and item.checkState() == 2]
        parameters = self.read_parameters('old', self.ui.spinBox_mean_win_old.value())
        try:
            result_sum = sums.sum_signals_old(self.state.old, parameters, channels, n_sum)
        except sums.SignalLengthError as exc:
            self.ui.label_info.setText(str(exc))
            self.ui.label_info.setStyleSheet('color:red')
            return False
        if result_sum:
            self.calc_defect_to_tab(self.state.old.signals, True)
            exports.write_table(self.state.old.signals, 'result_old.txt', sep='\t')
            exports.write_table(self.state.old.statistics, 'all_stat_old.txt', sep='\t')
            self.choice_signal()
            self.ui.label_info.setText('Выполнено сложение сигналов.')
            self.ui.label_info.setStyleSheet('color:green')
            result_sum = True
        return result_sum


    def sum1(self):
        """ Первый слот для суммы новое оборудование """
        self.ui.checkBox_sum1.setEnabled(False)
        self.ui.checkBox_sum1.setCheckState(0)
        self.ui.checkBox_sum1.setStyleSheet('background: #f0f0f0')
        self.ui.checkBox_sum1.setEnabled(True)
        if self.sum_signals(9):
            self.state.new.statistics['name'][9] = self.text_sum()
            self.ui.checkBox_sum1.setText('9 '+self.text_sum())
            self.ui.checkBox_sum1.setToolTip('Коэффициент затухания - ' + str(round(self.state.new.statistics['k_damp'][9], 3)) +
                                             '\nЭффективная глубина - ' + str(round(self.state.new.statistics['useful_depth'][9], 2)) +
                                             ' м\nМакс/Мин - ' + str(round(self.state.new.statistics['max_min'][9], 3)))
            self.ui.checkBox_sum1.setStyleSheet(self.k_damp_detect('new', 9))
        else:
            self.ui.checkBox_sum1.setEnabled(False)


    def sum2(self):
        """ Второй слот для суммы новое оборудование """
        self.ui.checkBox_sum2.setEnabled(False)
        self.ui.checkBox_sum2.setCheckState(0)
        self.ui.checkBox_sum2.setStyleSheet('background: #f0f0f0')
        self.ui.checkBox_sum2.setEnabled(True)
        if self.sum_signals(10):
            self.state.new.statistics['name'][10] = self.text_sum()
            self.ui.checkBox_sum2.setText('10 '+self.text_sum())
            self.ui.checkBox_sum2.setToolTip('Коэффициент затухания - ' + str(round(self.state.new.statistics['k_damp'][10], 3)) +
                                             '\nЭффективная глубина - ' + str(round(self.state.new.statistics['useful_depth'][10], 2)) +
                                             ' м\nМакс/Мин - ' + str(round(self.state.new.statistics['max_min'][10], 3)))
            self.ui.checkBox_sum2.setStyleSheet(self.k_damp_detect('new', 10))
        else:
            self.ui.checkBox_sum2.setEnabled(False)


    def sum3(self):
        """ Третий слот для суммы новое оборудование """
        self.ui.checkBox_sum3.setEnabled(False)
        self.ui.checkBox_sum3.setCheckState(0)
        self.ui.checkBox_sum3.setStyleSheet('background: #f0f0f0')
        self.ui.checkBox_sum3.setEnabled(True)
        if self.sum_signals(11):
            self.state.new.statistics['name'][11] = self.text_sum()
            self.ui.checkBox_sum3.setText('11 '+self.text_sum())
            self.ui.checkBox_sum3.setToolTip('Коэффициент затухания - ' + str(round(self.state.new.statistics['k_damp'][11], 3)) +
                                             '\nЭффективная глубина - ' + str(round(self.state.new.statistics['useful_depth'][11], 2)) +
                                             ' м\nМакс/Мин - ' + str(round(self.state.new.statistics['max_min'][11], 3)))
            self.ui.checkBox_sum3.setStyleSheet(self.k_damp_detect('new', 11))
        else:
            self.ui.checkBox_sum3.setEnabled(False)


    def sum1_old(self):
        """ Первый слот для суммы старое оборудование """
        self.ui.checkBox_sum1_old.setEnabled(False)
        self.ui.checkBox_sum1_old.setCheckState(0)
        self.ui.checkBox_sum1_old.setStyleSheet('background: #f0f0f0')
        self.ui.checkBox_sum1_old.setEnabled(True)
        if self.sum_signals_old(13):
            self.state.old.statistics['name'][13] = self.text_sum_old()
            self.ui.checkBox_sum1_old.setText('13 '+self.text_sum_old())
            self.ui.checkBox_sum1_old.setToolTip('Коэффициент затухания - ' + str(round(self.state.old.statistics['k_damp'][13], 3)) +
                                          '\nЭффективная глубина - ' + str(round(self.state.old.statistics['useful_depth'][13], 2)) +
                                          ' м\nМакс/Мин - ' + str(round(self.state.old.statistics['max_min'][13], 3)))
            self.ui.checkBox_sum1_old.setStyleSheet(self.k_damp_detect('old', 13))
        else:
            self.ui.checkBox_sum1_old.setEnabled(False)


    def sum2_old(self):
        """ Второй слот для суммы старое оборудование """
        self.ui.checkBox_sum2_old.setEnabled(False)
        self.ui.checkBox_sum2_old.setCheckState(0)
        self.ui.checkBox_sum2_old.setStyleSheet('background: #f0f0f0')
        self.ui.checkBox_sum2_old.setEnabled(True)
        if self.sum_signals_old(14):
            self.state.old.statistics['name'][14] = self.text_sum_old()
            self.ui.checkBox_sum2_old.setText('14 '+self.text_sum_old())
            self.ui.checkBox_sum2_old.setToolTip('Коэффициент затухания - ' + str(round(self.state.old.statistics['k_damp'][14], 3)) +
                                          '\nЭффективная глубина - ' + str(round(self.state.old.statistics['useful_depth'][14], 2)) +
                                          ' м\nМакс/Мин - ' + str(round(self.state.old.statistics['max_min'][14], 3)))
            self.ui.checkBox_sum2_old.setStyleSheet(self.k_damp_detect('old', 14))
        else:
            self.ui.checkBox_sum2_old.setEnabled(False)


    def sum3_old(self):
        """ Третий слот для суммы старое оборудование """
        self.ui.checkBox_sum3_old.setEnabled(False)
        self.ui.checkBox_sum3_old.setCheckState(0)
        self.ui.checkBox_sum3_old.setStyleSheet('background: #f0f0f0')
        self.ui.checkBox_sum3_old.setEnabled(True)
        if self.sum_signals_old(15):
            self.state.old.statistics['name'][15] = self.text_sum_old()
            self.ui.checkBox_sum3_old.setText('15 '+self.text_sum_old())
            self.ui.checkBox_sum3_old.setToolTip('Коэффициент затухания - ' + str(round(self.state.old.statistics['k_damp'][15], 3)) +
                                          '\nЭффективная глубина - ' + str(round(self.state.old.statistics['useful_depth'][15], 2)) +
                                          ' м\nМакс/Мин - ' + str(round(self.state.old.statistics['max_min'][15], 3)))
            self.ui.checkBox_sum3_old.setStyleSheet(self.k_damp_detect('old', 15))
        else:
            self.ui.checkBox_sum3_old.setEnabled(False)


    def calc_pdf(self):
        self.invalidate('new')
        self.ui.label_info.setText('Расчет функция плотности вероятности...')
        self.ui.label_info.setStyleSheet('color:blue')
        self.ui.checkBox_pdf.setCheckState(0)
        pdf_signal = []
        list_sig = []
        text = 'PDF '
        self.ui.progressBar.reset()
        self.ui.progressBar.setMaximum(len(self.state.new.signals['depth'])-1)
        checkboxes = self.ui.new_device.findChildren(QCheckBox)
        for item in checkboxes:
            if item.isEnabled() and int(item.text()[0:2]) not in [9, 10, 11] and item.checkState() == 2:
                n_i = int(item.text()[0:2])
                text = text + str(n_i) + ' '
                list_sig.append(n_i)
        if len(list_sig) > 1:
            pdf.envelope_new(self.state.new, list_sig, self.ui.progressBar.setValue)
            self.ui.checkBox_pdf.setText(text)
            self.ui.checkBox_pdf.setCheckState(2)

            pdf.curves_new(self.state.new, self.read_parameters('new', self.ui.spinBox_mean_win.value()))
            self.ui.label_info.setText('Готово!')
            self.ui.label_info.setStyleSheet('color:green')
            self.choice_signal()
            exports.write_table(self.state.new.signals, 'all_signals.txt', sep='\t')

            self.ui.label_info.setText('Расчитана функция плотности вероятности')
            self.ui.label_info.setStyleSheet('color:green')
        else:
            self.ui.label_info.setText('Нужно выбрать несколько измерений')
            self.ui.label_info.setStyleSheet('color:red')
        self.ui.pushButton_cem_pdf_new.setEnabled(True)


    def calc_pdf_old(self):
        self.invalidate('old')
        self.ui.label_info.setText('Расчет функция плотности вероятности...')
        self.ui.label_info.setStyleSheet('color:blue')
        self.ui.checkBox_pdf_old.setCheckState(0)
        pdf_signal = []
        list_sig = []
        text = 'PDF '
        self.ui.progressBar.reset()
        self.ui.progressBar.setMaximum(len(self.state.old.signals['depth'])-1)
        checkboxes = self.ui.old_device.findChildren(QCheckBox)
        for item in checkboxes:
            if item.isEnabled() and int(item.text()[0:2]) not in [13, 14, 15] and item.checkState() == 2:
                n_i = int(item.text()[0:2])
                text = text + str(n_i) + ' '
                list_sig.append(n_i)
        if len(list_sig) > 1:
            pdf.envelope_old(self.state.old, list_sig, self.ui.progressBar.setValue)
            self.ui.checkBox_pdf_old.setText(text)
            self.ui.checkBox_pdf_old.setCheckState(2)

            pdf.curves_old(self.state.old, self.read_parameters('old', self.ui.spinBox_mean_win_old.value()))
            self.ui.label_info.setText('Расчитана функция плотности вероятности')
            self.ui.label_info.setStyleSheet('color:green')
        else:
            self.ui.label_info.setText('Нужно выбрать несколько измерений')
            self.ui.label_info.setStyleSheet('color:red')
        self.ui.pushButton_cem_pdf_old.setEnabled(True)


    def text_sum(self):
        """ Текст для названия суммированного сигнала """
        text = 'New: '
        checkboxes = self.ui.new_device.findChildren(QCheckBox)
        for item in checkboxes:
            if item.checkState() == 2:
                n_izm = int(item.text()[0:2])
                if n_izm < 9:
                    text = text + str(n_izm) + ' '
        return text


    def text_sum_old(self):
        """ Текст для названия суммированного сигнала """
        text = 'Old: '
        checkboxes = self.ui.old_device.findChildren(QCheckBox)
        for item in checkboxes:
            if item.checkState() == 2:
                n_izm = int(item.text()[0:2])
                if n_izm < 13:
                    text = text + str(n_izm) + ' '
        return text


    def k_damp_detect(self, device, n_izm):
        color = interpretation.damping_class(getattr(self.state, device).statistics['k_damp'][n_izm])
        return 'background: ' + color if color is not None else None
        # if device == 'new':
        #     if all_stat['k_damp'][n_izm] < 0.65:
        #         return 'background: red'
        #     elif 0.65 <= all_stat['k_damp'][n_izm] < 0.82:
        #         return 'background: yellow'
        #     elif all_stat['k_damp'][n_izm] >= 0.82:
        #         return 'background: green'
        # elif device == 'old':
        #     if all_stat_old['k_damp'][n_izm] < 0.65:
        #         return 'background: red'
        #     elif 0.65 <= all_stat_old['k_damp'][n_izm] < 0.82:
        #         return 'background: yellow'
        #     elif all_stat_old['k_damp'][n_izm] >= 0.82:
        #         return 'background: green'


    def calc_int_cement(self):
        """ Выбор сигнала для цементограммы """
        selected = [item for group in (self.ui.new_device, self.ui.old_device)
                    for item in group.findChildren(QCheckBox) if item.isChecked()]
        if len(selected) != 1:
            self.ui.label_info.setText('Для преобразования выберите ровно один сигнал в разделе «Данные».')
            self.ui.workspace.select(0)
            self.ui.sidebarToggle.setChecked(True)
            return
        self.clear_interpretation()
        self.ui.doubleSpinBox_defect1.setEnabled(False)
        self.ui.doubleSpinBox_defect2.setEnabled(False)
        self.ui.doubleSpinBox_min_cement.setEnabled(False)
        self.ui.doubleSpinBox_max_cement.setEnabled(False)
        self.ui.name_cement.setText('измерение цементограммы')
        selected_new = [item for item in self.ui.new_device.findChildren(QCheckBox) if item.checkState() == 2]
        selected_old = [item for item in self.ui.old_device.findChildren(QCheckBox) if item.checkState() == 2]
        equipment = 'new' if selected_new else 'old'
        measurements = getattr(self.state, equipment)
        for item in selected_new or selected_old:
            number = int(item.text()[0:2])
            maximum = self.calc_max_int(measurements.signals, measurements.statistics, number) if equipment == 'new' else measurements.interval_max
            self.state.interpretation.table = interpretation.prepare_cement(measurements, number, maximum)
            self.ui.name_cement.setText(item.text())
            self.bind_interpretation(equipment, item.text())
            self.state.interpretation.damping = measurements.statistics['k_damp'][number]
            self.state.interpretation.frequency = interpretation.frequency(equipment, number)
        if self.state.interpretation.table is None:
            return
        self.state.interpretation.automatic_table = self.state.interpretation.table.copy(deep=True)

        self.ui.doubleSpinBox_undefine_max.setMaximum(self.state.interpretation.table['depth'].max())
        self.ui.label_int_undefine.setText('Интервалы неопр. цемента:')
        self.state.interpretation.intervals.undefined = []
        self.state.interpretation.intervals.good = []  # интервалы качественного цемента
        self.state.interpretation.intervals.partial = []  # интервалы частичного цемента
        self.state.interpretation.intervals.absent = []  # интервалы отсутствия цемента
        min_value = self.state.interpretation.table['first_sig'].min()
        max_value = self.state.interpretation.table['first_sig'].max()
        self.ui.doubleSpinBox_min_cement.setValue(min_value - (10 * (max_value - min_value) / 100))
        self.ui.doubleSpinBox_max_cement.setValue(max_value + (10 * (max_value - min_value) / 100))

        self.ui.doubleSpinBox_defect1.setValue(min_value + (max_value - min_value)/3)
        self.ui.doubleSpinBox_defect2.setValue(min_value + (2*(max_value - min_value)/3))
        self.ui.doubleSpinBox_x1_line.setValue(self.state.interpretation.table['depth'].min())
        self.ui.doubleSpinBox_x2_line.setValue(self.state.interpretation.table['depth'].min()+10)
        self.ui.doubleSpinBox_y_line.setValue(self.ui.doubleSpinBox_defect2.value())
        self.draw_cement()
        self.ui.pushButton_add_undefine.setEnabled(True)
        self.ui.doubleSpinBox_defect1.setEnabled(True)
        self.ui.doubleSpinBox_defect2.setEnabled(True)
        self.ui.doubleSpinBox_min_cement.setEnabled(True)
        self.ui.doubleSpinBox_max_cement.setEnabled(True)


    def cement_from_pdf_new(self):
        if self.state.new.signals.empty:
            return
        """ Выбор PDF сигнала для цементограммы """
        self.bind_interpretation('new', 'pdf')
        self.state.interpretation.table = interpretation.prepare_cement(self.state.new, 'pdf')
        self.state.interpretation.automatic_table = self.state.interpretation.table.copy(deep=True)

        self.ui.doubleSpinBox_undefine_max.setMaximum(self.state.interpretation.table['depth'].max())
        self.ui.label_int_undefine.setText('Интервалы неопр. цемента:')
        self.state.interpretation.intervals.undefined = []
        self.state.interpretation.intervals.good = []  # интервалы качественного цемента
        self.state.interpretation.intervals.partial = []  # интервалы частичного цемента
        self.state.interpretation.intervals.absent = []  # интервалы отсутствия цемента
        min_value = self.state.interpretation.table['first_sig'].min()
        max_value = self.state.interpretation.table['first_sig'].max()
        self.ui.doubleSpinBox_min_cement.setValue(min_value - (10 * (max_value - min_value) / 100))
        self.ui.doubleSpinBox_max_cement.setValue(max_value + (10 * (max_value - min_value) / 100))

        self.ui.doubleSpinBox_defect1.setValue(min_value + (max_value - min_value) / 3)
        self.ui.doubleSpinBox_defect2.setValue(min_value + (2 * (max_value - min_value) / 3))
        self.ui.doubleSpinBox_x1_line.setValue(self.state.interpretation.table['depth'].min())
        self.ui.doubleSpinBox_x2_line.setValue(self.state.interpretation.table['depth'].min() + 10)
        self.ui.doubleSpinBox_y_line.setValue(self.ui.doubleSpinBox_defect2.value())
        self.draw_cement()
        self.ui.pushButton_add_undefine.setEnabled(True)


    def cement_from_pdf_old(self):
        if self.state.old.signals.empty:
            return
        """ Выбор PDF сигнала для цементограммы """
        self.bind_interpretation('old', 'pdf')
        self.state.interpretation.table = interpretation.prepare_cement(self.state.old, 'pdf')
        self.state.interpretation.automatic_table = self.state.interpretation.table.copy(deep=True)

        self.ui.doubleSpinBox_undefine_max.setMaximum(self.state.interpretation.table['depth'].max())
        self.ui.label_int_undefine.setText('Интервалы неопр. цемента:')
        self.state.interpretation.intervals.undefined = []
        self.state.interpretation.intervals.good = []  # интервалы качественного цемента
        self.state.interpretation.intervals.partial = []  # интервалы частичного цемента
        self.state.interpretation.intervals.absent = []  # интервалы отсутствия цемента
        min_value = self.state.interpretation.table['first_sig'].min()
        max_value = self.state.interpretation.table['first_sig'].max()
        self.ui.doubleSpinBox_min_cement.setValue(min_value - (10 * (max_value - min_value) / 100))
        self.ui.doubleSpinBox_max_cement.setValue(max_value + (10 * (max_value - min_value) / 100))
        self.ui.doubleSpinBox_defect1.setValue(min_value + (max_value - min_value) / 3)
        self.ui.doubleSpinBox_defect2.setValue(min_value + (2 * (max_value - min_value) / 3))
        self.ui.doubleSpinBox_x1_line.setValue(self.state.interpretation.table['depth'].min())
        self.ui.doubleSpinBox_x2_line.setValue(self.state.interpretation.table['depth'].min() + 10)
        self.ui.doubleSpinBox_y_line.setValue(self.ui.doubleSpinBox_defect2.value())
        self.draw_cement()
        self.ui.pushButton_add_undefine.setEnabled(True)


    def draw_cement(self):
        plotting.draw_cement(self.state, self.read_display_values((
            'checkBox_corr_mode.checkState',
            'doubleSpinBox_defect1.value',
            'doubleSpinBox_defect2.value',
            'doubleSpinBox_x1_line.value',
            'doubleSpinBox_x2_line.value',
            'doubleSpinBox_y_line.value',
        )), self.ui.graphicsView)
        if hasattr(self, 'workspace_actions'):
            self.workspace_actions.refresh(show_cement=True)


    def corr_sig(self):
        if self.state.interpretation.table is None:
            return
        self.workspace_actions.checkpoint()
        """ Функция вытягивания кривой цементограммы - по максимуму """
        self.state.interpretation.table = corrections.corr_sig(
            self.state.interpretation.table, self.ui.doubleSpinBox_x1_line.value(),
            self.ui.doubleSpinBox_x2_line.value(), self.ui.doubleSpinBox_y_line.value(),
            self.ui.checkBox_shift_corr.isChecked(), self.ui.checkBox_lin_cor.isChecked())
        self.draw_cement()
        self.record_correction('corr_sig')


    def corr_sig_bottom(self):
        if self.state.interpretation.table is None:
            return
        self.workspace_actions.checkpoint()
        """ Функция вытягивания кривой цементограммы - по минимуму """
        self.state.interpretation.table = corrections.corr_sig_bottom(
            self.state.interpretation.table, self.ui.doubleSpinBox_x1_line.value(),
            self.ui.doubleSpinBox_x2_line.value(), self.ui.doubleSpinBox_y_line.value(),
            self.ui.checkBox_shift_corr.isChecked(), self.ui.checkBox_lin_cor.isChecked())
        self.draw_cement()
        self.record_correction('corr_sig_bottom')


    def fix_corr(self):
        if self.state.interpretation.table is None:
            return
        self.workspace_actions.checkpoint()
        """ зафиксировать изменения цементограммы для последующей коррекции """
        self.state.interpretation.table = corrections.fix_corr(self.state.interpretation.table)
        self.draw_cement()
        self.record_correction('fix_corr')


    def plus_func(self):
        if self.state.interpretation.table is None:
            return
        self.workspace_actions.checkpoint()
        """ Сложение цементограммы с функцией затухания - поднимает начало сигнала """
        self.state.interpretation.table = corrections.plus_func(self.state.interpretation.table)
        self.draw_cement()
        self.record_correction('plus_func')


    def add_undefine(self):
        if self.state.interpretation.table is None:
            return
        self.workspace_actions.checkpoint()
        """ добавить интервал цемента вручную """
        min_cem = self.ui.doubleSpinBox_undefine_min.value()
        max_cem = self.ui.doubleSpinBox_undefine_max.value()
        text_int = self.ui.label_int_undefine.text()
        cem = ''
        if self.ui.radioButton_KC.isChecked():
            cem = 'КЦ. '
            self.state.interpretation.intervals = corrections.add_interval(self.state.interpretation.intervals, self.state.interpretation.table['depth'], min_cem, max_cem, 'good')
        elif self.ui.radioButton_ChC.isChecked():
            cem = 'ЧЦ. '
            self.state.interpretation.intervals = corrections.add_interval(self.state.interpretation.intervals, self.state.interpretation.table['depth'], min_cem, max_cem, 'partial')
        elif self.ui.radioButton_OC.isChecked():
            cem = 'ОЦ. '
            self.state.interpretation.intervals = corrections.add_interval(self.state.interpretation.intervals, self.state.interpretation.table['depth'], min_cem, max_cem, 'absent')
        elif self.ui.radioButton_NC.isChecked():
            cem = 'НЦ. '
            self.state.interpretation.intervals = corrections.add_interval(self.state.interpretation.intervals, self.state.interpretation.table['depth'], min_cem, max_cem, 'undefined')
        text_int = text_int + '\n' + cem + str(min_cem) + ' - ' + str(max_cem) + ' м.'
        self.ui.label_int_undefine.setText(text_int)
        self.record_correction('add_undefine')


    def cementogramma(self, *, show=True):
        if self.state.interpretation.table is None:
            return
        """ построение цеметограммы в Matplotlib """
        min_cement = self.ui.doubleSpinBox_min_cement.value()
        max_cement = self.ui.doubleSpinBox_max_cement.value()
        defect1 = self.ui.doubleSpinBox_defect1.value()
        defect2 = self.ui.doubleSpinBox_defect2.value()
        self.state.interpretation.table['quality'] = self.calc_defect(self.state.interpretation.table['corr_sig'], defect1, defect2, self.state.interpretation.intervals)

        figures.cementogramma(self.state.interpretation.table, min_cement, max_cement, defect1, defect2, self.ui.name_cement.text(), show=show)

        exports.write_table(self.state.interpretation.table, 'result_cement.txt', sep='\t')

        # словарь настроек для сохранения
        self.state.interpretation.options = {}
        self.state.interpretation.options['doubleSpinBox_int_min'] = self.ui.doubleSpinBox_int_min.value()
        self.state.interpretation.options['doubleSpinBox_int_max'] = self.ui.doubleSpinBox_int_max.value()
        self.state.interpretation.options['doubleSpinBox_min_cement'] = min_cement
        self.state.interpretation.options['doubleSpinBox_max_cement'] = max_cement
        self.state.interpretation.options['doubleSpinBox_defect1'] = defect1
        self.state.interpretation.options['doubleSpinBox_defect2'] = defect2
        self.state.interpretation.options['name_signal'] = self.ui.name_cement.text()
        self.state.interpretation.options['signal'] = self.state.interpretation.table['corr_sig'].tolist()


    def check_int_corr(self):
        """ проверка интервала коррекции"""
        self.ui.doubleSpinBox_x1_line.setMaximum(self.ui.doubleSpinBox_x2_line.value())
        self.ui.doubleSpinBox_x2_line.setMinimum(self.ui.doubleSpinBox_x1_line.value())
        self.draw_cement()


    def check_int_defect(self):
        """ проверка интервала уровней дефектов"""
        self.ui.doubleSpinBox_defect1.setMaximum(self.ui.doubleSpinBox_defect2.value())
        self.ui.doubleSpinBox_defect2.setMinimum(self.ui.doubleSpinBox_defect1.value())
        self.draw_cement()


    def calc_maxmin_cem(self):
        if self.state.interpretation.table is None:
            return
        min_cem = self.ui.doubleSpinBox_min_cement.value()
        max_cem = self.ui.doubleSpinBox_max_cement.value()
        defect1 = self.ui.doubleSpinBox_defect1.value()
        defect2 = self.ui.doubleSpinBox_defect2.value()
        if (defect1 - min_cem) > (max_cem - defect2):
            self.ui.doubleSpinBox_max_cement.setValue(defect2 + (defect1 - min_cem))
            self.check_int_defect()
        elif (defect1 - min_cem) < (max_cem - defect2):
            self.ui.doubleSpinBox_min_cement.setValue(defect1 - (max_cem - defect2))
            self.check_int_defect()


    def check_int_defect_general(self):
        """ проверка интервала уровней дефектов"""
        self.ui.doubleSpinBox_defect1_new.setMaximum(self.ui.doubleSpinBox_defect2_new.value())
        self.ui.doubleSpinBox_defect2_new.setMinimum(self.ui.doubleSpinBox_defect1_new.value())
        self.ui.doubleSpinBox_defect1_old.setMaximum(self.ui.doubleSpinBox_defect2_old.value())
        self.ui.doubleSpinBox_defect2_old.setMinimum(self.ui.doubleSpinBox_defect1_old.value())
        self.choice_signal()


    def check_int_undefine(self):
        """ проверка интервала неопределенного цемента"""
        self.ui.doubleSpinBox_undefine_min.setMaximum(self.ui.doubleSpinBox_undefine_max.value())
        self.ui.doubleSpinBox_undefine_max.setMinimum(self.ui.doubleSpinBox_undefine_min.value())


    def cement_temp(self):
        # Работа с шаблоном Word
        doc = docx.Document(TEMPLATE_PATH)
        table = doc.tables[0]
        self.ui.lineEdit_pipe_diameter.setText(table.rows[1].cells[0].paragraphs[0].runs[3].text)
        self.ui.lineEdit_client.setText(table.rows[2].cells[0].paragraphs[0].runs[1].text)
        self.ui.lineEdit_client2.setText(table.rows[3].cells[0].paragraphs[0].text)
        self.ui.lineEdit_interpreter.setText(table.rows[9].cells[2].paragraphs[0].text)


    def export_report(self):
        if self.state.interpretation.table is None:
            return
        self.cementogramma(show=False)
        self.save_cement()


    def save_cement(self):
        if self.state.interpretation.table is None:
            return
        """ Сохранение результирующей цементограммы в отчет в документ Word по шаблону 'template.docx' """
        fields = []
        for widget in self.window.findChildren(QtWidgets.QWidget):
            if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QDateEdit)):
                fields.append((widget.objectName(), widget.text()))
            elif isinstance(widget, QtWidgets.QComboBox):
                fields.append((widget.objectName(), widget.currentText()))
        self.state.interpretation.report = ReportDetails(tuple(fields))
        doc = word.build_report(self.state.interpretation.table, self.state.interpretation.frequency, self.read_display_values((
            'checkBox_golfstrim.checkState',
            'comboBox_pipe_type.currentText',
            'dateEdit.text',
            'doubleSpinBox_int_max.value',
            'lineEdit_client.text',
            'lineEdit_client2.text',
            'lineEdit_interpreter.text',
            'lineEdit_number_skv.text',
            'lineEdit_pipe_diameter.text',
        )), TEMPLATE_PATH)
        try:
            exports.write_document(doc, 'цементограмма_по_скв_' + self.ui.lineEdit_number_skv.text() + '.docx')
            self.ui.label_info.setText('Результат сохранён в файл "цементограмма_по_скв_' + self.ui.lineEdit_number_skv.text() +
                                    '.docx"')
            self.ui.label_info.setStyleSheet('color:green')
        except PermissionError:
            self.ui.label_info.setText('Сохранение невозможно! Закройте файл "цементограмма_по_скв_' +
                                    self.ui.lineEdit_number_skv.text() + '.docx"')
            self.ui.label_info.setStyleSheet('color:red')

        legacy_options.save('options_' + self.ui.lineEdit_number_skv.text() + '.vac', self.state.interpretation.options)


    def draw_total_cement(self, device, n_izm, n_graph, fig, count_sig):
        figures.draw_total_cement(self.state, self.read_display_values((
            'checkBox_gen_def_new.checkState',
            'checkBox_gen_def_old.checkState',
            'doubleSpinBox_defect1_new.value',
            'doubleSpinBox_defect1_old.value',
            'doubleSpinBox_defect2_new.value',
            'doubleSpinBox_defect2_old.value',
        )), device, n_izm, n_graph, fig, count_sig)
        self.ui.label_info.setText('Готово!')
        self.ui.label_info.setStyleSheet('color:green')


    def draw_match_def(self, device, fig, count_sig):
        figures.draw_match_def(self.state, self.read_display_values((
            'checkBox_gen_def_new.checkState',
            'checkBox_gen_def_old.checkState',
        )), device, fig, count_sig)


    def all_cement(self):
        channels = [(self.state.new.signals, i) for i in self.create_list_sig('new')]
        figures.selected_cement(self.state, self.read_display_values((
            'checkBox_gen_def_new.checkState',
            'checkBox_gen_def_old.checkState',
            'doubleSpinBox_defect1_new.value',
            'doubleSpinBox_defect1_old.value',
            'doubleSpinBox_defect2_new.value',
            'doubleSpinBox_defect2_old.value',
        )), channels, match_device=self.state.new.signals)
        self.ui.label_info.setText('Готово!')
        self.ui.label_info.setStyleSheet('color:green')


    def all_cement_old(self):
        channels = [(self.state.old.signals, i) for i in self.create_list_sig('old')]
        figures.selected_cement(self.state, self.read_display_values((
            'checkBox_gen_def_new.checkState',
            'checkBox_gen_def_old.checkState',
            'doubleSpinBox_defect1_new.value',
            'doubleSpinBox_defect1_old.value',
            'doubleSpinBox_defect2_new.value',
            'doubleSpinBox_defect2_old.value',
        )), channels, match_device=self.state.old.signals)
        self.ui.label_info.setText('Готово!')
        self.ui.label_info.setStyleSheet('color:green')


    def draw_graphics(self):
        figures.draw_graphics(self.state, self.read_display_values((
            'checkBox_gen_def_new.checkState',
        )))


    def draw_graphics_old(self):
        figures.draw_graphics_old(self.state, self.read_display_values((
            'checkBox_gen_def_old.checkState',
        )))


    def choice_cement(self):
        """ Отрисовка всех выбранных цеметограмм на одном листе """
        choice_signal_new = []
        choice_signal_old = []
        checkboxes = self.ui.new_device.findChildren(QCheckBox)
        for item in checkboxes:
            if item.checkState() == 2:
                choice_signal_new.append(int(item.text()[0:2]))
        checkboxes = self.ui.old_device.findChildren(QCheckBox)
        for item in checkboxes:
            if item.checkState() == 2:
                choice_signal_old.append(int(item.text()[0:2]))
        channels = [(self.state.new.signals, i) for i in choice_signal_new] + [(self.state.old.signals, i) for i in choice_signal_old]
        figures.selected_cement(self.state, self.read_display_values((
            'checkBox_gen_def_new.checkState',
            'checkBox_gen_def_old.checkState',
            'doubleSpinBox_defect1_new.value',
            'doubleSpinBox_defect1_old.value',
            'doubleSpinBox_defect2_new.value',
            'doubleSpinBox_defect2_old.value',
        )), channels)
        self.ui.label_info.setText('Готово!')
        self.ui.label_info.setStyleSheet('color:green')


    def create_list_sig(self, device):
        """
        Функция возвращает список существующих сигналов для итераций по ним
        :param device: 'old' or 'new'
        :return: list_sig
        """
        list_sig = []
        if device == 'new':
            checkboxes = self.ui.new_device.findChildren(QCheckBox)
        elif device == 'old':
            checkboxes = self.ui.old_device.findChildren(QCheckBox)
        for item in checkboxes:
            n_i = int(item.text()[0:2])
            if item.isEnabled():
                list_sig.append(n_i)
        list_sig.sort()
        return list_sig


    def create_list_sig_check(self, device):
        """
        Функция возвращает список существующих и выбранных сигналов для итераций по ним
        :param device: 'old' or 'new'
        :return: list_sig
        """
        list_sig = []
        if device == 'new':
            checkboxes = self.ui.new_device.findChildren(QCheckBox)
        elif device == 'old':
            checkboxes = self.ui.old_device.findChildren(QCheckBox)
        for item in checkboxes:
            n_i = int(item.text()[0:2])
            if item.isEnabled() and item.checkState() == 2:
                list_sig.append(n_i)
        list_sig.sort()
        return list_sig


    def background_white(self):
        """переключение цвета фона графика"""
        if self.ui.checkBox_background_white.checkState() == 2:
            self.ui.graphicsView.setBackground('w')
        else:
            self.ui.graphicsView.setBackground('k')


    def open_options(self):
        """Загрузка настроек обработки"""
        checkboxes = self.ui.new_device.findChildren(QCheckBox)
        for item in checkboxes:
            item.setCheckState(0)
        checkboxes = self.ui.old_device.findChildren(QCheckBox)
        for item in checkboxes:
            item.setCheckState(0)
        self.ui.checkBox_corr_mode.setCheckState(0)

        file_name = QFileDialog.getOpenFileName(filter='*.vac')
        load_options = legacy_options.load_trusted(file_name[0])

        self.ui.doubleSpinBox_int_min.setValue(load_options['doubleSpinBox_int_min'])
        self.ui.doubleSpinBox_int_max.setValue(load_options['doubleSpinBox_int_max'])

        if load_options['name_signal'][3:8] == 'Old: ':
            sum_sig = load_options['name_signal'][8:-1].split(' ')
            for i, item in enumerate(sum_sig):
                sum_sig[i] = int(item)
            checkboxes = self.ui.old_device.findChildren(QCheckBox)
            for item in checkboxes:
                if int(item.text()[0:2]) in sum_sig:
                    item.setCheckState(2)
            self.sum1_old()
            self.ui.checkBox_sum1_old.setCheckState(2)
            for item in checkboxes:
                if int(item.text()[0:2]) in sum_sig:
                    item.setCheckState(0)
        elif load_options['name_signal'][3:8] == 'New: ':
            sum_sig = load_options['name_signal'][8:-1].split(' ')
            for i, item in enumerate(sum_sig):
                sum_sig[i] = int(item)
            checkboxes = self.ui.new_device.findChildren(QCheckBox)
            for item in checkboxes:
                if int(item.text()[0:2]) in sum_sig:
                    item.setCheckState(2)
            self.sum1()
            self.ui.checkBox_sum1.setCheckState(2)
            for item in checkboxes:
                if int(item.text()[0:2]) in sum_sig:
                    item.setCheckState(0)
        else:
            checkboxes = self.ui.new_device.findChildren(QCheckBox)
            for item in checkboxes:
                item.setCheckState(0)
                if item.text() == load_options['name_signal']:
                    item.setCheckState(2)
            checkboxes = self.ui.old_device.findChildren(QCheckBox)
            for item in checkboxes:
                item.setCheckState(0)
                if item.text() == load_options['name_signal']:
                    item.setCheckState(2)
        self.calc()
        self.calc_int_cement()
        self.state.interpretation.table['corr_sig'] = load_options['signal']
        self.ui.doubleSpinBox_min_cement.setValue(load_options['doubleSpinBox_min_cement'])
        self.ui.doubleSpinBox_max_cement.setValue(load_options['doubleSpinBox_max_cement'])
        self.ui.doubleSpinBox_defect1.setValue(load_options['doubleSpinBox_defect1'])
        self.ui.doubleSpinBox_defect2.setValue(load_options['doubleSpinBox_defect2'])
        self.ui.checkBox_corr_mode.setCheckState(2)


    def set_int(self, tab):
        """ Установка интервала исследование сигнала в спинбоксы + проверка """
        self.ui.doubleSpinBox_int_max.setMaximum(tab['depth'].max())  # устанавливаем в спинбоксы интервала сигнала
        self.ui.doubleSpinBox_int_min.setMaximum(tab['depth'].max())  # максимально допустимые значения из файла
        self.ui.doubleSpinBox_int_max.setValue(tab['depth'].max())  # устанавливаем в спинбоксы значения мин и макс
        self.ui.doubleSpinBox_int_min.setValue(tab['depth'].min())


    def calc_max_int(self, tab1, tab2, i):
        return calc_max_int(tab1, tab2, i, self.ui.doubleSpinBox_int_max.value())


    def check_all_new(self):
        """ Выбрать все сигналы """
        checkboxes = self.ui.new_device.findChildren(QCheckBox)
        if self.ui.checkBox_all_new.checkState() == 2:
            for item in checkboxes:
                if item.isEnabled() and int(item.text()[0:2]) not in [9, 10, 11]:
                    item.setCheckState(2)
        else:
            for item in checkboxes:
                if item.isEnabled():
                    item.setCheckState(0)


    def check_all_old(self):
        """ Выбрать все сигналы """
        checkboxes = self.ui.old_device.findChildren(QCheckBox)
        if self.ui.checkBox_all_old.checkState() == 2:
            for item in checkboxes:
                if item.isEnabled() and int(item.text()[0:2]) not in [13, 14, 15]:
                    item.setCheckState(2)
        else:
            for item in checkboxes:
                if item.isEnabled():
                    item.setCheckState(0)


    def remove_sum(self, device, list_sig):
        """ Удалить суммы из списка сигналов """
        if device is self.state.new.signals:
            if 9 in list_sig:
                list_sig.remove(9)
            if 10 in list_sig:
                list_sig.remove(10)
            if 11 in list_sig:
                list_sig.remove(11)
        elif device is self.state.old.signals:
            if 13 in list_sig:
                list_sig.remove(13)
            if 14 in list_sig:
                list_sig.remove(14)
            if 15 in list_sig:
                list_sig.remove(15)
        return list_sig


    def corr_matrix(self):
        """ строим корреляционную матрицу между сигналами """
        list_sig_new, list_sig_old = self.create_list_sig_check('new'), self.create_list_sig_check('old')
        if len(list_sig_new) > 0:
            tabl_corr = pd.DataFrame(self.state.new.signals['depth'].iloc[86:self.state.new.interval_max].copy())
            for i in list_sig_new:
                tabl_corr[str(i)+'new'] = self.state.new.signals[str(i)+'_diff_result'].iloc[86:self.state.new.interval_max].copy()
                tabl_corr[str(i)+'new'] = st.rankdata(tabl_corr[str(i)+'new'])
        if len(list_sig_old) > 0:
            if len(list_sig_new) == 0:
                tabl_corr = pd.DataFrame(self.state.old.signals['depth'].iloc[86:self.state.old.interval_max].copy())
            for i in list_sig_old:
                tabl_corr[str(i)+'old'] = self.state.old.signals[str(i)+'_diff_result'].iloc[86:self.state.old.interval_max].copy()
                tabl_corr[str(i)+'old'] = st.rankdata(tabl_corr[str(i)+'old'])
        try:
            tabl_corr = tabl_corr.drop(['depth'], axis=1)
        except UnboundLocalError:
            self.ui.label_info.setText('Для расчета корреляции сигналов нужно выбрать минимум 2 сигнала')
            self.ui.label_info.setStyleSheet('color:red')

        names = tabl_corr.columns.tolist()
        if len(names) > 1:
            figures.correlation(tabl_corr)
            self.ui.label_info.setText('Расчитана корреляции сигналов')
            self.ui.label_info.setStyleSheet('color:green')
        else:
            self.ui.label_info.setText('Для расчета корреляции сигналов нужно выбрать минимум 2 сигнала')
            self.ui.label_info.setStyleSheet('color:red')


    def mouseMoved(self, evt):
        ''' Отслеживаем координаты курсора '''
        pos = evt[0]
        vb = self.ui.graphicsView.getPlotItem().vb

        if self.ui.graphicsView.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            self.ui.label_x.setText(str(round(mousePoint.x(), 3)))
            self.ui.label_y.setText(str(round(mousePoint.y(), 3)))

