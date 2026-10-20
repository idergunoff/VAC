"""Interactive curves: explicit state, scalar display values, and a plot target."""
import pyqtgraph as pg
from vac.calculation import calc_max_int


def plot_graph_old(state, values, plot, n_sig, sig_width, sig_dash, color):
    """
        выбор и построение кривых для выбранных сигналов
        принимает параметры линии для построения
        """
    plot.showGrid(x=True, y=True)
    if values['checkBox_envelop_old.checkState'] == 2:
        int_max_old = calc_max_int(state.old.signals, state.old.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:int_max_old].tolist(), y=state.old.signals[n_sig].iloc[state.old.interval_min:int_max_old].tolist(), pen=pg.mkPen(color=(255, 0, color), width=sig_width, dash=sig_dash))
    if values['checkBox_func_old.checkState'] == 2:
        int_max_old = calc_max_int(state.old.signals, state.old.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:int_max_old].tolist(), y=state.old.signals[str(n_sig) + '_func'].iloc[state.old.interval_min:int_max_old].tolist(), pen=pg.mkPen(color=(0, 255, color), width=sig_width, dash=sig_dash))
    if values['checkBox_mean_old.checkState'] == 2:
        int_max_old = calc_max_int(state.old.signals, state.old.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:int_max_old].tolist(), y=state.old.signals[str(n_sig) + '_mean'].iloc[state.old.interval_min:int_max_old].tolist(), pen=pg.mkPen(color=(100, 100, 255 - color), width=sig_width, dash=sig_dash))
    if values['checkBox_norm_old.checkState'] == 2:
        int_max_old = calc_max_int(state.old.signals, state.old.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:int_max_old].tolist(), y=state.old.signals[str(n_sig) + '_norm'].iloc[state.old.interval_min:int_max_old].tolist(), pen=pg.mkPen(color=(255, 255, color), width=sig_width, dash=sig_dash))
    if values['checkBox_rel_ampl_old.checkState'] == 2:
        int_max_old = calc_max_int(state.old.signals, state.old.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:int_max_old].tolist(), y=state.old.signals[str(n_sig) + '_rel_ampl'].iloc[state.old.interval_min:int_max_old].tolist(), pen=pg.mkPen(color='c', width=sig_width, dash=sig_dash))
    if values['checkBox_diff_result_old.checkState'] == 2:
        int_max_old = calc_max_int(state.old.signals, state.old.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:int_max_old].tolist(), y=state.old.signals[str(n_sig) + '_diff_result'].iloc[state.old.interval_min:int_max_old].tolist(), pen=pg.mkPen(color=(255, 255, color), width=sig_width, dash=sig_dash))


def choice_signal(state, values, plot):
    """ выбор сигналов для построения """
    plot.clear()
    if values['checkBox_signal1.checkState'] == 2 and values['checkBox_signal1.isEnabled']:
        plot_graph(state, values, plot, 1, 1.8, [1], 0)
    if values['checkBox_signal2.checkState'] == 2 and values['checkBox_signal2.isEnabled']:
        plot_graph(state, values, plot, 2, 1.8, [3, 2], 0)
    if values['checkBox_signal3.checkState'] == 2 and values['checkBox_signal3.isEnabled']:
        plot_graph(state, values, plot, 3, 1.5, [6, 4], 0)
    if values['checkBox_signal4.checkState'] == 2 and values['checkBox_signal4.isEnabled']:
        plot_graph(state, values, plot, 4, 1.5, [10, 5, 1, 5], 0)
    if values['checkBox_signal5.checkState'] == 2 and values['checkBox_signal5.isEnabled']:
        plot_graph(state, values, plot, 5, 1.8, [1], 50)
    if values['checkBox_signal6.checkState'] == 2 and values['checkBox_signal6.isEnabled']:
        plot_graph(state, values, plot, 6, 1.8, [3, 2], 50)
    if values['checkBox_signal7.checkState'] == 2 and values['checkBox_signal7.isEnabled']:
        plot_graph(state, values, plot, 7, 1.5, [6, 4], 50)
    if values['checkBox_signal8.checkState'] == 2 and values['checkBox_signal8.isEnabled']:
        plot_graph(state, values, plot, 8, 1.5, [10, 5, 1, 5], 50)
    if values['checkBox_sum1.checkState'] == 2 and values['checkBox_sum1.isEnabled']:
        plot_graph(state, values, plot, 9, 2.5, [3, 3], 75)
    if values['checkBox_sum2.checkState'] == 2 and values['checkBox_sum2.isEnabled']:
        plot_graph(state, values, plot, 10, 2.5, [3, 3], 75)
    if values['checkBox_sum3.checkState'] == 2 and values['checkBox_sum3.isEnabled']:
        plot_graph(state, values, plot, 11, 2.5, [3, 3], 75)
    if values['checkBox_signal_old_1.checkState'] == 2 and values['checkBox_signal_old_1.isEnabled']:
        plot_graph_old(state, values, plot, 1, 1.8, [1], 100)
    if values['checkBox_signal_old_2.checkState'] == 2 and values['checkBox_signal_old_2.isEnabled']:
        plot_graph_old(state, values, plot, 2, 1.8, [3, 2], 100)
    if values['checkBox_signal_old_3.checkState'] == 2 and values['checkBox_signal_old_3.isEnabled']:
        plot_graph_old(state, values, plot, 3, 1.5, [6, 4], 100)
    if values['checkBox_signal_old_4.checkState'] == 2 and values['checkBox_signal_old_4.isEnabled']:
        plot_graph_old(state, values, plot, 4, 1.5, [10, 5, 1, 5], 100)
    if values['checkBox_signal_old_5.checkState'] == 2 and values['checkBox_signal_old_5.isEnabled']:
        plot_graph_old(state, values, plot, 5, 1.8, [1], 150)
    if values['checkBox_signal_old_6.checkState'] == 2 and values['checkBox_signal_old_6.isEnabled']:
        plot_graph_old(state, values, plot, 6, 1.8, [3, 2], 150)
    if values['checkBox_signal_old_7.checkState'] == 2 and values['checkBox_signal_old_7.isEnabled']:
        plot_graph_old(state, values, plot, 7, 1.5, [6, 4], 150)
    if values['checkBox_signal_old_8.checkState'] == 2 and values['checkBox_signal_old_8.isEnabled']:
        plot_graph_old(state, values, plot, 8, 1.5, [10, 5, 1, 5], 150)
    if values['checkBox_signal_old_9.checkState'] == 2 and values['checkBox_signal_old_9.isEnabled']:
        plot_graph_old(state, values, plot, 9, 1.8, [1], 200)
    if values['checkBox_signal_old_10.checkState'] == 2 and values['checkBox_signal_old_10.isEnabled']:
        plot_graph_old(state, values, plot, 10, 1.8, [3, 2], 200)
    if values['checkBox_signal_old_11.checkState'] == 2 and values['checkBox_signal_old_11.isEnabled']:
        plot_graph_old(state, values, plot, 11, 1.5, [6, 4], 200)
    if values['checkBox_signal_old_12.checkState'] == 2 and values['checkBox_signal_old_12.isEnabled']:
        plot_graph_old(state, values, plot, 12, 1.5, [10, 5, 1, 5], 200)
    if values['checkBox_sum1_old.checkState'] == 2 and values['checkBox_sum1_old.isEnabled']:
        plot_graph_old(state, values, plot, 13, 2.5, [3, 3], 175)
    if values['checkBox_sum2_old.checkState'] == 2 and values['checkBox_sum2_old.isEnabled']:
        plot_graph_old(state, values, plot, 14, 2.5, [3, 3], 175)
    if values['checkBox_sum3_old.checkState'] == 2 and values['checkBox_sum3_old.isEnabled']:
        plot_graph_old(state, values, plot, 15, 2.5, [3, 3], 175)
    if values['checkBox_noise.checkState'] == 2 and values['checkBox_noise.isEnabled']:
        plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max].tolist(), y=state.new.signals['noise'].iloc[state.new.interval_min:state.new.interval_max].tolist(), pen=pg.mkPen(color='c', dash=[2, 1, 4], width=1))
    if values['checkBox_noise_old.checkState'] == 2 and values['checkBox_noise_old.isEnabled']:
        plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max].tolist(), y=state.old.signals['noise'].iloc[state.old.interval_min:state.old.interval_max].tolist(), pen=pg.mkPen(color='c', dash=[2, 1, 4], width=1))
    if values['checkBox_pdf.checkState'] == 2:
        plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max].tolist(), y=state.new.signals['pdf_envelope'].iloc[state.new.interval_min:state.new.interval_max].tolist(), pen=pg.mkPen(color='#9CCC00', width=2.5, dash=[2, 2]))
    if values['checkBox_pdf_old.checkState'] == 2:
        plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max].tolist(), y=state.old.signals['pdf_envelope'].iloc[state.old.interval_min:state.old.interval_max].tolist(), pen=pg.mkPen(color='#11CC00', width=2.5, dash=[2, 2]))
    if values['checkBox_diff_res_pdf_new.checkState'] == 2:
        plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max].tolist(), y=state.new.signals['pdf_diff_result'].iloc[state.new.interval_min:state.new.interval_max].tolist(), pen=pg.mkPen(color='#DDFE71', width=2.5, dash=[2, 2]))
    if values['checkBox_diff_res_pdf_old.checkState'] == 2:
        plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max].tolist(), y=state.old.signals['pdf_diff_result'].iloc[state.old.interval_min:state.old.interval_max].tolist(), pen=pg.mkPen(color='#FEE771', width=2.5, dash=[2, 2]))
    if values['checkBox_match_def_new.checkState'] == 2:
        if values['checkBox_gen_def_new.checkState'] == 2:
            plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max].tolist(), y=state.new.signals['match_def_mean'].iloc[state.new.interval_min:state.new.interval_max].tolist(), pen=pg.mkPen(color='#1FFFEF', width=2.5, dash=[2, 2]))
        else:
            plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max].tolist(), y=state.new.signals['match_def_uniq'].iloc[state.new.interval_min:state.new.interval_max].tolist(), pen=pg.mkPen(color='#1FFFEF', width=2.5, dash=[2, 2]))
    if values['checkBox_match_def_old.checkState'] == 2:
        if values['checkBox_gen_def_old.checkState'] == 2:
            plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max].tolist(), y=state.old.signals['match_def_mean'].iloc[state.old.interval_min:state.old.interval_max].tolist(), pen=pg.mkPen(color='#1FFFB6', width=2.5, dash=[2, 2]))
        else:
            plot.plot(x=state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max].tolist(), y=state.old.signals['match_def_uniq'].iloc[state.old.interval_min:state.old.interval_max].tolist(), pen=pg.mkPen(color='#1FFFB6', width=2.5, dash=[2, 2]))
    if values['checkBox_diff_result.checkState'] == 2 and values['checkBox_gen_def_new.checkState'] == 2:
        defect1 = values['doubleSpinBox_defect1_new.value']
        defect2 = values['doubleSpinBox_defect2_new.value']
        d1 = pg.InfiniteLine(pos=defect1, angle=0, pen=pg.mkPen(color='#E59D38', width=0.9, dash=[5, 5]))
        d2 = pg.InfiniteLine(pos=defect2, angle=0, pen=pg.mkPen(color='#FF1705', width=0.9, dash=[5, 5]))
        plot.addItem(d1)
        plot.addItem(d2)
    if values['checkBox_diff_result_old.checkState'] == 2 and values['checkBox_gen_def_old.checkState'] == 2:
        defect1 = values['doubleSpinBox_defect1_old.value']
        defect2 = values['doubleSpinBox_defect2_old.value']
        d1 = pg.InfiniteLine(pos=defect1, angle=0, pen=pg.mkPen(color='#B8FF29', width=0.9, dash=[5, 5]))
        d2 = pg.InfiniteLine(pos=defect2, angle=0, pen=pg.mkPen(color='#FF0597', width=0.9, dash=[5, 5]))
        plot.addItem(d1)
        plot.addItem(d2)


def draw_cement(state, values, plot):
    if state.interpretation.table is None:
        return
    ' отрисовка цементограммы в pyqtgraph (в окне программы обработки) '
    plot.showGrid(x=False, y=False)
    defect1 = values['doubleSpinBox_defect1.value']
    defect2 = values['doubleSpinBox_defect2.value']
    plot.clear()
    plot.plot(x=state.interpretation.table['depth'], y=state.interpretation.table['corr_sig'], pen=pg.mkPen(color='#B4FEAE', width=2, dash=[2, 2]))
    d1 = pg.InfiniteLine(pos=defect1, angle=0, pen=pg.mkPen(color='#E59D38', width=0.9, dash=[5, 5]))
    d2 = pg.InfiniteLine(pos=defect2, angle=0, pen=pg.mkPen(color='#FF1705', width=0.9, dash=[5, 5]))
    plot.addItem(d1)
    plot.addItem(d2)
    if values['checkBox_corr_mode.checkState'] == 2:
        plot.plot(x=state.interpretation.table['depth'], y=state.interpretation.table['first_sig'], pen=pg.mkPen(color='#FF3D70', width=2, dash=[8, 3, 2, 3]))
        line_x1 = values['doubleSpinBox_x1_line.value']
        line_x2 = values['doubleSpinBox_x2_line.value']
        line_y = values['doubleSpinBox_y_line.value']
        lx1 = pg.InfiniteLine(pos=line_x1, angle=90, pen=pg.mkPen(width=0.7, dash=[8, 2]))
        lx2 = pg.InfiniteLine(pos=line_x2, angle=90, pen=pg.mkPen(width=0.7, dash=[8, 2]))
        ly = pg.InfiniteLine(pos=line_y, angle=0, pen=pg.mkPen(width=0.7, dash=[8, 2]))
        plot.addItem(lx1)
        plot.addItem(lx2)
        plot.addItem(ly)


def plot_graph(state, values, plot, n_sig, sig_width, sig_dash, color):
    """
        выбор и построение кривых для выбранных сигналов
        принимает параметры линии для построения
        """
    plot.showGrid(x=True, y=True)
    if values['checkBox_origin_sig.checkState'] == 2:
        int_max = calc_max_int(state.new.signals, state.new.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:int_max].tolist(), y=state.new.signals[n_sig].iloc[state.new.interval_min:int_max].tolist(), pen=pg.mkPen(width=sig_width, dash=sig_dash))
    if values['checkBox_envelop.checkState'] == 2:
        int_max = calc_max_int(state.new.signals, state.new.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:int_max].tolist(), y=state.new.signals[str(n_sig) + '_envelope'].iloc[state.new.interval_min:int_max].tolist(), pen=pg.mkPen(color=(255, 0, color), width=sig_width, dash=sig_dash))
    if values['checkBox_func.checkState'] == 2:
        int_max = calc_max_int(state.new.signals, state.new.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:int_max].tolist(), y=state.new.signals[str(n_sig) + '_func'].iloc[state.new.interval_min:int_max].tolist(), pen=pg.mkPen(color=(0, 255, color), width=sig_width, dash=sig_dash))
    if values['checkBox_mean.checkState'] == 2:
        int_max = calc_max_int(state.new.signals, state.new.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:int_max].tolist(), y=state.new.signals[str(n_sig) + '_mean'].iloc[state.new.interval_min:int_max].tolist(), pen=pg.mkPen(color=(100, 100, 255 - color), width=sig_width, dash=sig_dash))
    if values['checkBox_norm.checkState'] == 2:
        int_max = calc_max_int(state.new.signals, state.new.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:int_max].tolist(), y=state.new.signals[str(n_sig) + '_norm'].iloc[state.new.interval_min:int_max].tolist(), pen=pg.mkPen(color=(255, 255, color), width=sig_width, dash=sig_dash))
    if values['checkBox_rel_ampl.checkState'] == 2:
        int_max = calc_max_int(state.new.signals, state.new.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:int_max].tolist(), y=state.new.signals[str(n_sig) + '_rel_ampl'].iloc[state.new.interval_min:int_max].tolist(), pen=pg.mkPen(color='c', width=sig_width, dash=sig_dash))
    if values['checkBox_diff_result.checkState'] == 2:
        int_max = calc_max_int(state.new.signals, state.new.statistics, n_sig, values['doubleSpinBox_int_max.value'])
        plot.plot(x=state.new.signals['depth'].iloc[state.new.interval_min:int_max].tolist(), y=state.new.signals[str(n_sig) + '_diff_result'].iloc[state.new.interval_min:int_max].tolist(), pen=pg.mkPen(color=(255, 255, color), width=sig_width, dash=sig_dash))
