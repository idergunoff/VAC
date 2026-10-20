"""Legacy print layouts. No window or widget reads; outputs retain legacy names."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from vac.io import exports
from vac.reference import good_new, good_old, bad_new, bad_old, depth_new, depth_old


def draw_total_cement(state, values, device, n_izm, n_graph, fig, count_sig):
    """ функция отрисовки цементограммы по всей длине """
    if device is state.old.signals:
        int_min_self = state.old.interval_min
        int_max_self = state.old.interval_max
    else:
        int_min_self = state.new.interval_min
        int_max_self = state.new.interval_max
    depth = device['depth'].iloc[int_min_self:int_max_self].tolist()
    curve = device[str(n_izm) + '_diff_result'].iloc[int_min_self:int_max_self].tolist()
    if device is state.new.signals:
        if values['checkBox_gen_def_new.checkState'] == 2:
            quality = np.array(device[str(n_izm) + '_qual_mean'].iloc[int_min_self:int_max_self].tolist())
        else:
            quality = np.array(device[str(n_izm) + '_quality'].iloc[int_min_self:int_max_self].tolist())
    elif values['checkBox_gen_def_old.checkState'] == 2:
        quality = np.array(device[str(n_izm) + '_qual_mean'].iloc[int_min_self:int_max_self].tolist())
    else:
        quality = np.array(device[str(n_izm) + '_quality'].iloc[int_min_self:int_max_self].tolist())
    max_depth = np.max(depth)
    min_depth = np.min(depth)
    min_value = np.min(curve)
    max_value = np.max(curve)
    min_cement = min_value - 10 * (max_value - min_value) / 100
    max_cement = max_value + 10 * (max_value - min_value) / 100
    if device is state.new.signals:
        if values['checkBox_gen_def_new.checkState'] == 2:
            defect1 = values['doubleSpinBox_defect1_new.value']
            defect2 = values['doubleSpinBox_defect2_new.value']
        else:
            defect1 = min_value + (max_value - min_value) / 3
            defect2 = min_value + 2 * (max_value - min_value) / 3
    elif values['checkBox_gen_def_old.checkState'] == 2:
        defect1 = values['doubleSpinBox_defect1_old.value']
        defect2 = values['doubleSpinBox_defect2_old.value']
    else:
        defect1 = min_value + (max_value - min_value) / 3
        defect2 = min_value + 2 * (max_value - min_value) / 3
    ax = fig.add_subplot(1, count_sig, n_graph)
    ax.axvline(x=defect1, linewidth=0.5, color='black', linestyle=':')
    ax.axvline(x=defect2, linewidth=0.5, color='black', linestyle=':')
    ax.fill_betweenx(depth, max_cement, curve, where=quality >= 1, hatch='//', facecolor='#EDEDED')
    ax.fill_betweenx(depth, max_cement, curve, where=quality >= 2, hatch='\\\\\\\\', facecolor='#BDBDBD')
    ax.fill_betweenx(depth, min_cement, curve, where=quality >= 1, hatch='//', facecolor='#EDEDED')
    ax.fill_betweenx(depth, min_cement, curve, where=quality >= 2, hatch='\\\\\\\\', facecolor='#BDBDBD')
    ax.plot(curve, depth, 'black')
    plt.ylim(min_depth, max_depth)
    plt.xlim(min_cement, max_cement)
    if device is state.new.signals:
        plt.title(state.new.statistics['name'][n_izm][-35:-23])
    if device is state.old.signals:
        plt.title(state.old.statistics['name'][n_izm])
    ax.get_xaxis().set_visible(False)
    plt.yticks(np.arange(min_depth, max_depth, 5))
    ax.grid(axis='y', color='black', linestyle=':', linewidth=0.5)
    ax.invert_yaxis()


def draw_match_def(state, values, device, fig, count_sig):
    """ функция отрисовки кривой совпадения дефектов по всей длине """
    if device is state.old.signals:
        int_min_self = state.old.interval_min
        int_max_self = state.old.interval_max
    else:
        int_min_self = state.new.interval_min
        int_max_self = state.new.interval_max
    depth = device['depth'].iloc[int_min_self:int_max_self].tolist()
    if device is state.new.signals:
        if values['checkBox_gen_def_new.checkState'] == 2:
            curve = state.new.signals['match_def_mean'].iloc[int_min_self:int_max_self].tolist()
        else:
            curve = state.new.signals['match_def_uniq'].iloc[int_min_self:int_max_self].tolist()
    elif values['checkBox_gen_def_old.checkState'] == 2:
        curve = state.old.signals['match_def_mean'].iloc[int_min_self:int_max_self].tolist()
    else:
        curve = state.old.signals['match_def_uniq'].iloc[int_min_self:int_max_self].tolist()
    max_depth = np.max(depth)
    min_depth = np.min(depth)
    min_value = np.min(curve)
    max_value = np.max(curve)
    min_cement = min_value - 10 * (max_value - min_value) / 100
    max_cement = max_value + 10 * (max_value - min_value) / 100
    ax = fig.add_subplot(1, count_sig, count_sig)
    ax.plot(curve, depth, 'black')
    plt.ylim(min_depth, max_depth)
    plt.xlim(min_cement, max_cement)
    plt.title('совп деф')
    ax.get_xaxis().set_visible(False)
    plt.yticks(np.arange(min_depth, max_depth, 5))
    ax.grid(axis='y', color='black', linestyle=':', linewidth=0.5)
    ax.invert_yaxis()


def draw_graphics(state, values):
    """ Отрисовка всех графиков исходного сигнала на одном листе новое оборудование """
    state.new.signals['sum5'] = (state.new.signals['1_envelope'] + state.new.signals['2_envelope'] + state.new.signals['5_envelope'] + state.new.signals['6_envelope']) / 4
    state.new.signals['sum2'] = (state.new.signals['3_envelope'] + state.new.signals['4_envelope'] + state.new.signals['7_envelope'] + state.new.signals['8_envelope']) / 4
    fig = plt.figure(figsize=(16.5, 11.7))
    ax = fig.add_subplot(5, 2, 1)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['1_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='551', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['5_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='552', color='black', linestyle='dashed', linewidth=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['1_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='551', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['5_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='552', color='black', linestyle='dashed', linewidth=2)
    plt.ylim(0, 2.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 3)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['2_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='531', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['6_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='532', color='black', linestyle='dashed', linewidth=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 4)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['2_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='531', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['6_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='532', color='black', linestyle='dashed', linewidth=2)
    plt.ylim(0, 2.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 5)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['3_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='251', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['7_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='252', color='black', linestyle='dashed', linewidth=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 6)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['3_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='251', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['7_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='252', color='black', linestyle='dashed', linewidth=2)
    plt.ylim(0, 2.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 7)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['4_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='231', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['8_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='232', color='black', linestyle='dashed', linewidth=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 8)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['4_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='231', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['8_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='232', color='black', linestyle='dashed', linewidth=2)
    plt.ylim(0, 2.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 9)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['sum5'].iloc[state.new.interval_min:state.new.interval_max], label='Сумма 5Гц', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['sum2'].iloc[state.new.interval_min:state.new.interval_max], label='Сумма 2Гц', color='black', linestyle='dashed', linewidth=2)
    ax.plot(depth_new, good_new, label='скв.2093 после ремонта', color='grey', linewidth=2.5)
    ax.plot(depth_new, bad_new, label='скв.2093 до ремонта', color='grey', linestyle='dashed', linewidth=2.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 10)
    if values['checkBox_gen_def_new.checkState'] == 2:
        ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['match_def_mean'].iloc[state.new.interval_min:state.new.interval_max], label='Совпадение дефектов')
    else:
        ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['match_def_uniq'].iloc[state.new.interval_min:state.new.interval_max], label='Совпадение дефектов')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    fig.suptitle('Новое оборудование')
    fig.tight_layout()
    exports.write_figure(fig, 'graphics_new1.png', dpi=200)
    fig.show()
    fig = plt.figure(figsize=(8.3, 11.7))
    ax = fig.add_subplot(5, 1, 1)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['1_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='551', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['5_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='552', color='black', linestyle='dashed', linewidth=2)
    plt.ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 1, 2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['2_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='531', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['6_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='532', color='black', linestyle='dashed', linewidth=2)
    plt.ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 1, 3)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['3_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='251', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['7_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='252', color='black', linestyle='dashed', linewidth=2)
    plt.ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 1, 4)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['4_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='231', color='black', linewidth=2)
    ax.plot(state.new.signals['depth'].iloc[state.new.interval_min:state.new.interval_max], state.new.signals['8_envelope'].iloc[state.new.interval_min:state.new.interval_max], label='232', color='black', linestyle='dashed', linewidth=2)
    plt.ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.new.signals['depth'][state.new.interval_min], state.new.signals['depth'][state.new.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    fig.suptitle('Новое оборудование_2')
    fig.tight_layout()
    exports.write_figure(fig, 'graphics_new2.png', dpi=200)
    fig.show()


def draw_graphics_old(state, values):
    """ Отрисовка всех графиков исходного сигнала на одном листе старое оборудование """
    state.old.signals['sum5'] = (state.old.signals[1] + state.old.signals[2] + state.old.signals[5] + state.old.signals[6] + state.old.signals[9] + state.old.signals[10]) / 6
    state.old.signals['sum2'] = (state.old.signals[3] + state.old.signals[4] + state.old.signals[7] + state.old.signals[8] + state.old.signals[11] + state.old.signals[12]) / 6
    fig = plt.figure(figsize=(16.5, 11.7))
    ax = fig.add_subplot(5, 2, 1)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[1].iloc[state.old.interval_min:state.old.interval_max], label='551', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[5].iloc[state.old.interval_min:state.old.interval_max], label='552', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[9].iloc[state.old.interval_min:state.old.interval_max], label='553', color='black', dashes=[1, 1], linewidth=1.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[1].iloc[state.old.interval_min:state.old.interval_max], label='551', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[5].iloc[state.old.interval_min:state.old.interval_max], label='552', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[9].iloc[state.old.interval_min:state.old.interval_max], label='553', color='black', dashes=[1, 1], linewidth=1.3)
    plt.ylim(0, 2.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 3)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[2].iloc[state.old.interval_min:state.old.interval_max], label='531', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[6].iloc[state.old.interval_min:state.old.interval_max], label='532', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[10].iloc[state.old.interval_min:state.old.interval_max], label='533', color='black', dashes=[1, 1], linewidth=1.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 4)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[2].iloc[state.old.interval_min:state.old.interval_max], label='531', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[6].iloc[state.old.interval_min:state.old.interval_max], label='532', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[10].iloc[state.old.interval_min:state.old.interval_max], label='533', color='black', dashes=[1, 1], linewidth=1.3)
    plt.ylim(0, 2.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 5)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[3].iloc[state.old.interval_min:state.old.interval_max], label='251', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[7].iloc[state.old.interval_min:state.old.interval_max], label='252', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[11].iloc[state.old.interval_min:state.old.interval_max], label='253', color='black', dashes=[1, 1], linewidth=1.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 6)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[3].iloc[state.old.interval_min:state.old.interval_max], label='251', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[7].iloc[state.old.interval_min:state.old.interval_max], label='252', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[11].iloc[state.old.interval_min:state.old.interval_max], label='253', color='black', dashes=[1, 1], linewidth=1.3)
    plt.ylim(0, 2.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 7)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[4].iloc[state.old.interval_min:state.old.interval_max], label='231', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[8].iloc[state.old.interval_min:state.old.interval_max], label='232', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[12].iloc[state.old.interval_min:state.old.interval_max], label='233', color='black', dashes=[1, 1], linewidth=1.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 8)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[4].iloc[state.old.interval_min:state.old.interval_max], label='231', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[8].iloc[state.old.interval_min:state.old.interval_max], label='232', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[12].iloc[state.old.interval_min:state.old.interval_max], label='233', color='black', dashes=[1, 1], linewidth=1.3)
    plt.ylim(0, 2.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 9)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals['sum5'].iloc[state.old.interval_min:state.old.interval_max], label='Сумма 5Гц', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals['sum2'].iloc[state.old.interval_min:state.old.interval_max], label='Сумма 2Гц', color='black', linestyle='dashed', linewidth=2)
    ax.plot(depth_old, good_old, label='скв.2093 после ремонта', color='grey', linewidth=2.5)
    ax.plot(depth_old, bad_old, label='скв.2093 до ремонта', color='grey', linestyle='dashed', linewidth=2.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 2, 10)
    if values['checkBox_gen_def_old.checkState'] == 2:
        ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals['match_def_mean'].iloc[state.old.interval_min:state.old.interval_max], label='Совпадение дефектов')
    else:
        ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals['match_def_uniq'].iloc[state.old.interval_min:state.old.interval_max], label='Совпадение дефектов')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    fig.suptitle('Старое оборудование')
    fig.tight_layout()
    exports.write_figure(fig, 'graphics_old1.png', dpi=200)
    fig.show()
    fig = plt.figure(figsize=(8.3, 11.7))
    ax = fig.add_subplot(5, 1, 1)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[1].iloc[state.old.interval_min:state.old.interval_max], label='551', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[5].iloc[state.old.interval_min:state.old.interval_max], label='552', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[9].iloc[state.old.interval_min:state.old.interval_max], label='553', color='black', dashes=[1, 1], linewidth=1.3)
    plt.ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 1, 2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[2].iloc[state.old.interval_min:state.old.interval_max], label='531', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[6].iloc[state.old.interval_min:state.old.interval_max], label='532', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[10].iloc[state.old.interval_min:state.old.interval_max], label='533', color='black', dashes=[1, 1], linewidth=1.3)
    plt.ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 1, 3)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[3].iloc[state.old.interval_min:state.old.interval_max], label='251', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[7].iloc[state.old.interval_min:state.old.interval_max], label='252', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[11].iloc[state.old.interval_min:state.old.interval_max], label='253', color='black', dashes=[1, 1], linewidth=1.3)
    plt.ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    ax = fig.add_subplot(5, 1, 4)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[4].iloc[state.old.interval_min:state.old.interval_max], label='231', color='black', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[8].iloc[state.old.interval_min:state.old.interval_max], label='232', color='black', linestyle='dashed', linewidth=2)
    ax.plot(state.old.signals['depth'].iloc[state.old.interval_min:state.old.interval_max], state.old.signals[12].iloc[state.old.interval_min:state.old.interval_max], label='233', color='black', dashes=[1, 1], linewidth=1.3)
    plt.ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.xlim(state.old.signals['depth'][state.old.interval_min], state.old.signals['depth'][state.old.interval_max])
    ax.tick_params(labelsize=8, labelrotation=90, axis='x', which='major')
    ax.legend()
    ax.grid(which='both')
    fig.suptitle('Старое оборудование_2')
    fig.tight_layout()
    exports.write_figure(fig, 'graphics_old2.png', dpi=200)
    fig.show()


def cementogramma(table, min_cement, max_cement, defect1, defect2, title):
    def draw_cement_int(ax, up):
        ax.axvline(x=defect1, linewidth=0.5, color='black', linestyle=':')
        ax.axvline(x=defect2, linewidth=0.5, color='black', linestyle=':')
        plt.fill_betweenx(table['depth'], max_cement, table['corr_sig'], where=table['quality'] >= 1,
                          hatch='//', facecolor='#EDEDED')
        plt.fill_betweenx(table['depth'], max_cement, table['corr_sig'], where=table['quality'] >= 2,
                          hatch='\\\\\\\\', facecolor='#BDBDBD')
        plt.fill_betweenx(table['depth'], max_cement, table['corr_sig'], where=table['quality'] >= 3,
                          hatch='...', facecolor='white', edgecolor='grey')
        plt.fill_betweenx(table['depth'], min_cement, table['corr_sig'], where=table['quality'] >= 1,
                          hatch='//', facecolor='#EDEDED')
        plt.fill_betweenx(table['depth'], min_cement, table['corr_sig'], where=table['quality'] >= 2,
                          hatch='\\\\\\\\', facecolor='#BDBDBD')
        plt.fill_betweenx(table['depth'], min_cement, table['corr_sig'], where=table['quality'] >= 3,
                          hatch='...', facecolor='white', edgecolor='grey')
        plt.plot(table['corr_sig'], table['depth'], 'black')
        plt.ylim(up, up + 100)
        plt.xlim(min_cement, max_cement)
        plt.title('ЧЦ', fontsize=9)
        plt.title('КЦ', loc='left', fontsize=9)
        plt.title('ОЦ', loc='right', fontsize=9)
        ax.get_xaxis().set_visible(False)
        plt.yticks(np.arange(up, up + 101, 5))
        plt.grid(axis='y', color='black', linestyle=':', linewidth=0.5)
        ax.invert_yaxis()

    fig = plt.figure(figsize=(6.3, 8.45))
    ax1 = plt.subplot(141)
    draw_cement_int(ax1, 0)
    ax2 = plt.subplot(142)
    draw_cement_int(ax2, 100)
    ax3 = plt.subplot(143)
    draw_cement_int(ax3, 200)
    ax4 = plt.subplot(144)
    draw_cement_int(ax4, 300)

    fig.tight_layout()
    exports.write_figure(fig, 'temp.png', dpi=200)
    fig.suptitle(title)
    fig.tight_layout()
    fig.show()

    if table['depth'].max() > 400:
        fig = plt.figure(figsize=(6.3, 8.45))
        ax1 = plt.subplot(141)
        draw_cement_int(ax1, 400)
        ax2 = plt.subplot(142)
        draw_cement_int(ax2, 500)
        ax3 = plt.subplot(143)
        draw_cement_int(ax3, 600)
        ax4 = plt.subplot(144)
        draw_cement_int(ax4, 700)

        fig.tight_layout()
        exports.write_figure(fig, 'temp1.png', dpi=200)
        fig.suptitle(title)
        fig.tight_layout()
        fig.show()

    if table['depth'].max() > 800:
        fig = plt.figure(figsize=(6.3, 8.45))
        ax1 = plt.subplot(141)
        draw_cement_int(ax1, 800)
        ax2 = plt.subplot(142)
        draw_cement_int(ax2, 900)
        ax3 = plt.subplot(143)
        draw_cement_int(ax3, 1000)
        ax4 = plt.subplot(144)
        draw_cement_int(ax4, 1100)

        fig.tight_layout()
        exports.write_figure(fig, 'temp2.png', dpi=200)
        fig.suptitle(title)
        fig.tight_layout()
        fig.show()




def selected_cement(state, values, channels, *, match_device=None, figsize=None):
    fig = plt.figure(figsize=figsize or (len(channels) * 1.5, 9))
    count = len(channels) + int(match_device is not None)
    for n, (device, channel) in enumerate(channels):
        draw_total_cement(state, values, device, channel, n + 1, fig, count)
    if match_device is not None:
        draw_match_def(state, values, match_device, fig, count)
    fig.tight_layout()
    fig.show()



def correlation(tabl_corr):
    names = tabl_corr.columns.tolist()
    corr_gist = []
    for i in tabl_corr.corr():
        corr_gist.append(tabl_corr.corr()[i].mean())

    fig = plt.figure(figsize=(21, 12), dpi=80)

    ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    sns.heatmap(tabl_corr.corr(), xticklabels=names, yticklabels=names, cmap='YlOrRd', annot=True, linewidths=0.25)
    plt.title('Корреляция сигналов', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax = plt.subplot2grid((1, 3), (0, 2))
    ax.barh(range(1, len(names)+1), corr_gist, height=1, align='edge', tick_label=names, color='#800026')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    plt.xlim(np.min(corr_gist) - 0.05, np.max(corr_gist)+0.05)
    plt.ylim(1, len(names)+1)
    ax.invert_yaxis()
    ax.grid()
    fig.tight_layout()
    fig.show()

