"""KDE maximum on 1000 points; separate new/old envelope smoothing is retained."""
import numpy as np
from scipy import stats as st
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from ..calculation import func1

def envelope_new(measurements, channels, progress=lambda value: None):
    signals = measurements.signals
    list_sig = channels
    pdf_signal = []
    for row in range(len(signals['depth'])):
        sel = []
        for n_izm in list_sig:
            sel.append(signals[n_izm][row])
        signal_interval = np.linspace(np.min(sel), np.max(sel), 1000)
        try:
            pdf = st.gaussian_kde(sel)
        except np.linalg.LinAlgError:
            pdf_signal.append(sel[0])
            progress(row)
            continue
        pdf_func = pdf.evaluate(signal_interval)
        pdf_func_max_index = pdf_func.argmax()
        pdf_func_max = signal_interval[pdf_func_max_index]
        pdf_signal.append(pdf_func_max)
        progress(row)
    signals['pdf'] = pdf_signal
    n_max = ((np.diff(np.sign(np.diff(signals['pdf']))) < 0).nonzero()[0] + 1).tolist()
    n_max.insert(0, 0)
    n_max.append(int(len(signals['pdf'])) - 1)
    max_x = signals.loc[n_max, 'depth']  # выбираем x и y максимумов
    max_y = signals.loc[n_max, 'pdf']
    envelope = interp1d(max_x, max_y, kind='linear')  # выполняем интерполяцию по максимуму, получая огибающую
    signals['pdf_envelope'] = envelope(signals['depth'])
    signals['pdf_envelope'] = signals['pdf_envelope'].rolling(80, min_periods=1, center=True).mean()


def curves_new(measurements, parameters):
    signals = measurements.signals
    interval_min, interval_max = measurements.interval_min, measurements.interval_max
    mean_win = parameters.mean_window
    signals['pdf_envelope-1'] = signals['pdf_envelope'].shift(-1)
    signals['pdf_mean'] = signals['pdf_envelope'].rolling(mean_win, min_periods=1, center=True).mean()
    popt, pcov = curve_fit(func1, signals['depth'].iloc[interval_min:interval_max], signals['pdf_envelope'].iloc[interval_min:interval_max])
    signals['pdf_func'] = func1(signals['depth'], popt[0], popt[1]) + 0.1
    signals['pdf_rel_ampl'] = (signals['pdf_envelope-1'] - signals['pdf_func']) / signals['pdf_envelope-1']
    signals['pdf_mean-1'] = signals['pdf_mean'].shift(-1)
    signals['pdf_diff_norm'] = (signals['pdf_mean-1'] - signals['pdf_mean']) / signals['pdf_mean']
    signals['pdf_diff_norm-1'] = signals['pdf_diff_norm'].shift(-1)
    coeff_dif_res = parameters.difference_scale
    signals['pdf_diff_result'] = (signals['pdf_diff_norm-1'] + signals['pdf_diff_norm']) * coeff_dif_res
    signals.at[interval_max, 'pdf_diff_result'] = signals.at[interval_max - 2, 'pdf_diff_result']
    signals.at[interval_max - 1, 'pdf_diff_result'] = signals.at[interval_max - 2, 'pdf_diff_result']



def envelope_old(measurements, channels, progress=lambda value: None):
    signals = measurements.signals
    list_sig = channels
    pdf_signal = []
    for row in range(len(signals['depth'])):
        sel = []
        for n_izm in list_sig:
            sel.append(signals[n_izm][row])
        signal_interval = np.linspace(np.min(sel), np.max(sel), 1000)
        try:
            pdf = st.gaussian_kde(sel)
        except np.linalg.LinAlgError:
            pdf_signal.append(sel[0])
            progress(row)
            continue
        pdf_func = pdf.evaluate(signal_interval)
        pdf_func_max_index = pdf_func.argmax()
        pdf_func_max = signal_interval[pdf_func_max_index]
        pdf_signal.append(pdf_func_max)
        progress(row)
    signals['pdf_envelope'] = pdf_signal
    signals['pdf_envelope'] = signals['pdf_envelope'].rolling(10, min_periods=1, center=True).mean()


def curves_old(measurements, parameters):
    signals = measurements.signals
    interval_min, interval_max = measurements.interval_min, measurements.interval_max
    mean_win = parameters.mean_window
    signals['pdf_envelope-1'] = signals['pdf_envelope'].shift(-1)
    signals['pdf_mean'] = signals['pdf_envelope'].rolling(mean_win, min_periods=1, center=True).mean()
    popt, pcov = curve_fit(func1, signals['depth'].iloc[interval_min:interval_max],
                           signals['pdf_envelope'].iloc[interval_min:interval_max])
    signals['pdf_func'] = func1(signals['depth'], popt[0], popt[1]) + 0.1
    signals['pdf_rel_ampl'] = (signals['pdf_envelope-1'] - signals['pdf_func']) / signals[
        'pdf_envelope-1']
    signals['pdf_mean-1'] = signals['pdf_mean'].shift(-1)
    signals['pdf_diff_norm'] = (signals['pdf_mean-1'] - signals['pdf_mean']) / signals['pdf_mean']
    signals['pdf_diff_norm-1'] = signals['pdf_diff_norm'].shift(-1)
    coeff_dif_res = parameters.difference_scale
    signals['pdf_diff_result'] = (signals['pdf_diff_norm-1'] + signals['pdf_diff_norm']) * coeff_dif_res
    signals.at[interval_max, 'pdf_diff_result'] = signals.at[interval_max - 2, 'pdf_diff_result']
    signals.at[interval_max - 1, 'pdf_diff_result'] = signals.at[interval_max - 2, 'pdf_diff_result']



def calculate_pdf(measurements, parameters, channels, progress=lambda value: None):
    if len(channels) < 2:
        raise ValueError('Нужно выбрать несколько измерений')
    if measurements.equipment == 'new':
        envelope_new(measurements, channels, progress)
        curves_new(measurements, parameters)
    else:
        envelope_old(measurements, channels, progress)
        curves_old(measurements, parameters)
