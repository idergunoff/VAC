"""Compare less common presentation paths with the actual stage-4 controller.

Requires git object f56fcc5 (the stage-5 base). Writes only to a fresh --output.
No fixture is updated; the oracle source is always read from that fixed commit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/unit')]
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets
from VAC_dialog import Ui_MainWindow
from vac.controller import Controller
from test_calculation import read_table

BASE = 'f56fcc56969d05cc3a8f20101dbd71fa04af5b10'


def digest(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def snapshot(controller):
    figures = []
    for n in plt.get_fignums():
        fig = plt.figure(n)
        axes = []
        for ax in fig.axes:
            axes.append({'limits': [*ax.get_xlim(), *ax.get_ylim()],
                         'title': [ax.get_title(loc=loc) for loc in ('left', 'center', 'right')],
                         'labels': [ax.get_xlabel(), ax.get_ylabel()],
                         'lines': [(digest(line.get_xdata()), digest(line.get_ydata()), line.get_label(),
                                    str(line.get_color()), line.get_linestyle(), line.get_linewidth()) for line in ax.lines]})
        fig.canvas.draw()
        figures.append({'axes': axes, 'pixels': hashlib.sha256(fig.canvas.buffer_rgba()).hexdigest()})
    curves = [(digest(curve.getData()[0]), digest(curve.getData()[1])) for curve in controller.ui.graphicsView.listDataItems()]
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.png')}
    return {'figures': figures, 'curves': curves, 'files': files,
            'status': [controller.ui.label_info.text(), controller.ui.label_info.styleSheet()]}


def prepare(controller):
    ui = controller.ui
    blockers = [QtCore.QSignalBlocker(w) for w in controller.window.findChildren(QtCore.QObject)]
    for equipment in ('new', 'old'):
        directory = ROOT / 'tests/baselines/stage-1' / equipment / '06-quality'
        suffix = '' if equipment == 'new' else '_old'
        settings = json.loads((directory / 'settings.json').read_text())
        for name, value in settings.items():
            widget = getattr(ui, name, None)
            if isinstance(widget, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
                widget.setMaximum(max(widget.maximum(), value)); widget.setValue(value)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(value['checked']); widget.setEnabled(value['enabled']); widget.setText(value['text'])
        dataset = getattr(controller.state, equipment)
        dataset.signals = read_table(directory, 'all_signals' + suffix)
        dataset.statistics = read_table(directory, 'all_stat' + suffix)
        depth = dataset.signals['depth']
        dataset.interval_min = int(np.argmin(abs(depth - 10)))
        dataset.interval_max = int(np.argmin(abs(depth - 150)))
    for equipment in ('new', 'old'):
        dataset = getattr(controller.state, equipment)
        group = getattr(ui, 'new_device' if equipment == 'new' else 'old_device')
        for widget in group.findChildren(QtWidgets.QCheckBox):
            channel = int(widget.text()[:2])
            widget.setEnabled(pd.notna(dataset.statistics.at[channel, 'name']))
            widget.setText(str(channel) + ' ' + str(dataset.statistics.at[channel, 'name']))
            widget.setChecked(channel in (1, 2))
    for name in ('checkBox_signal1', 'checkBox_signal2', 'checkBox_signal_old_1', 'checkBox_signal_old_2'):
        getattr(ui, name).setChecked(True)
    # Exercise all ordinary signal layers simultaneously.
    for name in ('checkBox_origin_sig', 'checkBox_envelop', 'checkBox_func', 'checkBox_mean',
                 'checkBox_norm', 'checkBox_rel_ampl', 'checkBox_diff_result', 'checkBox_envelop_old',
                 'checkBox_func_old', 'checkBox_mean_old', 'checkBox_norm_old',
                 'checkBox_rel_ampl_old', 'checkBox_diff_result_old'):
        getattr(ui, name).setChecked(True)
    del blockers


def run(controller_type, directory):
    directory.mkdir()
    previous = Path.cwd()
    os.chdir(directory)
    result = {}
    try:
        for general in (False, True):
            for operation in ('choice_signal', 'all_cement', 'all_cement_old', 'choice_cement',
                              'draw_graphics', 'draw_graphics_old', 'corr_matrix'):
                window = QtWidgets.QMainWindow(); ui = Ui_MainWindow(); ui.setupUi(window)
                controller = controller_type(window, ui)
                prepare(controller)
                blockers = [QtCore.QSignalBlocker(ui.checkBox_gen_def_new), QtCore.QSignalBlocker(ui.checkBox_gen_def_old)]
                ui.checkBox_gen_def_new.setChecked(general); ui.checkBox_gen_def_old.setChecked(general)
                del blockers
                for image in Path('.').glob('*.png'): image.unlink()
                plt.close('all')
                try:
                    with patch.object(plt.Figure, 'show'):
                        getattr(controller, operation)()
                    result[f'{operation}/{general}'] = snapshot(controller)
                finally:
                    plt.close('all'); window.close()
        return result
    finally:
        os.chdir(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if output.exists(): parser.error('output must be a new directory')
    source = subprocess.check_output(['git', 'show', BASE + ':vac/controller.py'], cwd=ROOT)
    oracle = types.ModuleType('vac._stage5_oracle')
    exec(compile(source, BASE + ':vac/controller.py', 'exec'), oracle.__dict__)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    output.mkdir(parents=True)
    expected = run(oracle.Controller, output / 'oracle')
    actual = run(Controller, output / 'actual')
    for name in expected:
        assert expected[name] == actual[name], name
    summary = {'base': BASE, 'controller_sha256': hashlib.sha256(source).hexdigest(),
               'scenarios': list(expected), 'status': 'PASS', 'comparison': 'axes, titles, lines, RGBA pixels, PNG bytes, pyqtgraph curves and status'}
    (output / 'validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print('PASS', len(expected), 'presentation scenarios against', BASE[:7])


if __name__ == '__main__':
    main()
