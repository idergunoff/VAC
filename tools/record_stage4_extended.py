"""Explicitly record small PDF/correction oracles from the stage-3 commit.

Requires Qt offscreen; writes candidates only to a required new directory.
Unit tests read the reviewed fixtures and never invoke this recorder.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from vac.app import create_window

BASE = '08088a1'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
source = subprocess.check_output(['git', 'show', BASE + ':vac/controller.py'], cwd=ROOT)
arrays = {}
metadata = {'source_commit': BASE, 'controller_sha256': hashlib.sha256(source).hexdigest(),
            'description': 'Stage-3 Controller on deterministic synthetic input; widget signals blocked.',
            'pdf': {}, 'corrections': {}}
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
with tempfile.TemporaryDirectory() as tmp:
    temporary = Path(tmp)
    oracle = temporary / 'controller.py'
    oracle.write_bytes(source)
    spec = importlib.util.spec_from_file_location('vac._stage3_oracle', oracle)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    window, current = create_window()
    controller = module.Controller(window, current.ui)
    for value in vars(controller.ui).values():
        if isinstance(value, QtCore.QObject):
            value.blockSignals(True)
    controller.choice_signal = lambda: None
    controller.draw_cement = lambda: None
    old_cwd = Path.cwd()
    os.chdir(temporary)
    try:
        depth = np.linspace(0, 20, 96)
        table = pd.DataFrame({'depth': depth})
        for n in [1, 2, 3]:
            table[n] = (5 + 0.15 * n) * np.exp(-depth / 13) + 0.08 * np.sin(depth * (n + 1))
        # Include singular KDE rows, matching the existing LinAlgError fallback.
        for n in [1, 2, 3]:
            table.loc[0, n] = 5
        arrays['input_pdf'] = table.to_numpy()
        for equipment in ['new', 'old']:
            measurements = getattr(controller.state, equipment)
            measurements.signals = table.copy(deep=True)
            measurements.interval_min, measurements.interval_max = 4, 94
            panel = getattr(controller.ui, equipment + '_device')
            for box in panel.findChildren(QtWidgets.QCheckBox):
                number = int(box.text()[:2])
                box.setEnabled(number in [1, 2, 3])
                box.setChecked(number in [1, 2, 3])
            suffix = '' if equipment == 'new' else '_old'
            getattr(controller.ui, 'spinBox_mean_win' + suffix).setValue(7)
            getattr(controller.ui, 'spinBox_coeff_dif_res' + suffix).setValue(2)
            getattr(controller, 'calc_pdf' + suffix)()
            columns = list(measurements.signals.columns)
            metadata['pdf'][equipment] = {'columns': columns, 'interval': [4, 94], 'window': 7, 'difference_scale': 2}
            arrays['pdf_' + equipment] = measurements.signals.to_numpy()
        curve = pd.DataFrame({'depth': np.arange(100, dtype=float),
                              'first_sig': np.sin(np.arange(100) / 8) + np.arange(100) / 100,
                              'func': np.exp(-np.arange(100) / 20)})
        curve['corr_sig'] = curve['first_sig'].copy()
        curve['corr_coeff'] = 1
        arrays['input_correction'] = curve.to_numpy()
        metadata['correction_columns'] = list(curve.columns)
        for operation in ['corr_sig', 'corr_sig_bottom']:
            for shift in [False, True]:
                for linear in [False, True]:
                    controller.state.interpretation.table = curve.copy(deep=True)
                    controller.ui.doubleSpinBox_x1_line.setValue(10)
                    controller.ui.doubleSpinBox_x2_line.setValue(50)
                    controller.ui.doubleSpinBox_y_line.setValue(0.5)
                    controller.ui.checkBox_shift_corr.setChecked(shift)
                    controller.ui.checkBox_lin_cor.setChecked(linear)
                    getattr(controller, operation)()
                    key = f'{operation}_{int(shift)}_{int(linear)}'
                    result = controller.state.interpretation.table
                    metadata['corrections'][key] = {'operation': operation, 'shift': shift, 'linear': linear,
                                                    'x1': 10, 'x2': 50, 'y': 0.5, 'columns': list(result.columns)}
                    arrays[key] = result.to_numpy()
    finally:
        os.chdir(old_cwd)
        window.close()
np.savez_compressed(output / 'arrays.npz', **arrays)
metadata['arrays_sha256'] = hashlib.sha256((output / 'arrays.npz').read_bytes()).hexdigest()
(output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
print('Recorded candidate:', output)
