"""LVM reader. Directory traversal retains os.walk order and reverse file order."""
import os
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from ..domain.models import MeasurementSet
from ..processing.preprocessing import envelope

def read_lvm(path):
    signal = pd.read_table(path, delimiter='\t', header=None)
    if signal.empty or not {1, 2}.issubset(signal.columns):
        raise ValueError('LVM должен содержать глубину и сигнал в столбцах 1 и 2')
    if not all(pd.api.types.is_numeric_dtype(signal[i]) for i in (1, 2)):
        raise ValueError('Глубина и сигнал LVM должны быть числами')
    return signal


def load_directory(directory, mean, source_records=None):
    measurements = MeasurementSet('new')
    signals, statistics = measurements.signals, measurements.statistics
    files = []
    n = 1
    for root, dirs, names in os.walk(directory):
        for name in sorted(names, reverse=True):
            if name.endswith('_5150m_c.lvm'):
                files.append(name)
                path = Path(root) / name
                if source_records is None:
                    signal = read_lvm(path)
                else:
                    content = path.read_bytes()
                    signal = read_lvm(BytesIO(content))
                    source_records.append((path.relative_to(directory).as_posix(), content))
                signals['depth'] = signal[1]
                if n == 9:
                    signals['noise'] = np.abs(signal[2])
                else:
                    signals[n] = np.abs(signal[2])
                    signals[str(n) + '_envelope'] = envelope(signals['depth'], signals[n], mean)
                    statistics.at[n, 'name'] = name
                    statistics.at[n, 'max_h'] = signals['depth'].max()
                n += 1
    if len(files) != 9:
        raise ValueError('Ожидаются 9 файлов LVM: 8 измерений и шум')
    measurements.capture_original()
    return measurements, files
