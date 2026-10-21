"""TWF: 130-byte header, little-endian signed int32. No sorting is introduced."""
import os
import struct
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from ..domain.models import MeasurementSet
from ..processing.preprocessing import twf_depth, interpolate_twf

@dataclass(frozen=True)
class TwfRecord:
    # Header fields are undocumented; preserve bytes without inferring units.
    header: bytes
    samples: tuple[int, ...]


def read_twf_record(path):
    with (BytesIO(path) if isinstance(path, bytes) else open(path, 'rb')) as stream:
        header = stream.read(130)
        payload = stream.read()
    if len(header) != 130 or not payload or len(payload) % 4:
        raise ValueError('TWF должен содержать заголовок 130 байт и непустой массив int32')
    length = len(payload) // 4
    return TwfRecord(header, struct.unpack('<' + str(length) + 'i', payload))


def read_twf(path):
    return read_twf_record(path).samples


def load_directory(directory, source_records=None):
    measurements = MeasurementSet('old')
    signals, statistics = measurements.signals, measurements.statistics
    files = []
    n = 1
    for name in os.listdir(directory):
        if name.endswith('.TWF'):
            files.append(name)
            if source_records is None:
                signal = read_twf(os.path.join(directory, name))
            else:
                content = (Path(directory) / name).read_bytes()
                signal = read_twf(content)
                source_records.append((name, content))
            maximum = len(signal) * 0.515
            if n == 1:
                signals['depth'] = twf_depth(len(signal))
                signals['noise'] = interpolate_twf(signal, signals['depth'])
            else:
                signals[n - 1] = interpolate_twf(signal, signals['depth'])
                statistics.at[n - 1, 'name'] = name
                statistics.at[n - 1, 'max_h'] = maximum
            n += 1
            if n == 14:
                break
    if len(files) != 13:
        raise ValueError('Ожидаются 13 файлов TWF: шум и 12 измерений')
    column = signals.columns[signals.loc[0] == signals.loc[0].max()].tolist()[0]
    scale = signals[column].max() / 5
    signals['noise'] = signals['noise'] / scale
    for i in range(1, len(signals.loc[0]) - 1):
        signals[i] = signals[i] / scale
    # Legacy directory import uses a local coefficient; single import keeps scale=1.
    measurements.capture_original()
    return measurements, files
