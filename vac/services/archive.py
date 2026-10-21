"""Import and version services. Qt-free; loader order is an explicit part of provenance."""
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import logging
import pickle
import sqlite3

import numpy as np
from ..domain.models import MeasurementSet
from ..domain.state import ApplicationState
from ..io import lvm, twf
from ..processing.preprocessing import envelope
from ..storage import Source, InputChannel, Study, Run, Signal
from . import snapshot

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ImportReport:
    accepted: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)
    assignments: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    study_id: str | None = None


class ImportFailure(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('; '.join(report.errors))


def import_directory(archive, well_id, directory, equipment, mean=5):
    log.info('archive_import_started equipment=%s', equipment)
    directory = Path(directory)
    report = ImportReport()
    records = []
    try:
        if equipment not in ('new', 'old'):
            raise ValueError('Неизвестный тип оборудования')
        loaded, order = (lvm.load_directory(directory, mean, records) if equipment == 'new'
                         else twf.load_directory(directory, records))
        sources = [Source(name, 'lvm' if equipment == 'new' else 'twf', data) for name, data in records]
        report.accepted = [s.name for s in sources]
        report.skipped = [p.relative_to(directory).as_posix() for p in directory.rglob('*')
                          if p.is_file() and p.relative_to(directory).as_posix() not in report.accepted]
        for source in sources:
            matches = archive.find_sources(source)
            if matches:
                report.duplicates.append({'file': source.name, 'matches': matches})
        frame = loaded.original
        channels = []
        for column in frame:
            if column == 'depth':
                continue
            ordinal = (8 if equipment == 'new' else 0) if column == 'noise' else (
                int(column) - 1 if equipment == 'new' else int(column))
            channels.append(InputChannel(str(column), frame.depth.to_numpy(), frame[column].to_numpy(),
                role='noise' if column == 'noise' else 'signal', source_ordinal=ordinal,
                transformation={'loader': 'stage4-directory-v1', 'mean': mean,
                                'max_h': None if column == 'noise' else float(loaded.statistics.at[column, 'max_h'])}))
            report.assignments.append({'channel': str(column), 'role': channels[-1].role,
                                       'file': sources[ordinal].name, 'ordinal': ordinal})
        report.study_id = archive.save_study(well_id, Study(equipment, 'stage4-directory-v1', sources, channels))
    except (ValueError, OSError, KeyError, IndexError, sqlite3.Error) as exc:
        report.accepted = [name for name, _ in records]
        report.errors.append(str(exc))
        log.warning('archive_import_failed equipment=%s error_type=%s', equipment, type(exc).__name__)
        raise ImportFailure(report) from exc
    log.info('archive_import_complete equipment=%s sources=%d channels=%d duplicates=%d',
             equipment, len(sources), len(channels), len(report.duplicates))
    return report


def open_measurements(archive, study_id):
    study = archive.open_study(study_id)
    result = MeasurementSet(study.equipment)
    for c in study.channels:
        if result.signals.empty:
            result.signals['depth'] = c.axis
        elif not np.array_equal(result.signals.depth.to_numpy(), c.axis):
            raise ValueError('GUI требует общую ось каналов')
        name = 'noise' if c.role == 'noise' else int(c.name)
        result.signals[name] = c.values
        if name != 'noise':
            result.statistics.at[name, 'name'] = Path(study.sources[c.source_ordinal].name).name
            result.statistics.at[name, 'max_h'] = c.transformation['max_h']
            if study.equipment == 'new':
                result.signals[str(name) + '_envelope'] = envelope(result.signals.depth, result.signals[name], c.transformation['mean'])
    result.capture_original()
    return result


def code_hash():
    digest = sha256()
    for path in sorted(ROOT.rglob('*.py')):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_version(archive, study_id, state, widgets, provenance=None):
    study = archive.open_study(study_id)
    measurements = getattr(state, study.equipment)
    if measurements.result is None or measurements.result.stale:
        raise ValueError('Сначала выполните обработку с текущими параметрами')
    # A study cannot silently acquire the inputs of another GUI session.
    expected = open_measurements(archive, study_id).original
    import pandas as pd
    try:
        pd.testing.assert_frame_equal(measurements.original.loc[:, expected.columns], expected, check_like=True)
    except (AssertionError, KeyError) as exc:
        raise ValueError('Исходные измерения формы не совпадают с выбранным исследованием') from exc
    owned = ApplicationState()
    setattr(owned, study.equipment, deepcopy(measurements))
    current = state.interpretation
    if current.signal is not None and current.signal.equipment == study.equipment:
        if current.signal.revision != measurements.revision:
            raise ValueError('Интерпретация относится к другому набору измерений')
        owned.interpretation = deepcopy(current)
    metadata, arrays = snapshot.encode({'state': owned, 'widgets': widgets})
    frame = measurements.signals
    signals = [Signal(str(c), frame.depth.to_numpy(), frame[c].to_numpy())
               for c in frame if c != 'depth' and frame[c].dtype.kind in 'biufc']
    interp = owned.interpretation
    if interp.table is not None:
        signals.insert(0, Signal('!cement', interp.table.depth.to_numpy(), interp.table.corr_sig.to_numpy()))
    def table_arrays(table):
        return {} if table is None else {str(c): table[c].to_numpy() for c in table if table[c].dtype.kind in 'biufc'}
    params = {'widgets': widgets, 'processing': vars(measurements.result.parameters)}
    ids = [c['id'] for c in archive.get_channels(study_id)]
    statistics = {}
    for channel in archive.get_channels(study_id):
        if channel['role'] == 'signal':
            row = measurements.statistics.loc[int(channel['signal'].name)]
            statistics[channel['id']] = {name: (float(value), None) for name, value in row.items() if name != 'name'}
    version = code_hash()
    algorithm = 'legacy-phase-A:' + version
    reference = sha256((ROOT / 'reference.py').read_bytes()).hexdigest()
    if provenance is not None:
        if provenance['study_id'] != study_id:
            raise ValueError('Версия относится к другому исследованию')
        algorithm, reference = provenance['algorithm_version'], provenance['reference_hash']
    run = Run(ids, params, 'gui-state-v1', algorithm, reference, version, signals, statistics)
    identity = archive.save_gui_version(study_id, run, metadata, arrays,
        table_arrays(interp.automatic_table), table_arrays(interp.table),
        [(c.operation, dict(c.parameters)) for c in interp.corrections])
    log.info('archive_version_saved algorithm=%s program=%s reference=%s', algorithm, version, reference)
    return identity


def open_version(archive, run_id):
    metadata, arrays, interpretation_id = archive.open_gui_version(run_id)
    value = snapshot.decode(metadata, arrays)
    log.info('archive_version_opened snapshot_schema=gui-state-v1')
    return value['state'], value['widgets'], interpretation_id


def import_trusted_vac(archive, study_id, path):
    """Explicit trusted-source action; preserve only present fields, never invent raw data."""
    content = Path(path).read_bytes()
    options = pickle.loads(content)
    if not isinstance(options, dict):
        raise ValueError('Ожидается словарь настроек .vac')
    metadata, arrays = snapshot.encode(options)
    # Legacy lists contain JSON scalars; numeric ndarray fields are preserved in
    # the original, with an explicit limitation rather than an invented conversion.
    limitations = ['Нет исходных измерений, полной истории обработки, версий алгоритма и эталонов',
                   'Импорт не создаёт воспроизводимый запуск и не меняет открытую интерпретацию']
    if arrays:
        limitations.append('Массивы дополнительных полей доступны только в оригинале .vac')
    return archive.save_legacy_attachment(study_id, Path(path).name, content, metadata, limitations), limitations
