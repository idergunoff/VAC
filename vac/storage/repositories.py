"""Append-only archive operations; every aggregate is written atomically."""
from datetime import datetime, timezone
import json
import math
from uuid import uuid4

import numpy as np

from . import codec
from .models import Source, Signal, InputChannel, Study, Run, Correction, Interval, Interpretation
from .session import connect, migrate, transaction


def _id():
    return uuid4().hex


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _required(*values):
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError('Required nonempty metadata')


class Archive:
    def __init__(self, path):
        self._db = connect(path)
        try:
            migrate(self._db)
        except BaseException:
            self._db.close()
            raise

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _row(self, table, identity):
        # Table identifiers are internal constants, never caller-supplied SQL.
        row = self._db.execute(f'SELECT * FROM {table} WHERE id=?', (identity,)).fetchone()
        if row is None:
            raise KeyError(identity)
        return dict(row)

    def _array(self, array):
        a, data = codec.encode(array)
        identity = _id()
        self._db.execute('INSERT INTO numeric_array VALUES(?,?,?,?,?,?,?)',
                         (identity, 'npy-v1', a.dtype.str, _json(list(a.shape)), len(data), codec.digest(data), data))
        return identity

    def _read_array(self, identity):
        row = self._row('numeric_array', identity)
        if row['codec'] != 'npy-v1' or row['size'] != len(row['payload']):
            raise ValueError('Invalid array record')
        return codec.decode(row['payload'], row['dtype'], json.loads(row['shape']), row['sha256'])

    def _signal(self, signal):
        a, v = np.asarray(signal.axis), np.asarray(signal.values)
        if a.ndim != 1 or v.ndim != 1 or not len(a) or len(a) != len(v):
            raise ValueError('Signal and axis must have matching nonempty 1D shapes')
        if a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
            raise ValueError('Axis must be finite real coordinates')
        axis_id = _id()
        self._db.execute('INSERT INTO axis VALUES(?,?,?,?)',
                         (axis_id, self._array(a), signal.axis_meaning, signal.axis_units))
        return axis_id, self._array(v)

    def _read_signal(self, row):
        axis = self._row('axis', row['axis_id'])
        return Signal(row['name'], self._read_array(axis['array_id']),
                      self._read_array(row['values_id']), row['units'], axis['units'], axis['meaning'])

    def create_well(self, name, details=None):
        _required(name)
        identity = _id()
        with transaction(self._db):
            self._db.execute('INSERT INTO well VALUES(?,?,?)', (identity, name, _json(details or {})))
        return identity

    def find_wells(self, name=''):
        # Case-sensitive substring search also works for Cyrillic without SQLite NOCASE.
        return [dict(r) | {'details': json.loads(r['details'])} for r in self._db.execute(
            'SELECT * FROM well WHERE instr(name,?)>0 ORDER BY name,id', (name,))]

    def list_studies(self, well_id):
        return [dict(r) for r in self._db.execute('SELECT * FROM study WHERE well_id=? ORDER BY created_at,id', (well_id,))]

    def find_sources(self, source):
        """Report equal bytes and same names separately; never silently deduplicate."""
        return [dict(r) for r in self._db.execute(
            'SELECT id,study_id,name,sha256,sha256=? AS same_content,name=? AS same_name FROM source_file WHERE sha256=? OR name=?',
            (codec.digest(source.content), source.name, codec.digest(source.content), source.name))]

    def save_study(self, well_id, study):
        _required(study.loader_version)
        if study.equipment not in ('new', 'old') or not study.sources or not study.channels:
            raise ValueError('Incomplete measurement set')
        roles = [c.role for c in study.channels]
        if roles.count('noise') != 1 or not 1 <= roles.count('signal') <= (11 if study.equipment == 'new' else 15):
            raise ValueError('Expected one noise channel and supported signal count')
        identity = _id()
        # Manifest includes exact source bytes, order, axis/values and transformations.
        manifest = {'equipment': study.equipment, 'loader': study.loader_version,
                    'sources': [(s.name, s.format, codec.digest(s.content)) for s in study.sources],
                    'channels': [(c.name, c.role, c.source_ordinal, c.units, c.axis_units, c.axis_meaning,
                                  c.transformation, codec.digest(codec.encode(c.axis)[1]),
                                  codec.digest(codec.encode(c.values)[1])) for c in study.channels]}
        input_hash = codec.digest(_json(manifest).encode())
        with transaction(self._db):
            self._db.execute('INSERT INTO study VALUES(?,?,?,?,?,?,?,?)',
                (identity, well_id, study.observed_at, study.equipment, study.description, study.loader_version, input_hash, _now()))
            sources = []
            for ordinal, source in enumerate(study.sources):
                _required(source.name)
                if not source.content or len(source.content) > codec.MAX_BYTES:
                    raise ValueError('Empty or oversized original')
                sid = _id()
                self._db.execute('INSERT INTO source_file VALUES(?,?,?,?,?,?,?,?)',
                    (sid, identity, ordinal, source.name, source.format, len(source.content), codec.digest(source.content), source.content))
                sources.append(sid)
            for ordinal, channel in enumerate(study.channels):
                _required(channel.name)
                if not 0 <= channel.source_ordinal < len(sources):
                    raise ValueError('Unknown channel source')
                axis, values = self._signal(channel)
                self._db.execute('INSERT INTO channel VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (_id(), identity, sources[channel.source_ordinal], ordinal, channel.name, channel.role,
                     axis, values, channel.units, _json(channel.transformation)))
        return identity

    def get_channels(self, study_id):
        result = []
        for row in self._db.execute('SELECT * FROM channel WHERE study_id=? ORDER BY ordinal', (study_id,)):
            signal = self._read_signal(row)
            result.append({'id': row['id'], 'source_id': row['source_id'], 'ordinal': row['ordinal'],
                           'role': row['role'], 'transformation': json.loads(row['transformation']), 'signal': signal})
        return result

    def open_study(self, identity):
        row = self._row('study', identity)
        sources, ids = [], []
        for s in self._db.execute('SELECT * FROM source_file WHERE study_id=? ORDER BY ordinal', (identity,)):
            if len(s['payload']) != s['size'] or codec.digest(s['payload']) != s['sha256']:
                raise ValueError('Corrupt source')
            ids.append(s['id'])
            sources.append(Source(s['name'], s['format'], s['payload']))
        channels = [InputChannel(**vars(c['signal']), source_ordinal=ids.index(c['source_id']),
                                 role=c['role'], transformation=c['transformation']) for c in self.get_channels(identity)]
        return Study(row['equipment'], row['loader_version'], sources, channels, row['description'], row['observed_at'])

    def read_interval(self, channel_id, start, end):
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError('Invalid interval')
        s = self._read_signal(self._row('channel', channel_id))
        mask = (s.axis >= start) & (s.axis <= end)
        return Signal(s.name, s.axis[mask], s.values[mask], s.units, s.axis_units, s.axis_meaning)

    def save_run(self, study_id, run):
        _required(run.parameter_schema, run.algorithm_version, run.reference_hash, run.program_version)
        study = self._row('study', study_id)
        if not run.channel_ids or len(set(run.channel_ids)) != len(run.channel_ids):
            raise ValueError('Explicit unique run inputs required')
        if run.failure is None and not run.signals:
            raise ValueError('Completed run requires output')
        if run.failure is not None and (not run.failure or run.signals or run.statistics):
            raise ValueError('Failed run requires reason and no completed output')
        identity = _id()
        with transaction(self._db):
            self._db.execute('INSERT INTO processing_run VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (identity, study_id, study['input_hash'], _json(run.parameters), run.parameter_schema,
                 run.algorithm_version, run.reference_hash, run.program_version, _now(),
                 'failed' if run.failure is not None else 'complete', run.failure))
            for ordinal, channel in enumerate(run.channel_ids):
                self._db.execute('INSERT INTO run_input VALUES(?,?,?,?)', (identity, study_id, channel, ordinal))
            for signal in run.signals:
                _required(signal.name)
                axis, values = self._signal(signal)
                self._db.execute('INSERT INTO result_signal VALUES(?,?,?,?,?,?)',
                                 (_id(), identity, signal.name, axis, values, signal.units))
            for channel, metrics in run.statistics.items():
                if channel not in run.channel_ids:
                    raise ValueError('Statistics channel was not a run input')
                for name, (value, units) in metrics.items():
                    self._db.execute('INSERT INTO result_statistics VALUES(?,?,?,?,?,?)',
                                     (identity, study_id, channel, name, self._array(np.asarray(value)), units))
        return identity

    def list_runs(self, study_id):
        return [dict(r) for r in self._db.execute('SELECT * FROM processing_run WHERE study_id=? ORDER BY created_at,id', (study_id,))]

    def open_run(self, identity):
        row = self._row('processing_run', identity)
        row['parameters'] = json.loads(row['parameters'])
        row['channel_ids'] = [r[0] for r in self._db.execute('SELECT channel_id FROM run_input WHERE run_id=? ORDER BY ordinal', (identity,))]
        row['signals'] = [{'id': r['id'], 'signal': self._read_signal(r)} for r in self._db.execute('SELECT * FROM result_signal WHERE run_id=? ORDER BY name', (identity,))]
        row['statistics'] = {}
        for r in self._db.execute('SELECT * FROM result_statistics WHERE run_id=?', (identity,)):
            row['statistics'].setdefault(r['channel_id'], {})[r['name']] = (self._read_array(r['value_id']), r['units'])
        return row

    def save_interpretation(self, run_id, interpretation):
        _required(interpretation.class_system)
        if self._row('processing_run', run_id)['status'] != 'complete':
            raise ValueError('Failed run has no interpretation')
        if not interpretation.arrays or not interpretation.classes or len(set(interpretation.classes)) != len(interpretation.classes):
            raise ValueError('Explicit result arrays and class vocabulary required')
        if interpretation.base_id:
            base = self._row('interpretation', interpretation.base_id)
            if base['run_id'] != run_id or base['class_system'] != interpretation.class_system or json.loads(base['classes']) != interpretation.classes:
                raise ValueError('Incompatible base interpretation')
        elif interpretation.corrections:
            raise ValueError('Manual corrections require a base interpretation')
        signals = {r['id']: r for r in self._db.execute('SELECT * FROM result_signal WHERE run_id=?', (run_id,))}
        identity = _id()
        with transaction(self._db):
            self._db.execute('INSERT INTO interpretation VALUES(?,?,?,?,?,?,?)',
                (identity, run_id, interpretation.base_id, interpretation.class_system, _json(interpretation.classes),
                 _json(list(interpretation.arrays)), _now()))
            for name, array in interpretation.arrays.items():
                self._db.execute('INSERT INTO interpretation_array VALUES(?,?,?)', (identity, name, self._array(array)))
            for ordinal, correction in enumerate(interpretation.corrections):
                if correction.signal_id not in signals:
                    raise ValueError('Correction signal belongs to another run')
                _required(correction.operation)
                self._db.execute('INSERT INTO manual_correction VALUES(?,?,?,?,?,?,?)',
                    (identity, ordinal, correction.signal_id, correction.operation, _json(correction.parameters), correction.author, _now()))
            for ordinal, interval in enumerate(interpretation.intervals):
                if interval.signal_id not in signals or interval.label not in interpretation.classes or not all(map(math.isfinite, (interval.start, interval.end))):
                    raise ValueError('Invalid classification interval')
                self._db.execute('INSERT INTO quality_interval VALUES(?,?,?,?,?,?,?)',
                    (identity, ordinal, signals[interval.signal_id]['axis_id'], interval.start, interval.end, interval.label, interval.boundary))
        return identity

    def open_interpretation(self, identity):
        row = self._row('interpretation', identity)
        row['classes'] = json.loads(row['classes'])
        row['arrays'] = {r['name']: self._read_array(r['array_id']) for r in self._db.execute('SELECT * FROM interpretation_array WHERE interpretation_id=?', (identity,))}
        row['corrections'] = [dict(r) | {'parameters': json.loads(r['parameters'])} for r in self._db.execute('SELECT * FROM manual_correction WHERE interpretation_id=? ORDER BY ordinal', (identity,))]
        row['intervals'] = [dict(r) for r in self._db.execute('SELECT * FROM quality_interval WHERE interpretation_id=? ORDER BY ordinal', (identity,))]
        return row

    def list_interpretations(self, run_id):
        return [dict(r) for r in self._db.execute('SELECT * FROM interpretation WHERE run_id=? ORDER BY created_at,id', (run_id,))]

    def save_report(self, interpretation_id, content, *, details, template_version, template_hash, format='docx'):
        _required(template_version, template_hash, format)
        if not content or len(content) > codec.MAX_BYTES:
            raise ValueError('Empty or oversized report')
        identity = _id()
        with transaction(self._db):
            self._db.execute('INSERT INTO report VALUES(?,?,?,?,?,?,?,?,?)',
                (identity, interpretation_id, _json(details), template_version, template_hash, _now(), format, codec.digest(content), content))
        return identity

    def list_reports(self, interpretation_id):
        return [r[0] for r in self._db.execute('SELECT id FROM report WHERE interpretation_id=? ORDER BY created_at,id', (interpretation_id,))]

    def open_report(self, identity):
        row = self._row('report', identity)
        if codec.digest(row['payload']) != row['sha256']:
            raise ValueError('Corrupt report')
        row['details'] = json.loads(row['details'])
        return row
