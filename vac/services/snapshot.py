"""Explicit, versioned state codec: JSON structure and numeric NPY arrays; no pickle."""
from dataclasses import fields, is_dataclass
import math
import numpy as np
import pandas as pd
from ..domain import models
from ..domain.state import ApplicationState

CLASSES = {c.__name__: c for c in (ApplicationState, models.Channel, models.ProcessingParameters,
    models.ProcessingResult, models.MeasurementSet, models.QualityIntervals, models.SignalIdentity,
    models.ManualCorrection, models.ReportDetails, models.Interpretation)}


def encode(value):
    arrays = {}

    def pack(v):
        if isinstance(v, np.generic):
            v = v.item()
        if v is None or isinstance(v, (str, bool, int)):
            return v
        if isinstance(v, float):
            return {'kind': 'float', 'value': repr(v)} if not math.isfinite(v) else v
        if isinstance(v, np.ndarray):
            if v.dtype.kind == 'O':
                return {'kind': 'object', 'values': pack(v.tolist())}
            name = str(len(arrays))
            arrays[name] = v.copy()
            return {'kind': 'array', 'name': name}
        if isinstance(v, pd.DataFrame):
            return {'kind': 'frame', 'columns': pack(list(v.columns)), 'index': pack(list(v.index)),
                    'range': isinstance(v.index, pd.RangeIndex),
                    'data': [pack(v[c].to_numpy()) for c in v]}
        if is_dataclass(v) and type(v).__name__ in CLASSES:
            return {'kind': 'model', 'name': type(v).__name__,
                    'fields': {f.name: pack(getattr(v, f.name)) for f in fields(v)}}
        if isinstance(v, (list, tuple)):
            return {'kind': 'tuple' if isinstance(v, tuple) else 'list', 'items': [pack(i) for i in v]}
        if isinstance(v, dict):
            return {'kind': 'dict', 'items': [[pack(k), pack(i)] for k, i in v.items()]}
        raise ValueError('Unsupported snapshot value: ' + type(v).__name__)

    return pack(value), arrays


def decode(metadata, arrays):
    def unpack(v):
        if not isinstance(v, dict):
            return v
        kind = v['kind']
        if kind == 'float':
            return float(v['value'])
        if kind == 'array':
            return arrays[v['name']].copy()
        if kind == 'object':
            return np.array(unpack(v['values']), dtype=object)
        if kind in ('tuple', 'list'):
            items = [unpack(i) for i in v['items']]
            return tuple(items) if kind == 'tuple' else items
        if kind == 'dict':
            return {unpack(k): unpack(i) for k, i in v['items']}
        if kind == 'frame':
            index = unpack(v['index'])
            if v['range'] and index:
                step = index[1] - index[0] if len(index) > 1 else 1
                index = pd.RangeIndex(index[0], index[-1] + step, step)
            return pd.DataFrame({c: unpack(a) for c, a in zip(unpack(v['columns']), v['data'])}, index=index)
        if kind == 'model':
            cls = CLASSES[v['name']]
            values = {k: unpack(i) for k, i in v['fields'].items()}
            obj = cls(**{f.name: values[f.name] for f in fields(cls) if f.init})
            for f in fields(cls):
                if not f.init:
                    setattr(obj, f.name, values[f.name])
            return obj
        raise ValueError('Unknown snapshot kind')
    return unpack(metadata)
