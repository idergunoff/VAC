"""Lossless, bounded numeric NPY payloads; no Python object deserialization."""
import hashlib
import io
import math
import numpy as np

MAX_BYTES = 256 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(array):
    array = np.asarray(array)
    if array.dtype.kind not in 'biufc' or array.nbytes > MAX_BYTES:
        raise ValueError('Expected bounded numeric array')
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
    payload = stream.getvalue()
    if len(payload) > MAX_BYTES:
        raise ValueError('Array payload too large')
    return array, payload


def decode(payload, dtype, shape, checksum):
    if len(payload) > MAX_BYTES or digest(payload) != checksum:
        raise ValueError('Array size/checksum mismatch')
    stream = io.BytesIO(payload)
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError('Unsupported NPY version')
    actual_shape, _, actual_dtype = np.lib.format.read_array_header_1_0(stream)
    if (actual_dtype.kind not in 'biufc' or actual_dtype.str != dtype
            or list(actual_shape) != shape
            or math.prod(actual_shape) * actual_dtype.itemsize != len(payload) - stream.tell()):
        raise ValueError('Invalid array metadata')
    stream.seek(0)
    return np.load(stream, allow_pickle=False)
