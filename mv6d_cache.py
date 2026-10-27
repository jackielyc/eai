"""Atomic, non-pickle storage for visual 6D dataclasses and numpy arrays."""
import dataclasses
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def save_cache(path, payload):
    arrays = {}
    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('缓存不支持 object 数组')
            key = f'array_{len(arrays)}'
            arrays[key] = value
            return {'array': key}
        if dataclasses.is_dataclass(value):
            return {'dataclass': type(value).__name__, 'fields':
                    {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}}
        if isinstance(value, dict):
            return {'dict': {str(k): encode(v) for k, v in value.items()}}
        if isinstance(value, (list, tuple)):
            return {'tuple' if isinstance(value, tuple) else 'list': [encode(v) for v in value]}
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ValueError(f'不支持的缓存类型：{type(value).__name__}')
    metadata = json.dumps({'version': 1, 'payload': encode(payload)}, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, metadata=np.array(metadata), **arrays)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_cache(path, classes):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        if metadata.get('version') != 1:
            raise ValueError('不支持的6D缓存版本')
        def decode(value):
            if not isinstance(value, dict):
                return value
            if 'array' in value:
                return archive[value['array']].copy()
            if 'dataclass' in value:
                return classes[value['dataclass']](**{k: decode(v) for k, v in value['fields'].items()})
            if 'dict' in value:
                return {k: decode(v) for k, v in value['dict'].items()}
            if 'tuple' in value:
                return tuple(decode(v) for v in value['tuple'])
            if 'list' in value:
                return [decode(v) for v in value['list']]
            raise ValueError('损坏的6D缓存结构')
        return decode(metadata['payload'])
