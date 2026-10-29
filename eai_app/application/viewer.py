"""Backward-compatible public API for the modular EAI viewer.

Implementation modules use qualified imports. This facade only serves existing
entrypoints, integrations and patches that previously imported show_camera_topics.
"""
import importlib
import json
from pathlib import Path
import sys
import types

_registry = json.loads(Path(__file__).with_name('viewer_registry.json').read_text())
_exports = _registry['exports']

# Preserve the original ordering of configuration/Qt registration side effects.
for _module_name in _registry['initialization_order']:
    importlib.import_module(_module_name)


class _ViewerAPI(types.ModuleType):
    def __getattr__(self, name):
        target = _exports.get(name)
        if target:
            return getattr(importlib.import_module(target), name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        target = _exports.get(name)
        if target:
            setattr(importlib.import_module(target), name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        target = _exports.get(name)
        if target:
            delattr(importlib.import_module(target), name)
        else:
            super().__delattr__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(_exports))


sys.modules[__name__].__class__ = _ViewerAPI
__all__ = [name for name, module in _exports.items()
           if not name.startswith('_') and hasattr(importlib.import_module(module), name)]
