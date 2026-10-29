"""Thin legacy entrypoints; implementation modules never live in this layer."""
import importlib
import json
from pathlib import Path
import runpy
import sys
import types

MODULE_ALIASES = json.loads(Path(__file__).with_name('module_aliases.json').read_text())


def alias_module(name, target):
    module = importlib.import_module(target)
    sys.modules[name] = module
    return module


def run_entrypoint(target):
    runpy.run_module(target, run_name='__main__', alter_sys=True)


def forward_exports(module_name, exports):
    """Preserve legacy assignment/patch semantics without duplicating globals."""
    class ForwardingModule(types.ModuleType):
        def __getattr__(self, name):
            if name in exports:
                return getattr(importlib.import_module(exports[name]), name)
            raise AttributeError(name)

        def __setattr__(self, name, value):
            if name in exports:
                setattr(importlib.import_module(exports[name]), name, value)
            else:
                super().__setattr__(name, value)

        def __delattr__(self, name):
            if name in exports:
                delattr(importlib.import_module(exports[name]), name)
            else:
                super().__delattr__(name)

        def __dir__(self):
            return sorted(set(super().__dir__()) | set(exports))
    sys.modules[module_name].__class__ = ForwardingModule
