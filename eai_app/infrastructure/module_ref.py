"""Lazy, qualified references for the migrated UI's formerly circular bindings.

Only the legacy UI migration uses these references. No shared exec namespace is
used: each module owns its functions and state, and lookups resolve to that owner.
"""
from importlib import import_module


class ModuleRef:
    __slots__ = ('_name',)

    def __init__(self, name):
        object.__setattr__(self, '_name', name)

    def __getattr__(self, name):
        return getattr(import_module(self._name), name)

    def __setattr__(self, name, value):
        setattr(import_module(self._name), name, value)

    def __delattr__(self, name):
        delattr(import_module(self._name), name)


def module_ref(name):
    return ModuleRef(name)
