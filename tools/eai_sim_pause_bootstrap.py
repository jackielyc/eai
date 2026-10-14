"""Best-effort pause gate for extended / third-party sim entrypoints.

Imported early by ``run_external_bench`` so gym / gymnasium ``Env.step``
honors EAI ``sim_control.json`` (「暂停」按钮) when those stacks are used.
"""

from __future__ import annotations

import os
import sys


def _wait() -> None:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        eai = os.path.abspath(os.path.join(here, ".."))
        if eai not in sys.path:
            sys.path.insert(0, eai)
        from sim_gui_teleop import wait_while_paused

        wait_while_paused()
    except Exception:
        pass


def install() -> bool:
    """Patch gymnasium / gym Env.step once. Returns True if anything patched."""
    patched = False
    for mod_name in ("gymnasium", "gym"):
        try:
            mod = __import__(mod_name)
        except Exception:
            continue
        env_cls = getattr(mod, "Env", None)
        if env_cls is None or getattr(env_cls, "_eai_pause_patched", False):
            continue
        raw = env_cls.step

        def _step(self, action, _raw=raw):  # type: ignore[no-untyped-def]
            _wait()
            return _raw(self, action)

        env_cls.step = _step  # type: ignore[method-assign]
        env_cls._eai_pause_patched = True  # type: ignore[attr-defined]
        patched = True
    return patched


# Auto-install on import when launched under EAI.
if os.environ.get("EAI_SIM_PAUSE", "1").strip() not in ("0", "false", "False", "no"):
    try:
        install()
    except Exception:
        pass
