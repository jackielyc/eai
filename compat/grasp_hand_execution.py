"""Compatibility entrypoint. Edit eai_app/domain/grasp/hand_execution.py instead."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    from eai_app.infrastructure.compat import run_entrypoint
    run_entrypoint('eai_app.domain.grasp.hand_execution')
else:
    from eai_app.infrastructure.compat import alias_module
    alias_module(__name__, 'eai_app.domain.grasp.hand_execution')
