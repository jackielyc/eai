"""EAI viewer entrypoint. Implementation: eai_app.application."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    from eai_app.application.viewer import main
    raise SystemExit(main())
else:
    from eai_app.infrastructure.compat import alias_module
    alias_module(__name__, 'eai_app.application.viewer')
