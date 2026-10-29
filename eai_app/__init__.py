"""EAI application: domain logic, adapters, application services and Qt views."""

# Existing integrations still import legacy module names. Keep those small
# forwarding modules out of the project root, sharing canonical module identity.
import sys as _sys
from pathlib import Path as _Path
_compat_path = str(_Path(__file__).resolve().parents[1] / 'compat')
if _compat_path not in _sys.path:
    _sys.path.append(_compat_path)
