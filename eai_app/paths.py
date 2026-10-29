"""Stable resource locations, independent of source file placement."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / 'config'
CACHE_ROOT = PROJECT_ROOT / '.cache'
TEST_DATA_ROOT = PROJECT_ROOT / 'tests' / 'data'


def source_path(legacy_filename):
    """Resolve an implementation for provenance hashes, not its compatibility stub."""
    from eai_app.infrastructure.compat import MODULE_ALIASES
    import importlib.util
    name = str(legacy_filename).removesuffix('.py')
    target = MODULE_ALIASES.get(name)
    if target:
        spec = importlib.util.find_spec(target)
        if spec and spec.origin:
            return Path(spec.origin)
    return PROJECT_ROOT / legacy_filename
