"""Pure-Python source consistency boundary shared by GUI and worker processes."""
import hashlib
from pathlib import Path
from eai_app.paths import PROJECT_ROOT

_check = None


def install_check(callback):
    global _check
    _check = callback


def assert_sources_current():
    if _check is not None:
        _check()


def source_revision():
    root = PROJECT_ROOT / 'eai_app'
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_revision(expected):
    if expected and source_revision() != expected:
        raise ValueError('执行代码在规划/启动期间发生变化，请重新加载代码并重新规划')
