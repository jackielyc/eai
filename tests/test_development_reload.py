import sys
import tempfile
import types
import unittest
from pathlib import Path
from eai_app.development.reloader import SourceReloader
from eai_app.development import source_guard


class ReloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root/'feature.py'
        self.original = ('from __future__ import annotations\n'
                         'events = []\n'
                         'class Feature:\n'
                         '    def value(self, n=2):\n'
                         '        events.append(n)\n'
                         '        return n + 1\n')
        self.path.write_text(self.original)
        self.module = types.ModuleType('feature')
        self.module.__file__ = str(self.path)
        exec(compile(self.original, str(self.path), 'exec'), vars(self.module))
        self.busy = ''
        self.reload = SourceReloader(self.root, lambda: self.busy, {'feature': self.module})

    def test_connected_callback_identity_and_runtime_state_survive(self):
        obj = self.module.Feature()
        callback = obj.value
        self.assertEqual(callback(), 3)
        self.path.write_text(self.original.replace('n + 1', 'n + 8'))
        self.assertEqual(self.reload.check()[0], 'applied')
        self.assertEqual(callback(), 10)
        self.assertIs(callback.__func__, self.module.Feature.value)
        self.assertEqual(self.module.events, [2, 2])

    def test_syntax_failure_is_atomic_across_files(self):
        other = self.root/'other.py'
        other.write_text('x = 1\n')
        self.reload.sources = self.reload.read_sources()
        self.path.write_text(self.original.replace('n + 1', 'n + 8'))
        other.write_text('def broken(:')
        self.assertEqual(self.reload.check()[0], 'error')
        self.assertEqual(self.module.Feature().value(), 3)

    def test_paused_or_busy_defers_until_idle(self):
        self.path.write_text(self.original.replace('n + 1', 'n + 8'))
        self.busy = 'paused task'
        self.assertEqual(self.reload.check()[0], 'deferred')
        self.assertEqual(self.module.Feature().value(), 3)
        self.busy = ''
        self.assertEqual(self.reload.check()[0], 'applied')

    def test_signature_constants_imports_and_new_methods_require_restart(self):
        for change in (self.original.replace('n=2', 'n=3'), self.original + '\nX = 3\n',
                       self.original + '\nimport os\n', self.original + '    def new(self): pass\n'):
            self.path.write_text(change)
            self.assertEqual(self.reload.check()[0], 'restart')
            self.assertEqual(self.module.Feature().value(), 3)

    def test_constructor_edit_is_not_silently_applied(self):
        original = 'class Feature:\n    def __init__(self): self.x = 1\n'
        self.path.write_text(original)
        exec(compile(original, str(self.path), 'exec'), vars(self.module))
        self.reload.sources = self.reload.read_sources()
        self.path.write_text(original.replace('= 1', '= 2'))
        self.assertEqual(self.reload.check()[0], 'restart')

    def test_super_closure_keeps_original_class(self):
        original = 'class Base:\n    def value(self): return 2\nclass Feature(Base):\n    def value(self): return super().value() + 1\n'
        self.path.write_text(original)
        exec(compile(original, str(self.path), 'exec'), vars(self.module))
        callback = self.module.Feature().value
        self.reload.sources = self.reload.read_sources()
        self.path.write_text(original.replace('+ 1', '+ 5'))
        self.assertEqual(self.reload.check()[0], 'applied')
        self.assertEqual(callback(), 7)

    def test_worker_revision_rejects_changed_code(self):
        with self.assertRaisesRegex(ValueError, '发生变化'):
            source_guard.verify_revision('stale')

    def test_execution_guard_propagates_pending_change(self):
        def pending(): raise ValueError('pending update')
        source_guard.install_check(pending)
        try:
            with self.assertRaisesRegex(ValueError, 'pending update'):
                source_guard.assert_sources_current()
        finally:
            source_guard.install_check(None)


if __name__ == '__main__': unittest.main()
