"""Compatibility surface for task definitions and their Qt flow view."""
from eai_app.domain.tasks import definitions as _definitions
from eai_app.ui.grasp.flow import FlowStepLabel, TaskFlowWidget
from eai_app.infrastructure.compat import forward_exports as _forward_exports

_exports = {name: _definitions.__name__ for name in vars(_definitions) if not name.startswith('__')}
__all__ = [name for name in _exports if not name.startswith('_')] + ['FlowStepLabel', 'TaskFlowWidget']
_forward_exports(__name__, _exports)
