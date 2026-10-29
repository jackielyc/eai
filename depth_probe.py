"""Compatibility entrypoint. Edit eai_app/tools/depth_probe.py instead."""
if __name__ == '__main__':
    from eai_app.infrastructure.compat import run_entrypoint
    run_entrypoint('eai_app.tools.depth_probe')
else:
    from eai_app.infrastructure.compat import alias_module
    alias_module(__name__, 'eai_app.tools.depth_probe')
