"""Compatibility entrypoint. Edit eai_app/workers/spatial/lingbot_map.py instead."""
if __name__ == '__main__':
    from eai_app.infrastructure.compat import run_entrypoint
    run_entrypoint('eai_app.workers.spatial.lingbot_map')
else:
    from eai_app.infrastructure.compat import alias_module
    alias_module(__name__, 'eai_app.workers.spatial.lingbot_map')
