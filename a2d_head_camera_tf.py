"""Compatibility entrypoint. Edit eai_app/adapters/ros/head_camera_tf.py instead."""
if __name__ == '__main__':
    from eai_app.infrastructure.compat import run_entrypoint
    run_entrypoint('eai_app.adapters.ros.head_camera_tf')
else:
    from eai_app.infrastructure.compat import alias_module
    alias_module(__name__, 'eai_app.adapters.ros.head_camera_tf')
