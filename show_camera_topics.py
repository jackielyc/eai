"""EAI viewer entrypoint. Implementation: eai_app.application."""
if __name__ == '__main__':
    from eai_app.application.viewer import main
    raise SystemExit(main())
else:
    from eai_app.infrastructure.compat import alias_module
    alias_module(__name__, 'eai_app.application.viewer')
