# 旧模块兼容入口

实现位于 `eai_app/`。这里仅保留旧模块名的转发入口，独立运行可用 `python compat/<名称>.py`。新代码请导入 `eai_app`。外部程序若直接导入旧模块名，请先 `import eai_app`，或把本目录加入 `PYTHONPATH`。
