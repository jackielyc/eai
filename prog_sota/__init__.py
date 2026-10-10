"""编程冲 SOTA：文本/图像归纳手臂动作，编程实现任务并对照业内榜单。"""

from prog_sota.catalog import BACKENDS, TASKS
from prog_sota.loop import LoopConfig, run_loop

__all__ = ["BACKENDS", "TASKS", "LoopConfig", "run_loop"]
