"""PhysicalRSI 1.0：System 2 改写 System 1 harness，按实体评测选择并继承。"""

from physical_rsi.loop import LoopConfig, run_loop
from physical_rsi.tasks import TASKS

__all__ = ["LoopConfig", "TASKS", "run_loop"]
