"""adapters.simulation.robodojo — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

from eai_app.adapters.simulation.robodojo_parts.actions import ActionsMixin
from eai_app.adapters.simulation.robodojo_parts.actions_start_new_process import ActionsStartNewProcessMixin
from eai_app.adapters.simulation.robodojo_parts.actions_restart_keepalive import ActionsRestartKeepaliveMixin

class RoboDojoEvalLauncher(ActionsMixin, ActionsStartNewProcessMixin, ActionsRestartKeepaliveMixin, _eai_platform_bootstrap.QObject):
    """启动/停止 RoboDojo 单任务评测（GUI: run_gui_eval.sh / headless: robodojo.sh eval）。

    默认 EVAL_NUM=keep：评测进程与 Isaac 常驻，直到显式点「停止评测」(stop())。
    EAI 界面退出 (shutdown) 不杀 Isaac：进程经 persistent_model_shell 脱离 UI。
    启动时优先接管已在运行的 eval_client / 持久化 session / Isaac Kit，避免重复拉起。
    """
    PERSIST_SLOT = "robodojo_eval"
    log_line = _eai_platform_bootstrap.pyqtSignal(str)
    status_message = _eai_platform_bootstrap.pyqtSignal(str)
    running_changed = _eai_platform_bootstrap.pyqtSignal(bool)
    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._pgid: _eai_platform_bootstrap.Optional[int] = None
        self._stopping = False
        self._use_gui = True
        self._keep_alive = True
        self._restart_pending = False
        self._last_start_kwargs: _eai_platform_bootstrap.Optional[dict] = None
        self._external_pids: _eai_platform_bootstrap.List[int] = []
        self._adopted = False
        # Isaac/eval 在 UI 退出后仍保留；仅「停止评测」会真正杀掉。
        self._keep_on_exit = True
        self._persist_pid_path = _eai_adapters_models_qwen_local.persistent_slot_paths(self.PERSIST_SLOT)[1]
        self._adopt_watch_timer = _eai_platform_bootstrap.QTimer(self)
        self._adopt_watch_timer.setInterval(2000)
        self._adopt_watch_timer.timeout.connect(self._on_adopt_watch_tick)
