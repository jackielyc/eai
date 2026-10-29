"""ui.window.models chat — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class ModelsChatMixin:
    def _on_chat_attached_image_changed(self, path: str) -> None:
        if not path:
            self._set_test_scenario_selection("")
            return
        for name, p in self._test_image_paths.items():
            if _eai_platform_bootstrap.os.path.abspath(p) == _eai_platform_bootstrap.os.path.abspath(path):
                self._set_test_scenario_selection(name)
                return

    def _on_local_ai_deploy_clicked(self) -> None:
        if self._local_ai_launcher.is_sam3_running():
            # 已部署（本窗口进程 / 外部进程 / HTTP 健康）→ 停止，不重复启动
            self._local_ai_launcher.stop()
            self._update_local_ai_deploy_btn()
            self._refresh_sam3_status()
            return

        if _eai_perception_segmentation.resolve_sam3_run_script() is None and _eai_perception_segmentation.is_running_in_docker():
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "本地部署 AI",
                "当前 viewer 在 Docker 内运行，无法直接启动宿主机 SAM3。\n\n"
                "请在宿主机终端执行：\n"
                "  cd ~/workspace_liyichao/eai\n"
                "  bash scripts/launch/run_sam3.sh --host 0.0.0.0\n\n"
                "启动后本界面会自动通过 HTTP 连接 SAM3，并尝试启动 Ollama。",
            )

        self._local_ai_launcher.start_deploy()
        self._configure_sam3_for_local_http()
        self.chat_panel.apply_local_ollama_preset(silent=True)
        _eai_platform_bootstrap.QTimer.singleShot(2500, self._after_local_ai_deploy)

    def _deploy_status_color(self, level: str) -> str:
        return {
            "ok": _eai_configuration_policies.UI_ACCENT_GREEN,
            "warn": _eai_configuration_policies.UI_ACCENT_ORANGE,
            "err": _eai_configuration_policies.UI_ACCENT_RED,
            "muted": _eai_configuration_policies.UI_TEXT_MUTED,
        }.get(level, _eai_configuration_policies.UI_TEXT_MUTED)

