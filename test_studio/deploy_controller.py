# -*- coding: utf-8 -*-
"""测试页 / 测试工作室 共用的推理部署控制器。

UI 只负责摆控件；启停、扫目录、健康检查、上次配置均在此，保证两端同步。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication


@dataclass
class DeployView:
    """一套部署相关控件（测试 Tab 或测试工作室页各有一份）。"""

    target_combo: Any
    root_combo: Any
    model_combo: Any
    refresh_btn: Any
    load_last_btn: Any
    start_btn: Any
    stop_btn: Any
    path_label: Any
    status_label: Any
    log_edit: Any
    show_message: Callable[[str], None]
    get_chat_panel: Callable[[], Any]
    focus_host: Any = None


class QwenDeployController(QObject):
    """本地/远程 Qwen 推理部署：多 View 共享同一组 launcher。"""

    status_message = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # 延迟导入，避免与 show_camera_topics 循环依赖
        import show_camera_topics as sct

        self._sct = sct
        self.local_launcher = sct.LocalQwenServiceLauncher(self)
        self.remote_launcher = sct.RemoteQwenServiceLauncher(self)
        self._list_bridge = sct.DeployModelListBridge(self)
        self._list_bridge.finished.connect(self._on_model_list_ready)
        self.local_launcher.status_message.connect(self._on_launcher_status)
        self.local_launcher.log_line.connect(self.append_log)
        self.local_launcher.running_changed.connect(self.update_all_ui)
        self.remote_launcher.status_message.connect(self._on_launcher_status)
        self.remote_launcher.log_line.connect(self.append_log)
        self.remote_launcher.running_changed.connect(self.update_all_ui)

        self._views: List[DeployView] = []
        self._active: Optional[DeployView] = None
        self._refreshing = False
        self._pending_model_path = ""
        self._pending_list_view: Optional[DeployView] = None
        self._switch_tries = 0
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self.update_all_ui)
        self._health_timer.start(4000)

    def register_view(self, view: DeployView) -> None:
        if view in self._views:
            return
        self._views.append(view)
        if self._active is None:
            self._active = view
        view.target_combo.currentIndexChanged.connect(
            lambda _i, v=view: self.on_target_changed(v)
        )
        view.root_combo.currentIndexChanged.connect(
            lambda _i, v=view: self.on_root_changed(v)
        )
        view.model_combo.currentIndexChanged.connect(
            lambda _i, v=view: self.on_model_changed(v)
        )
        view.refresh_btn.clicked.connect(lambda: self.refresh_model_list(view))
        view.load_last_btn.clicked.connect(lambda: self.load_last_config(view))
        view.start_btn.clicked.connect(lambda: self.start(view))
        view.stop_btn.clicked.connect(lambda: self.stop(view))
        self.refresh_scan_roots(view)
        self.refresh_model_list(view)
        self.update_ui(view)

    def append_log(self, line: str) -> None:
        for view in self._views:
            try:
                view.log_edit.append(line)
                bar = view.log_edit.verticalScrollBar()
                bar.setValue(bar.maximum())
            except Exception:
                pass

    def _on_launcher_status(self, msg: str) -> None:
        self.append_log(msg)
        self.status_message.emit(msg)
        for view in self._views:
            try:
                view.status_label.setText(f"服务: {msg}")
                view.show_message(msg)
            except Exception:
                pass

    def selected_target(self, view: DeployView) -> str:
        target = str(view.target_combo.currentData() or "local")
        if target == "local":
            return "local"
        if target.startswith("remote:") or target == "remote":
            return "remote"
        return "local"

    def selected_remote_host(self, view: DeployView) -> str:
        target = str(view.target_combo.currentData() or "")
        if target.startswith("remote:"):
            return target.split(":", 1)[1].strip() or self._sct.REMOTE_QWEN_SSH_HOST
        return self._sct.REMOTE_QWEN_SSH_HOST

    def selected_scan_root(self, view: DeployView) -> str:
        root = view.root_combo.currentData()
        if isinstance(root, str) and root.strip():
            return root.strip()
        return ""

    def selected_deploy_spec(self, view: DeployView) -> Dict[str, str]:
        data = view.model_combo.currentData()
        if isinstance(data, dict):
            path = str(data.get("path") or "").strip()
            if path:
                return {
                    "path": path,
                    "model_id": str(
                        data.get("model_id")
                        or os.path.basename(path.rstrip("/"))
                    ),
                    "label": str(
                        data.get("label")
                        or data.get("name")
                        or os.path.basename(path.rstrip("/"))
                    ),
                    "kind": str(data.get("kind") or ""),
                    "root": str(data.get("root") or ""),
                }
        return {}

    def selected_model_key(self, view: DeployView) -> str:
        spec = self.selected_deploy_spec(view)
        if spec.get("path"):
            return str(spec.get("model_id") or spec.get("path"))
        key = view.model_combo.currentData()
        if isinstance(key, str) and key:
            return key
        return self._sct.LOCAL_QWEN_DEPLOY_MODELS[0][0]

    def refresh_scan_roots(self, view: DeployView) -> None:
        sct = self._sct
        target = self.selected_target(view)
        prev = self.selected_scan_root(view)
        view.root_combo.blockSignals(True)
        view.root_combo.clear()
        if target == "local":
            roots = sct.local_deploy_scan_roots()
        else:
            roots = sct.remote_deploy_scan_roots(self.selected_remote_host(view))
        for path, label in roots:
            view.root_combo.addItem(label, path)
            idx = view.root_combo.count() - 1
            view.root_combo.setItemData(idx, path, Qt.ToolTipRole)
        if prev:
            idx = view.root_combo.findData(prev)
            if idx >= 0:
                view.root_combo.setCurrentIndex(idx)
        view.root_combo.blockSignals(False)

    def on_target_changed(self, view: DeployView) -> None:
        self._active = view
        if self.selected_target(view) == "remote":
            self.remote_launcher.set_host(self.selected_remote_host(view))
        self.refresh_scan_roots(view)
        self.refresh_model_list(view)
        self.update_all_ui()

    def on_root_changed(self, view: DeployView) -> None:
        self._active = view
        self.refresh_model_list(view)

    def on_model_changed(self, view: DeployView) -> None:
        self._active = view
        self.refresh_path_label(view)

    def refresh_model_list(self, view: DeployView) -> None:
        if self._refreshing:
            return
        root = self.selected_scan_root(view)
        if not root:
            return
        self._refreshing = True
        self._pending_list_view = view
        self._active = view
        fw = QApplication.focusWidget()
        if fw is view.model_combo or (
            fw is not None and view.model_combo.isAncestorOf(fw)
        ):
            host = view.focus_host
            if host is not None:
                host.setFocus(Qt.OtherFocusReason)
        view.refresh_btn.setEnabled(False)
        view.model_combo.setEnabled(False)
        self._sct._schedule_fcitx_restore(self._sct._IME_LAST_TEXT_WIDGET)
        target = self.selected_target(view)
        host_id = self.selected_remote_host(view)
        self.append_log(f"扫描可部署目录: {root} …")

        def _work() -> None:
            ok, models, err = self._sct.fetch_deploy_model_dirs(
                target=target, host_id=host_id, root=root
            )
            self._list_bridge.finished.emit(
                {
                    "ok": ok,
                    "models": models,
                    "message": err,
                    "root": root,
                    "target": target,
                }
            )

        threading.Thread(target=_work, daemon=True).start()

    def _on_model_list_ready(self, payload: object) -> None:
        sct = self._sct
        self._refreshing = False
        data = payload if isinstance(payload, dict) else {}
        ok = bool(data.get("ok"))
        models = data.get("models") if isinstance(data.get("models"), list) else []
        root = str(data.get("root") or "")
        view = self._pending_list_view or self._active
        if view is None:
            return
        prev_path = self._pending_model_path or str(
            self.selected_deploy_spec(view).get("path") or ""
        )
        self._pending_model_path = ""

        # 同步填充所有 view 的模型列表（保持两端一致）
        for v in self._views:
            self._fill_model_combo(v, models, root, prev_path)

        if ok:
            self.append_log(f"已加载 {len(models)} 个可部署目录（{root}）")
        else:
            msg = str(data.get("message") or "扫描失败")
            self.append_log(f"目录扫描失败: {msg}")
            view.show_message(f"目录扫描失败: {msg}")
        self.update_all_ui()

    def _fill_model_combo(
        self,
        view: DeployView,
        models: list,
        root: str,
        prev_path: str,
    ) -> None:
        sct = self._sct
        view.model_combo.blockSignals(True)
        view.model_combo.clear()
        for entry in models:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            if not path:
                continue
            name = str(entry.get("name") or os.path.basename(path.rstrip("/")))
            kind = str(entry.get("kind") or "")
            kind_label = sct.deploy_model_kind_label(kind)
            item_label = f"{name} ({kind_label})"
            spec = {
                "path": path,
                "name": name,
                "model_id": str(entry.get("model_id") or name),
                "label": str(entry.get("label") or name),
                "kind": kind,
                "root": str(entry.get("root") or root),
            }
            view.model_combo.addItem(item_label, spec)
            idx = view.model_combo.count() - 1
            tip = (
                f"{name}\nmodel_id={spec['model_id']}\n"
                f"路径: {path}\n类型: {kind_label}\n根目录: {spec['root']}"
            )
            view.model_combo.setItemData(idx, tip, Qt.ToolTipRole)
        if prev_path:
            for i in range(view.model_combo.count()):
                spec = view.model_combo.itemData(i)
                if isinstance(spec, dict) and spec.get("path") == prev_path:
                    view.model_combo.setCurrentIndex(i)
                    break
        elif view.model_combo.count() > 0:
            prefer = view.model_combo.findText("Qwen3.5-35B-A3B", Qt.MatchContains)
            view.model_combo.setCurrentIndex(prefer if prefer >= 0 else 0)
        view.model_combo.blockSignals(False)
        self.refresh_path_label(view)

    def refresh_path_label(self, view: DeployView) -> None:
        sct = self._sct
        spec = self.selected_deploy_spec(view)
        target = self.selected_target(view)
        if spec.get("path"):
            path = spec["path"]
            label = spec.get("label") or os.path.basename(path.rstrip("/"))
            mid = spec.get("model_id") or label
            kind = sct.deploy_model_kind_label(str(spec.get("kind") or ""))
            if target == "remote":
                host_id = self.selected_remote_host(view)
                prof = sct.remote_qwen_profile(host_id)
                py = sct.resolve_deploy_python_for_spec(
                    spec, target="remote", host_id=host_id
                )
                api = str(
                    prof.get("api_base") or sct.remote_qwen_api_base_for_host(host_id)
                )
                view.path_label.setText(f"{host_id}:{path}  ·  id={mid}")
                view.path_label.setStyleSheet(f"color: {sct.UI_TEXT_SECONDARY};")
                view.path_label.setToolTip(
                    f"远程部署 {label}（{kind}）\nSSH Host: {host_id}\n"
                    f"model_id={mid}\n路径: {path}\nPython: {py}\n"
                    f"工作目录: {prof.get('remote_work')}\n隧道 API: {api}"
                )
            else:
                view.path_label.setText(f"{path}  ·  id={mid}")
                view.path_label.setStyleSheet(f"color: {sct.UI_TEXT_SECONDARY};")
                view.path_label.setToolTip(
                    f"本地部署 {label}（{kind}）\nmodel_id={mid}\n路径: {path}\n"
                    f"API: {sct.LOCAL_QWEN_API_BASE_DEFAULT}"
                )
            return

        key = self.selected_model_key(view)
        label = sct.local_qwen_label_for_key(key)
        mid = sct.local_qwen_model_id_for_key(key)
        dirname = sct.qwen_deploy_path_spec_for_key(key)
        if target == "remote":
            host_id = self.selected_remote_host(view)
            prof = sct.remote_qwen_profile(host_id)
            remote_root = str(prof.get("model_root") or "")
            remote_path = sct.qwen_model_path_from_spec(dirname, root=remote_root)
            view.path_label.setText(f"{host_id}:{remote_path}  ·  id={mid}")
            view.path_label.setStyleSheet(f"color: {sct.UI_TEXT_SECONDARY};")
            return
        path = sct.local_qwen_model_dir_for_key(key)
        missing = sct.qwen_model_path_from_spec(dirname)
        if path:
            view.path_label.setText(f"{path}  ·  id={mid}")
            view.path_label.setStyleSheet(f"color: {sct.UI_TEXT_SECONDARY};")
        else:
            view.path_label.setText(f"未找到权重: {missing}")
            view.path_label.setStyleSheet(f"color: {sct.UI_ACCENT_ORANGE};")

    def save_deploy_config(self, view: DeployView) -> None:
        spec = self.selected_deploy_spec(view)
        try:
            self._sct.save_test_qwen_last_config(
                {
                    "target": str(view.target_combo.currentData() or "local"),
                    "scan_root": self.selected_scan_root(view),
                    "model_path": str(spec.get("path") or ""),
                    "model_id": str(spec.get("model_id") or ""),
                    "model_label": str(spec.get("label") or ""),
                    "saved_at": self._sct._utc_now_iso(),
                }
            )
        except Exception as exc:
            self.append_log(f"保存上次配置失败: {exc}")

    def load_last_config(self, view: DeployView) -> None:
        self._active = view
        cfg = self._sct.load_test_qwen_last_config()
        target = str(cfg.get("target") or "").strip()
        scan_root = str(cfg.get("scan_root") or "").strip()
        model_path = str(cfg.get("model_path") or "").strip()
        model_label = str(cfg.get("model_label") or model_path).strip()
        if not target and not scan_root and not model_path:
            self.append_log("无上次模型配置（需先成功点过一次「启动推理服务」）")
            view.show_message("无上次推理配置")
            return

        for v in self._views:
            v.target_combo.blockSignals(True)
            idx = v.target_combo.findData(target or "local")
            if idx >= 0:
                v.target_combo.setCurrentIndex(idx)
            v.target_combo.blockSignals(False)
            if self.selected_target(v) == "remote":
                self.remote_launcher.set_host(self.selected_remote_host(v))
            self.refresh_scan_roots(v)
            if scan_root:
                root_idx = v.root_combo.findData(scan_root)
                if root_idx >= 0:
                    v.root_combo.setCurrentIndex(root_idx)

        self._pending_model_path = model_path
        self.refresh_model_list(view)
        saved_at = str(cfg.get("saved_at") or "").strip()
        when = f" ({saved_at})" if saved_at else ""
        self.append_log(
            f"已加载上次配置{when}: {target or 'local'} / "
            f"{model_label or model_path or scan_root}"
        )
        view.show_message("已加载上次推理配置")

    def start(self, view: DeployView) -> None:
        sct = self._sct
        self._active = view
        spec = self.selected_deploy_spec(view)
        key = self.selected_model_key(view)
        target = self.selected_target(view)
        self._switch_tries = 0
        if not spec.get("path") and view.model_combo.count() == 0:
            self.append_log("请先点「刷新」加载可部署目录，并选择一个模型。")
            view.show_message("未选择模型")
            return
        self.save_deploy_config(view)
        if target == "remote":
            host_id = self.selected_remote_host(view)
            path = spec.get("path") or ""
            if (
                path
                and sct.LAKE_QWEN35_OUTPUT_ROOT in path
                and host_id != "psi_motus_2_for_liyichao"
            ):
                self.append_log("Lake 训练 output 目前仅在远程 psi_motus 可访问。")
                view.show_message("请切换部署位置为 psi_motus")
                return
            if path and not path.startswith("/share_data") and host_id == "tione-develop":
                self.append_log(
                    "提示: 所选目录若不在 tione-develop 本机/共享盘，部署可能失败。"
                )
            api = sct.remote_qwen_api_base_for_host(host_id)
            label = spec.get("label") or key
            self.append_log(f"远程部署: {host_id} / {label}，隧道 {api}")
            self.remote_launcher.start(
                model_key=key,
                host_id=host_id,
                model_dir=spec.get("path"),
                model_id=spec.get("model_id"),
                model_label=spec.get("label"),
            )
            self.update_all_ui()
            QTimer.singleShot(2500, lambda: self._try_switch_chat_remote(view))
            return

        if spec.get("path"):
            if key.endswith("-a3b") or "35b" in key.lower():
                self.append_log(
                    "提示: 本地单卡部署 35B 会走 CPU/磁盘 offload，很慢；"
                    "建议把部署位置改为远程 GPU 机。"
                )
            self.local_launcher.start(
                model_dir=spec.get("path"),
                model_id=spec.get("model_id"),
                model_label=spec.get("label"),
            )
        else:
            if key == "qwen3.5-35b-a3b":
                self.append_log(
                    "提示: 本地单卡部署 35B 会走 CPU/磁盘 offload，很慢；"
                    "建议把部署位置改为远程 GPU 机。"
                )
            self.local_launcher.start(model_key=key)
        self.update_all_ui()
        QTimer.singleShot(2000, lambda: self._try_switch_chat_local(view))

    def stop(self, view: DeployView) -> None:
        self._active = view
        if self.selected_target(view) == "remote":
            self.remote_launcher.stop(host_id=self.selected_remote_host(view))
        else:
            self.local_launcher.stop()
        self.update_all_ui()

    def _try_switch_chat_local(self, view: DeployView) -> None:
        sct = self._sct
        if self.selected_target(view) != "local":
            return
        api = self.local_launcher.api_base()
        info = sct.fetch_local_qwen_server_info(api)
        chat = view.get_chat_panel()
        if info and info.get("ok"):
            mid = str(info.get("model") or self.local_launcher.last_model_id())
            if chat is not None:
                chat.apply_local_qwen_service_preset(
                    api_base=api, model_id=mid, silent=False
                )
            self.update_all_ui()
            return
        if sct.check_local_qwen_hostctl_health():
            _ok, body = sct.local_qwen_hostctl_request(
                "/status", method="GET", timeout_s=2.0
            )
            if _ok and not body.get("process_running") and not body.get("service_healthy"):
                self.local_launcher._starting = False
                if not self.local_launcher._fail_notified:
                    self.local_launcher._fail_notified = True
                    err = str(body.get("last_error") or "推理进程已退出")
                    self.append_log(f"启动失败: {err}")
                self.update_all_ui()
                return
        self._switch_tries += 1
        if self._switch_tries < 180:
            QTimer.singleShot(2000, lambda: self._try_switch_chat_local(view))
        else:
            self.local_launcher._starting = False
            self.append_log(
                "启动超时：模型仍未就绪，请查看 eai/log/local_qwen_service.log"
            )
        self.update_all_ui()

    def _try_switch_chat_remote(self, view: DeployView) -> None:
        sct = self._sct
        if self.selected_target(view) != "remote":
            return
        api = self.remote_launcher.api_base()
        info = sct.fetch_local_qwen_server_info(api)
        chat = view.get_chat_panel()
        if info and info.get("ok"):
            self.remote_launcher._starting = False
            mid = str(info.get("model") or self.remote_launcher.last_model_id())
            if chat is not None:
                chat.apply_remote_qwen_service_preset(
                    api_base=api, model_id=mid, silent=False
                )
            self.update_all_ui()
            return
        if (
            not getattr(self.remote_launcher, "_starting", False)
            and getattr(self.remote_launcher, "_fail_notified", False)
        ):
            self.update_all_ui()
            return
        self._switch_tries += 1
        if self._switch_tries < 300:
            QTimer.singleShot(2000, lambda: self._try_switch_chat_remote(view))
        else:
            self.remote_launcher._starting = False
            host_id = self.selected_remote_host(view)
            prof = sct.remote_qwen_profile(host_id)
            log_hint = str(prof.get("remote_work") or "") + "/local_qwen_service.log"
            self.append_log(f"远程启动超时（{host_id}）：请查看远程日志 {log_hint}")
        self.update_all_ui()

    def update_all_ui(self, *_args) -> None:
        for view in self._views:
            self.update_ui(view)

    def update_ui(self, view: DeployView) -> None:
        if self.selected_target(view) == "remote":
            self._update_ui_remote(view)
        else:
            self._update_ui_local(view)
        self.refresh_path_label(view)

    def _update_ui_remote(self, view: DeployView) -> None:
        sct = self._sct
        host_id = self.selected_remote_host(view)
        self.remote_launcher.set_host(host_id)
        api = self.remote_launcher.api_base()
        info = sct.fetch_local_qwen_server_info(api)
        healthy = bool(info and info.get("ok"))
        was_starting = bool(getattr(self.remote_launcher, "_starting", False))
        starting = was_starting and not healthy
        chat = view.get_chat_panel()
        if healthy:
            self.remote_launcher._starting = False
            starting = False
            if was_starting and chat is not None:
                mid = str(
                    (info or {}).get("model") or self.remote_launcher.last_model_id()
                )
                chat.apply_remote_qwen_service_preset(
                    api_base=api, model_id=mid, silent=False
                )
                mode_s = sct.qwen_deploy_mode_suffix(info).lstrip(" ·")
                extra = f" {mode_s}" if mode_s else ""
                self.append_log(f"远程 Qwen 已就绪: {api} ({mid}){extra}")
        live_model = str((info or {}).get("model") or "")
        if healthy:
            suffix = f" · {live_model}" if live_model else ""
            suffix += sct.qwen_deploy_mode_suffix(info)
            view.status_label.setText(f"服务: 远程在线 {api}{suffix}")
            view.status_label.setStyleSheet(f"color: {sct.UI_ACCENT_GREEN};")
        elif starting:
            label = self.remote_launcher.last_model_label()
            view.status_label.setText(f"服务: 远程启动中（{label} @ {host_id}）…")
            view.status_label.setStyleSheet(f"color: {sct.UI_ACCENT_ORANGE};")
        else:
            view.status_label.setText(f"服务: 远程离线（{host_id}）")
            view.status_label.setStyleSheet(f"color: {sct.UI_TEXT_SECONDARY};")
        self._set_busy_state(view, healthy, starting)

    def _update_ui_local(self, view: DeployView) -> None:
        sct = self._sct
        api = self.local_launcher.api_base()
        info = sct.fetch_local_qwen_server_info(api)
        healthy = bool(info and info.get("ok"))
        hostctl_running = False
        hostctl_starting = False
        hostctl_error = ""
        if sct.check_local_qwen_hostctl_health():
            _ok, body = sct.local_qwen_hostctl_request(
                "/status", method="GET", timeout_s=2.0
            )
            if _ok:
                hostctl_running = bool(body.get("process_running"))
                hostctl_starting = bool(body.get("starting")) or (
                    hostctl_running and not healthy
                )
                hostctl_error = str(body.get("last_error") or "").strip()
        if healthy:
            self.local_launcher._starting = False
        elif getattr(self.local_launcher, "_starting", False):
            from PyQt5.QtCore import QProcess

            local_proc_alive = (
                self.local_launcher._process is not None
                and self.local_launcher._process.state() == QProcess.Running
            )
            if self.local_launcher._via_hostctl:
                if not hostctl_running and not healthy:
                    self.local_launcher._starting = False
                    self.local_launcher._via_hostctl = False
                    if not self.local_launcher._fail_notified:
                        self.local_launcher._fail_notified = True
                        err = hostctl_error or (
                            "推理进程已退出，请查看 log/local_qwen_service.log"
                        )
                        self.append_log(f"启动失败: {err}")
                        view.show_message("本地 Qwen 启动失败")
            elif not local_proc_alive:
                self.local_launcher._starting = False
        starting = (
            bool(getattr(self.local_launcher, "_starting", False)) and not healthy
        )
        from PyQt5.QtCore import QProcess

        if (
            self.local_launcher._process is not None
            and self.local_launcher._process.state() == QProcess.Running
            and not healthy
        ):
            starting = True
        if hostctl_starting:
            starting = True
            self.local_launcher._starting = True
        live_model = str((info or {}).get("model") or "")
        if healthy:
            suffix = f" · {live_model}" if live_model else ""
            suffix += sct.qwen_deploy_mode_suffix(info)
            view.status_label.setText(f"服务: 在线 {api}{suffix}")
            view.status_label.setStyleSheet(f"color: {sct.UI_ACCENT_GREEN};")
        elif starting:
            label = self.local_launcher.last_model_label()
            view.status_label.setText(f"服务: 启动中（加载 {label}）…")
            view.status_label.setStyleSheet(f"color: {sct.UI_ACCENT_ORANGE};")
        elif hostctl_error and not healthy:
            short = (
                hostctl_error
                if len(hostctl_error) <= 80
                else hostctl_error[:77] + "…"
            )
            view.status_label.setText(f"服务: 启动失败 · {short}")
            view.status_label.setStyleSheet(f"color: {sct.UI_ACCENT_RED};")
            view.status_label.setToolTip(hostctl_error)
        else:
            view.status_label.setText("服务: 离线")
            view.status_label.setStyleSheet(f"color: {sct.UI_TEXT_SECONDARY};")
        self._set_busy_state(view, healthy, starting)

    def _set_busy_state(self, view: DeployView, healthy: bool, starting: bool) -> None:
        view.start_btn.setEnabled(not healthy and not starting)
        view.stop_btn.setEnabled(healthy or starting)
        service_busy = healthy or starting
        scanning = bool(self._refreshing)
        view.model_combo.setEnabled(not service_busy and not scanning)
        view.target_combo.setEnabled(not service_busy)
        view.root_combo.setEnabled(not service_busy)
        view.refresh_btn.setEnabled(not service_busy and not scanning)
        view.load_last_btn.setEnabled(not service_busy and not scanning)
