"""ui.window.simulation sim eval selected result json — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class SimulationSimEvalSelectedResultJsonMixin:
    def _sim_eval_selected_result_json(self) -> str:
        """Absolute path to selected RoboDojo `_result.json`, or empty."""
        tree = getattr(self, "sim_eval_tree", None)
        if tree is None:
            return ""
        item = tree.currentItem()
        if item is None:
            return ""
        path = str(item.data(0, _eai_platform_bootstrap.Qt.UserRole) or "")
        if not path:
            return ""
        result_json = _eai_platform_bootstrap.os.path.join(path, "_result.json")
        if _eai_platform_bootstrap.os.path.isfile(result_json):
            return result_json
        if _eai_platform_bootstrap.os.path.isfile(path) and path.endswith("_result.json"):
            return path
        return ""

    def _on_sim_leaderboard_merge_clicked(self) -> None:
        """Merge current (or manual) local eval result into the board for compare."""
        backend = self._sim_backend_id()
        result_json = self._sim_eval_selected_result_json()
        entry: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Any]] = None
        try:
            if result_json and backend in ("isaac", "mujoco"):
                # Prefer auto-fill from RoboDojo result; allow rename.
                draft = _eai_configuration_models.sim_eval_entry_from_result_json(
                    result_json, backend=backend
                )
                method, ok = _eai_platform_bootstrap.QInputDialog.getText(
                    self,
                    "合并到评测榜单",
                    "方法名（将与业内条目并排对比）:",
                    text=str(draft.get("method") or ""),
                )
                if not ok:
                    return
                method = (method or "").strip()
                if method:
                    draft["method"] = method
                entry = draft
            else:
                # Manual merge for any backend (and when no result selected).
                default_method = ""
                default_score = ""
                if result_json:
                    try:
                        draft = _eai_configuration_models.sim_eval_entry_from_result_json(
                            result_json, backend=backend
                        )
                        default_method = str(draft.get("method") or "")
                        default_score = str(draft.get("score") or "")
                        entry = draft
                    except Exception:
                        entry = None
                if entry is None:
                    method, ok = _eai_platform_bootstrap.QInputDialog.getText(
                        self,
                        "合并到评测榜单",
                        f"后端「{backend}」· 方法名:",
                        text=default_method,
                    )
                    if not ok or not (method or "").strip():
                        return
                    score, ok2 = _eai_platform_bootstrap.QInputDialog.getText(
                        self,
                        "合并到评测榜单",
                        "指标（如 48.0% 或 38.26 / 30.95%）:",
                        text=default_score,
                    )
                    if not ok2 or not (score or "").strip():
                        return
                    detail, ok3 = _eai_platform_bootstrap.QInputDialog.getText(
                        self,
                        "合并到评测榜单",
                        "说明（可选）:",
                        text="本地仿真评测",
                    )
                    if not ok3:
                        return
                    entry = _eai_configuration_models.sim_eval_entry_from_manual(
                        backend=backend,
                        method=method.strip(),
                        score=score.strip(),
                        detail=(detail or "").strip(),
                    )
                else:
                    method, ok = _eai_platform_bootstrap.QInputDialog.getText(
                        self,
                        "合并到评测榜单",
                        f"后端「{backend}」· 方法名:",
                        text=str(entry.get("method") or ""),
                    )
                    if not ok:
                        return
                    if (method or "").strip():
                        entry["method"] = method.strip()
            board = _eai_configuration_models.merge_sim_eval_local_entry(backend, entry)
            self._apply_sim_leaderboard_board(board)
            if hasattr(self, "status_bar") and self.status_bar is not None:
                self.status_bar.showMessage(
                    f"已合并本地评测到「{backend}」榜单 · {entry.get('method')} · {entry.get('score')}",
                    6000,
                )
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(self, "合并到榜单", f"合并失败: {exc}")

    def _on_sim_leaderboard_remove_local_clicked(self) -> None:
        """Remove the selected local row from the combined leaderboard."""
        table = getattr(self, "sim_leaderboard_table", None)
        if table is None:
            return
        row = table.currentRow()
        meta_list = getattr(self, "_sim_leaderboard_row_meta", None) or []
        if row < 0 or row >= len(meta_list):
            _eai_platform_bootstrap.QMessageBox.information(
                self, "移除本地", "请先在榜单中选中一行「本地」评测。"
            )
            return
        meta = meta_list[row]
        if str(meta.get("origin") or "") != "local" or not meta.get("local_id"):
            _eai_platform_bootstrap.QMessageBox.information(
                self, "移除本地", "当前选中的是业内条目，只能移除本地评测行。"
            )
            return
        backend = self._sim_backend_id()
        try:
            board = _eai_configuration_models.remove_sim_eval_local_entry(backend, str(meta["local_id"]))
            self._apply_sim_leaderboard_board(board)
            if hasattr(self, "status_bar") and self.status_bar is not None:
                self.status_bar.showMessage(
                    f"已移除本地行 · {meta.get('method') or meta.get('local_id')}",
                    4000,
                )
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(self, "移除本地", f"移除失败: {exc}")

    def _on_sim_extended_combo_changed(self, *_args) -> None:
        if (
            getattr(self, "sim_backend_extended_radio", None) is not None
            and self.sim_backend_extended_radio.isChecked()
        ):
            self._refresh_ext_bench_form()
            self._on_sim_backend_changed()

    def _on_sim_backend_changed(self, *_args) -> None:
        # radio.toggled(False) 是切换瞬间的中间态，等 True 再处理
        if len(_args) == 1 and _args[0] is False:
            return
        # radio 可能被程序 setChecked；保持顶部可拖动选择条与之同步
        if getattr(self, "sim_backend_tab_bar", None) is not None:
            self._sync_sim_backend_tab_from_radios()
        backend = self._sim_backend_id()
        extended = self._is_extended_sim_backend(backend)
        if hasattr(self, "sim_extended_combo"):
            self.sim_extended_combo.setEnabled(
                bool(
                    getattr(self, "sim_backend_extended_radio", None) is not None
                    and self.sim_backend_extended_radio.isChecked()
                )
            )
        if extended:
            self._refresh_ext_bench_form()
        idx = {
            "isaac": 0,
            "mujoco": 1,
            "molmospaces": 2,
            "psi_molmospaces": 2,
            "arena": 3,
            "libero": 4,
            "robotwin": 5,
        }.get(backend, 6 if extended else 0)
        if hasattr(self, "sim_backend_stack"):
            self.sim_backend_stack.setCurrentIndex(idx)
        if self._is_spaces_sim_backend(backend):
            self._sync_spaces_backend_defaults(backend)
        if hasattr(self, "_refresh_sim_leaderboard"):
            self._refresh_sim_leaderboard(backend)
        hints = {
            "isaac": (
                "Isaac / RoboDojo：启动评测后进程常驻，直到「停止评测」。"
                "图像预览：左腕 / 头部 / 右腕（共享目录 .npy → /camera/*）。"
            ),
            "mujoco": (
                "MuJoCo：本地 MJCF 交互 viewer（默认 VirtualGL→NVIDIA）。"
                "与 Isaac 评测互斥；不接 /camera/* 预览。"
            ),
            "molmospaces": (
                "MolmoSpaces：任务下拉框列出可评测 benchmark，启动时只跑当前选中的一项。"
                "与 Isaac 评测互斥；不接 /camera/* 预览。"
            ),
            "psi_molmospaces": (
                "Psi-MolmoSpaces（init_molmospaces）：任务列表=psi_molmo_spaces/*.json；"
                "「只开界面」与「启动」均按当前选中任务开场景；"
                "启动前可选视角：正面第三视角（默认，机器人+对象整体）/"
                "头部第一视角（手臂+对象）；Viewer=官方 MuJoCo GLFW；"
                "默认策略 WAM Psi-R1 Remote；下方「部署启动」拉起 :9317 服务并显示状态。"
                "与 Isaac 评测互斥；不接 /camera/* 预览。"
            ),
            "arena": (
                "IsaacLab-Arena：列出已注册 environment，启动时只跑当前选中的一项。"
                "与 Isaac / MuJoCo / MolmoSpaces 互斥；不接 /camera/* 预览。"
            ),
            "libero": (
                "LIBERO：对齐 RLinf openpi eval（libero_spatial/object/goal/10/90）。"
                "默认本地 ckpt 自动部署 Pi :8080；与其它仿真后端互斥。"
            ),
            "robotwin": (
                "RoboTwin：对齐 RLinf eval_embodiment（OpenPI + ALOHA / SAPIEN）。"
                "需 RLinf_support + assets；与其它仿真后端互斥。"
            ),
        }
        if extended:
            spec = _eai_configuration_models.get_sim_backend_spec(backend)
            hint = getattr(spec, "hint", "") if spec else ""
            if not hint:
                hint = (
                    f"扩展评测「{backend}」：通过 run_external_bench.py / "
                    "AllenAI VLA Harness Docker 启动；榜单可点刷新。"
                )
            hints[backend] = hint
        if hasattr(self, "sim_hint_label"):
            self.sim_hint_label.setText(hints.get(backend, ""))
        show_mj = backend in (
            "mujoco",
            "molmospaces",
            "psi_molmospaces",
            "arena",
            "libero",
            "robotwin",
        ) or extended
        if hasattr(self, "mujoco_run_row_widget"):
            self.mujoco_run_row_widget.setVisible(show_mj)
        if hasattr(self, "mujoco_log_edit"):
            self.mujoco_log_edit.setVisible(show_mj)
        bar = getattr(self, "mujoco_log_resize_bar", None)
        if bar is not None:
            bar.setVisible(show_mj)
        if hasattr(self, "_sim_log_pane"):
            self._sim_log_pane.setVisible(show_mj)
        if hasattr(self, "sim_rynnvalue_hud_panel"):
            self.sim_rynnvalue_hud_panel.setVisible(True)
            if hasattr(self, "_update_sim_rynnvalue_hud_controls"):
                self._update_sim_rynnvalue_hud_controls()
        if hasattr(self, "mujoco_start_btn"):
            start_labels = {
                "mujoco": "启动 MuJoCo",
                "molmospaces": "启动 MolmoSpaces",
                "psi_molmospaces": "启动 Psi-MolmoSpaces",
                "arena": "启动 Arena",
                "libero": "启动 LIBERO",
                "robotwin": "启动 RoboTwin",
            }
            if extended:
                spec = _eai_configuration_models.get_sim_backend_spec(backend)
                label = getattr(spec, "label", None) or backend
                self.mujoco_start_btn.setText(f"启动 {label}")
            else:
                self.mujoco_start_btn.setText(start_labels.get(backend, "启动"))
        if hasattr(self, "mujoco_open_model_btn"):
            self.mujoco_open_model_btn.setText(
                "打开模型目录" if backend == "mujoco" else "打开仓库目录"
            )
        if hasattr(self, "mujoco_status_label") and not (
            getattr(self, "_mujoco_launcher", None)
            and self._mujoco_launcher.is_running()
        ):
            if backend == "mujoco":
                self.mujoco_status_label.setText("空闲 · MuJoCo")
            elif backend == "molmospaces":
                self.mujoco_status_label.setText("空闲 · MolmoSpaces")
            elif backend == "psi_molmospaces":
                self.mujoco_status_label.setText("空闲 · Psi-MolmoSpaces")
            elif backend == "arena":
                self.mujoco_status_label.setText("空闲 · IsaacLab-Arena")
            elif backend == "libero":
                self.mujoco_status_label.setText("空闲 · LIBERO")
            elif backend == "robotwin":
                self.mujoco_status_label.setText("空闲 · RoboTwin")
            elif extended:
                spec = _eai_configuration_models.get_sim_backend_spec(backend)
                label = getattr(spec, "label", None) or backend
                self.mujoco_status_label.setText(f"空闲 · {label}")
        # 所有后端共用共享 .npy 预览；切后端时不关掉轮询（子进程写出 cam_* 即可刷新）。
        if hasattr(self, "sim_frame_status_label") and backend != "isaac":
            self.sim_frame_status_label.setText(
                "共享帧预览：直读 ISAAC_CAM_BRIDGE_DIR 下 cam_*.npy"
                "（LIBERO/其它后端写出后自动刷新 /camera/* 与多视角 6D）"
            )
        elif hasattr(self, "_refresh_sim_frame_status"):
            try:
                self._refresh_sim_frame_status()
            except Exception:
                pass
        self._update_mujoco_ui()
        self._update_sim_backend_mutex_ui()

    def _on_mujoco_backend_changed(self, *_args) -> None:
        # Backward-compatible alias: top-level sim backend radios drive the stack.
        self._on_sim_backend_changed()

    def _on_mujoco_unified_start_clicked(self) -> None:
        self._set_sim_paused(False, log=False)
        backend = self._sim_backend_id()
        if self._is_spaces_sim_backend(backend):
            self._on_molmospaces_run_clicked()
        elif backend == "arena":
            self._on_arena_run_clicked()
        elif backend == "libero":
            self._on_libero_run_clicked()
        elif backend == "robotwin":
            self._on_robotwin_run_clicked()
        elif self._is_extended_sim_backend(backend):
            self._on_ext_bench_run_clicked(viewer_only=False)
        else:
            self._on_mujoco_start_clicked()

    def _on_sim_viewer_only_clicked(self) -> None:
        """Start sim GUI only — no policy / model inference for any backend."""
        self._set_sim_paused(False, log=False)
        backend = self._sim_backend_id()
        if backend == "isaac":
            self._start_sim_eval(viewer_only=True)
        elif backend == "mujoco":
            self._on_mujoco_start_clicked(viewer_only=True)
        elif self._is_spaces_sim_backend(backend):
            self._on_molmospaces_run_clicked(viewer_only=True)
        elif backend == "arena":
            self._on_arena_run_clicked(viewer_only=True)
        elif backend == "libero":
            self._on_libero_run_clicked(viewer_only=True)
        elif backend == "robotwin":
            self._on_robotwin_viewer_only_clicked()
        elif self._is_extended_sim_backend(backend):
            self._on_ext_bench_run_clicked(viewer_only=True)
        else:
            self._on_mujoco_status(f"未知后端: {backend}")

    def _on_mujoco_open_selected_dir(self) -> None:
        backend = self._sim_backend_id()
        if self._is_spaces_sim_backend(backend):
            root = self._resolve_molmospaces_root()
        elif backend == "arena":
            root = self._resolve_arena_root()
        elif backend == "libero":
            root = _eai_adapters_models_discovery.resolve_rlinf_root(
                self.libero_rlinf_edit.text() if hasattr(self, "libero_rlinf_edit") else ""
            )
        elif backend == "robotwin":
            root = str(
                _eai_configuration_models.resolve_robotwin_root(
                    self.robotwin_root_edit.text()
                    if hasattr(self, "robotwin_root_edit")
                    else ""
                )
            )
        elif self._is_extended_sim_backend(backend):
            root = (
                self.ext_bench_root_edit.text().strip()
                if hasattr(self, "ext_bench_root_edit")
                else ""
            )
            root = _eai_configuration_models.resolve_external_bench_root(backend, root)
        else:
            self._on_mujoco_open_model_dir()
            return
        if _eai_platform_bootstrap.os.path.isdir(root):
            _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(root))
            self._append_mujoco_log(f"已打开目录: {root}")
        else:
            self._on_mujoco_status(f"目录不存在: {root}")

    def _on_ext_bench_root_browse(self) -> None:
        cur = (
            self.ext_bench_root_edit.text().strip()
            if hasattr(self, "ext_bench_root_edit")
            else ""
        ) or _eai_platform_bootstrap.os.path.expanduser("~")
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择扩展评测仓库根目录", cur)
        if selected and hasattr(self, "ext_bench_root_edit"):
            self.ext_bench_root_edit.setText(selected)

    def _on_ext_bench_probe_clicked(self) -> None:
        backend = self._sim_backend_id()
        if not self._is_extended_sim_backend(backend):
            self._on_mujoco_status("请先切换到「扩展评测」")
            return
        root = self.ext_bench_root_edit.text().strip() if hasattr(self, "ext_bench_root_edit") else ""
        py = self.ext_bench_python_edit.text().strip() if hasattr(self, "ext_bench_python_edit") else ""
        ok, msg = _eai_configuration_models.probe_external_bench_deps(backend, python_bin=py, root=root)
        line = ("OK · " if ok else "NEED · ") + msg
        self._append_mujoco_log(f"[{backend}] {line}")
        if hasattr(self, "ext_bench_status_label"):
            self.ext_bench_status_label.setText(line)
        self._on_mujoco_status(line)

    def _on_ext_bench_hint_clicked(self) -> None:
        backend = self._sim_backend_id()
        if not self._is_extended_sim_backend(backend):
            self._on_mujoco_status("请先切换到「扩展评测」")
            return
        py = self.ext_bench_python_edit.text().strip() if hasattr(self, "ext_bench_python_edit") else ""
        text = _eai_configuration_models.external_bench_install_hint(backend, py)
        self._append_mujoco_log("--- 安装指引 ---")
        for ln in (text or "").splitlines() or ["(empty)"]:
            self._append_mujoco_log(ln)
        self._on_mujoco_status(f"{backend} 安装指引已写入日志")

    def _on_ext_bench_docs_clicked(self) -> None:
        backend = self._sim_backend_id()
        spec = _eai_configuration_models.get_sim_backend_spec(backend)
        url = ""
        if spec is not None:
            url = getattr(spec, "source_url", "") or ""
        if not url:
            board = None
            try:
                board = _eai_configuration_models.load_sim_eval_board(backend)
            except Exception:
                board = None
            if isinstance(board, dict):
                url = str(board.get("source_url") or "")
        if url:
            _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl(url))
        else:
            self._on_mujoco_status(f"无文档链接: {backend}")

    def _on_ext_bench_run_clicked(self, *, viewer_only: bool = False) -> None:
        """Launch tools/run_external_bench.py for the selected extended suite."""
        backend = self._sim_backend_id()
        if not self._is_extended_sim_backend(backend):
            self._on_mujoco_status("请先将仿真后端切换为「扩展评测」")
            return
        if getattr(self, "_mujoco_launcher", None) and self._mujoco_launcher.is_running():
            self._on_mujoco_status("已有仿真进程在运行，请先停止")
            return
        if getattr(self, "_sim_eval_launcher", None) and self._sim_eval_launcher.is_running():
            self._on_mujoco_status("Isaac 评测仍在运行，请先停止")
            return

        root = self.ext_bench_root_edit.text().strip() if hasattr(self, "ext_bench_root_edit") else ""
        py = self.ext_bench_python_edit.text().strip() if hasattr(self, "ext_bench_python_edit") else ""
        mode = "viewer" if viewer_only else str(
            self.ext_bench_mode_combo.currentData()
            if hasattr(self, "ext_bench_mode_combo")
            else "eval"
        )
        if viewer_only:
            mode = "viewer"
        task = (
            self.ext_bench_task_edit.text().strip()
            if hasattr(self, "ext_bench_task_edit")
            else ""
        )
        episodes = (
            int(self.ext_bench_episodes_spin.value())
            if hasattr(self, "ext_bench_episodes_spin")
            else 1
        )
        viewer = bool(
            self.ext_bench_viewer_check.isChecked()
            if hasattr(self, "ext_bench_viewer_check")
            else True
        )
        harness = mode == "harness"
        policy = ""
        if hasattr(self, "ext_bench_policy_combo"):
            policy = str(self.ext_bench_policy_combo.currentData() or "")
        if (not viewer_only) and _eai_configuration_models.is_lingbot_va_policy(policy) and mode in (
            "eval",
            "probe",
            "hint",
        ):
            if mode == "hint":
                self._append_mujoco_log(_eai_configuration_models.lingbot_va_install_hint(py))
                self._on_mujoco_status("已打印 LingBot-VA 安装指引")
                return
            if mode == "probe":
                ok_va, va_msg = _eai_configuration_models.probe_lingbot_va()
                self._append_mujoco_log(f"[{backend}/LingBot-VA] {va_msg}")
                self._on_mujoco_status(
                    "LingBot-VA 就绪" if ok_va else "LingBot-VA 未就绪"
                )
                return
            self._append_mujoco_log(f"--- 扩展评测 {backend} · LingBot-VA ---")
            self._launch_lingbot_va_backend(
                backend,
                task=task,
                task_suite=task,
                label=f"ext-va:{backend}",
            )
            if hasattr(self, "ext_bench_status_label"):
                self.ext_bench_status_label.setText(
                    f"运行中 · {backend} · LingBot-VA"
                )
            return
        if (not viewer_only) and _eai_configuration_models.is_vpp2_policy(policy) and mode in (
            "eval",
            "probe",
            "hint",
        ):
            if mode == "hint":
                self._append_mujoco_log(_eai_configuration_models.vpp2_install_hint(py))
                self._on_mujoco_status("已打印 VPP2 安装指引")
                return
            if mode == "probe":
                ok_v, v_msg = _eai_configuration_models.probe_vpp2()
                self._append_mujoco_log(f"[{backend}/VPP2] {v_msg}")
                self._on_mujoco_status("VPP2 就绪" if ok_v else "VPP2 未就绪")
                return
            self._append_mujoco_log(f"--- 扩展评测 {backend} · VPP2 ---")
            self._launch_vpp2_backend(
                backend,
                task=task,
                task_suite=task,
                label=f"ext-vpp2:{backend}",
            )
            if hasattr(self, "ext_bench_status_label"):
                self.ext_bench_status_label.setText(
                    f"运行中 · {backend} · VPP2"
                )
            return
        try:
            plan = _eai_configuration_models.build_external_bench_argv(
                backend,
                root=root,
                python_bin=py,
                mode="harness" if harness else mode,
                task=task,
                episodes=episodes,
                viewer=viewer or viewer_only,
                harness=harness,
                policy=policy,
            )
        except Exception as exc:
            self._on_mujoco_status(f"构建启动命令失败: {exc}")
            self._append_mujoco_log(str(exc))
            return

        argv = list(plan.get("argv") or [])
        cwd = str(plan.get("cwd") or _eai_configuration_models.EAI_DIR)
        if not _eai_platform_bootstrap.os.path.isdir(cwd):
            cwd = _eai_configuration_models.EAI_DIR
        child_env = dict(plan.get("env") or {})
        try:
            child_env.update(self._rynnvalue_live_hud_child_env(backend))
        except Exception:
            pass
        note = str(plan.get("note") or "")
        self._append_mujoco_log(f"--- 扩展评测 {backend} · {mode} ---")
        if note:
            self._append_mujoco_log(note)
        self._append_mujoco_log("cmd: " + " ".join(argv))
        if hasattr(self, "ext_bench_status_label"):
            self.ext_bench_status_label.setText(f"运行中 · {backend} · {mode}")
        self._on_mujoco_status(f"启动扩展评测 · {backend}")
        self._mujoco_launcher.start_cwd_command(
            cwd=cwd,
            argv=argv,
            label=f"ext:{backend}",
            env_extra=child_env,
        )
        self._update_mujoco_ui()

    def _mujoco_model_label(self, path: str) -> str:
        """Short combo label; keep absolute path in userData."""
        abs_path = _eai_platform_bootstrap.os.path.abspath(path) if _eai_platform_bootstrap.os.path.isabs(path) else path
        urdf_model = _eai_platform_bootstrap.os.path.abspath(_eai_configuration_models.PSI_R1_MODEL_XML)
        if abs_path == urdf_model:
            return "psi_r1_ruiyan/urdf/model.xml  (psi_r1)"
        if _eai_platform_bootstrap.os.path.isabs(path) and path.startswith(_eai_configuration_models.EAI_DIR + _eai_platform_bootstrap.os.sep):
            return "eai/" + _eai_platform_bootstrap.os.path.relpath(path, _eai_configuration_models.EAI_DIR).replace(_eai_platform_bootstrap.os.sep, "/")
        return path

    def _on_mujoco_root_browse(self) -> None:
        cur = self.mujoco_root_edit.text().strip() or _eai_configuration_tabs.MUJOCO_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 MuJoCo 仓库", cur)
        if selected:
            self.mujoco_root_edit.setText(selected)
            self.mujoco_python_edit.setText(_eai_adapters_models_discovery.resolve_mujoco_python(selected))
            self._refresh_mujoco_models()

    def _on_mujoco_mjcf_browse(self) -> None:
        assets_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "assets")
        root = _eai_adapters_models_discovery.resolve_mujoco_root(self.mujoco_root_edit.text())
        if _eai_platform_bootstrap.os.path.isdir(assets_dir):
            start = assets_dir
        else:
            start = (
                _eai_platform_bootstrap.os.path.join(root, "model")
                if _eai_platform_bootstrap.os.path.isdir(_eai_platform_bootstrap.os.path.join(root, "model"))
                else root
            )
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 MJCF / XML",
            start,
            "MuJoCo MJCF (*.xml);;All (*)",
        )
        if selected:
            label = self._mujoco_model_label(selected)
            idx = self.mujoco_model_combo.findData(_eai_platform_bootstrap.os.path.abspath(selected))
            if idx < 0:
                self.mujoco_model_combo.addItem(label, _eai_platform_bootstrap.os.path.abspath(selected))
                idx = self.mujoco_model_combo.findData(_eai_platform_bootstrap.os.path.abspath(selected))
            if idx >= 0:
                self.mujoco_model_combo.setCurrentIndex(idx)
            else:
                self.mujoco_model_combo.setEditText(selected)

    def _on_mujoco_open_model_dir(self) -> None:
        root = _eai_adapters_models_discovery.resolve_mujoco_root(self.mujoco_root_edit.text())
        model_dir = _eai_platform_bootstrap.os.path.join(root, "model")
        path = model_dir if _eai_platform_bootstrap.os.path.isdir(model_dir) else root
        if _eai_platform_bootstrap.os.path.isdir(path):
            _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(path))

    def _sim_rynnvalue_hud_supported(self) -> bool:
        """All 仿真评测 backends can feed Live HUD (per-backend adapters)."""
        return self._sim_backend_id() in (
            "isaac",
            "mujoco",
            "molmospaces",
            "psi_molmospaces",
            "arena",
            "libero",
            "robotwin",
        )

    def _sim_rynnvalue_server_url(self) -> str:
        if hasattr(self, "sim_rynnvalue_url_edit"):
            return (
                self.sim_rynnvalue_url_edit.text().strip() or "http://127.0.0.1:8001"
            )
        return "http://127.0.0.1:8001"

    def _sim_rynnvalue_hud_enabled(self) -> bool:
        return bool(
            hasattr(self, "sim_rynnvalue_hud_check")
            and self.sim_rynnvalue_hud_check.isChecked()
        )

    def _sim_rynnvalue_hud_settings(self) -> dict:
        """Shared Live HUD knobs for any sim backend launcher."""
        return {
            "enabled": self._sim_rynnvalue_hud_enabled(),
            "server_url": self._sim_rynnvalue_server_url(),
            "refresh_sec": float(
                self.sim_rynnvalue_refresh_spin.value()
                if hasattr(self, "sim_rynnvalue_refresh_spin")
                else 1.0
            ),
            "num_frames": int(
                self.sim_rynnvalue_frames_spin.value()
                if hasattr(self, "sim_rynnvalue_frames_spin")
                else 8
            ),
            # Separate OpenCV Live HUD window (controls live in 工作区).
            "show_window": bool((_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip()),
        }

    def _rynnvalue_live_hud_child_env(self, backend: str) -> _eai_platform_bootstrap.Dict[str, str]:
        """Env vars injected into sim child processes for Live HUD."""
        settings = self._sim_rynnvalue_hud_settings()
        if not settings.get("enabled"):
            return {}
        try:
            from rynnvalue_hud_attach import child_env_from_settings
        except Exception:
            # show_camera_topics may not have tools/ on sys.path yet
            tools = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "tools")
            if tools not in _eai_platform_bootstrap.sys.path:
                _eai_platform_bootstrap.sys.path.insert(0, tools)
            from rynnvalue_hud_attach import child_env_from_settings
        status = _eai_platform_bootstrap.os.path.join(
            _eai_configuration_models.EAI_DIR, ".cache", "rynnvalue_live", f"{backend}_rynnvalue_live.json"
        )
        try:
            _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(status), exist_ok=True)
        except Exception:
            pass
        return child_env_from_settings(
            settings, backend=backend, status_path=status
        )

    def _on_isaac_rynnvalue_qt_hud_closed(self) -> None:
        self._append_sim_log("[rynnvalue] Live HUD 窗口已关闭；停止打分")
        self._stop_isaac_rynnvalue_npy_hud()

    def _rynnvalue_workspace_overlay_candidates(self) -> _eai_platform_bootstrap.List[str]:
        """Paths that may hold the latest LIBERO / Live HUD preview JPEG."""
        paths: _eai_platform_bootstrap.List[str] = []
        seen: set[str] = set()

        def _add(path: str) -> None:
            p = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path or ""))
            if not p or p in seen:
                return
            seen.add(p)
            paths.append(p)

        # Prefer Live HUD overlay when present; always include raw LIBERO preview.
        _add(_eai_configuration_policies.LIBERO_PREVIEW_JPG)
        cache = _eai_configuration_policies.RYNNVALUE_LIVE_CACHE_DIR
        try:
            if _eai_platform_bootstrap.os.path.isdir(cache):
                for name in sorted(_eai_platform_bootstrap.os.listdir(cache)):
                    if name.endswith("_overlay.jpg") or name == "libero_preview.jpg":
                        _add(_eai_platform_bootstrap.os.path.join(cache, name))
                for name in sorted(_eai_platform_bootstrap.os.listdir(cache)):
                    if not name.endswith(".json"):
                        continue
                    jp = _eai_platform_bootstrap.os.path.join(cache, name)
                    try:
                        with open(jp, "r", encoding="utf-8") as fh:
                            payload = _eai_platform_bootstrap.json.load(fh)
                        op = str((payload or {}).get("overlay_path") or "").strip()
                        if op:
                            _add(op)
                    except Exception:
                        pass
        except OSError:
            pass
        return paths

    def _rynnvalue_reward_runtime(self) -> _eai_platform_bootstrap.Tuple[bool, bool, bool]:
        """Return (launcher_running, port_listening, health_ok)."""
        launcher_running = bool(
            getattr(self, "_rynnvalue_launcher", None)
            and self._rynnvalue_launcher.is_running()
        )
        listening = False
        try:
            _host, port = _eai_adapters_models_discovery.parse_rynnvalue_server_url(self._sim_rynnvalue_server_url())
            listening = bool(_eai_configuration_models.port_listening(_host, port))
        except Exception:
            listening = False
        healthy = bool(getattr(self, "_sim_rynnvalue_last_ok", False))
        return launcher_running, listening, healthy

    def _rynnvalue_deploy_locked(self) -> bool:
        """True while deploying or the reward model/server is already up."""
        launcher_running, listening, healthy = self._rynnvalue_reward_runtime()
        return bool(launcher_running or listening or healthy)

    def _on_sim_rynnvalue_hud_toggled(self, checked: bool) -> None:
        self._update_sim_rynnvalue_hud_controls()
        if checked:
            self._expand_workspace_panel()
            self._set_rynnvalue_workspace_hud_active(True)
            self._append_mujoco_log(
                f"[rynnvalue] Live HUD 已勾选，探测 {self._sim_rynnvalue_server_url()}"
            )
            ok, msg = self._probe_and_show_rynnvalue_reward_status(log=True)
            if not ok:
                self._on_mujoco_status(f"RynnValue 未就绪: {msg}")
            else:
                self._on_mujoco_status(msg)
        else:
            self._set_rynnvalue_workspace_hud_active(False)
            self._probe_and_show_rynnvalue_reward_status(log=False)
            self._stop_isaac_rynnvalue_npy_hud()

    def _on_sim_rynnvalue_url_changed(self) -> None:
        if self._sim_rynnvalue_hud_enabled():
            self._probe_and_show_rynnvalue_reward_status(log=False)

    def _on_sim_rynnvalue_status_tick(self) -> None:
        self._probe_and_show_rynnvalue_reward_status(log=False)

    def _on_sim_rynnvalue_probe_clicked(self) -> None:
        ok, msg = self._probe_and_show_rynnvalue_reward_status(log=True)
        self._on_mujoco_status(msg if ok else f"RynnValue 未就绪: {msg}")

    def _on_rynnvalue_running_for_libero_hud(self, *_args) -> None:
        if self._sim_rynnvalue_hud_enabled():
            self._probe_and_show_rynnvalue_reward_status(log=False)

    def _on_sim_rynnvalue_deploy_clicked(self) -> None:
        if self._rynnvalue_deploy_locked():
            self._probe_and_show_rynnvalue_reward_status(log=False)
            if getattr(self, "_sim_rynnvalue_last_ok", False):
                tip = "RynnValue 模型已在运行"
            else:
                tip = "RynnValue 正在部署，请等待"
            self._on_mujoco_status(tip)
            self._set_sim_rynnvalue_status(f"◐ {tip}", level="warn")
            return
        ok, msg = self._probe_and_show_rynnvalue_reward_status(log=True)
        if ok:
            self._on_mujoco_status(msg)
            return
        backend = self._sim_backend_id()
        self._append_mujoco_log(f"[{backend}] 部署 RynnValue reward_server…")
        self._set_sim_rynnvalue_status("◐ 正在部署 RynnValue…", level="warn")
        self._on_mujoco_status("正在部署 RynnValue…")
        _eai_platform_bootstrap.QApplication.processEvents()
        ok, msg = self._ensure_rynnvalue_reward_server_ready(
            start_if_needed=True,
            wait_s=900.0,
        )
        if ok:
            self._on_mujoco_status(msg)
            return
        fail = f"RynnValue 部署失败: {msg}"
        self._on_mujoco_status(fail)
        self._set_sim_rynnvalue_status(f"○ {msg}", level="err")
        _eai_platform_bootstrap.QMessageBox.warning(self, "RynnValue 部署失败", msg)

    def _on_sim_rynnvalue_stop_clicked(self) -> None:
        msg = self._stop_rynnvalue_model(log=True)
        self._on_mujoco_status(msg)

    def _rynnvalue_deploy_model_path(self) -> str:
        if hasattr(self, "rynnvalue_model_edit"):
            text = self.rynnvalue_model_edit.text().strip()
            if text:
                return text
        return _eai_adapters_models_discovery.resolve_rynnvalue_model()

    def _rynnvalue_deploy_ckpt_path(self) -> str:
        if hasattr(self, "rynnvalue_ckpt_edit"):
            return self.rynnvalue_ckpt_edit.text().strip()
        return _eai_adapters_models_discovery.resolve_rynnvalue_ckpt()

    def _rynnvalue_gpu_preflight(self, *, cuda: str, model: str) -> _eai_platform_bootstrap.Tuple[bool, str]:
        """Refuse to start reward_server if the target GPU cannot hold the model."""
        idx = 0
        raw = (cuda or "0").strip().split(",")[0].strip()
        if raw.isdigit():
            idx = int(raw)
        used, free, total = self._cuda_mem_mib(idx)
        need = 28000 if "8B" in (model or "").upper() else 16000
        pi_on_gpu = False
        pi_pid = ""
        try:
            info = _eai_configuration_models.inspect_pi_server_on_port(_eai_configuration_tabs.MOLMOSPACES_PI_PORT_DEFAULT)
            pi_pid = str((info or {}).get("pid") or "")
        except Exception:
            pi_pid = ""
        if pi_pid:
            mems = self._nvidia_gpu_mem_mib_by_pid()
            if pi_pid in mems:
                pi_on_gpu = True
        if total and free < need:
            extra = ""
            if pi_on_gpu:
                extra = (
                    f" Pi serve（pid={pi_pid}）仍占 GPU{idx}。"
                    "请先在 LIBERO 点「停止 Pi」再部署 RynnValue。"
                )
            return (
                False,
                f"GPU{idx} 显存不足：剩余 {free}MiB / 共 {total}MiB"
                f"（已用 {used}MiB），RynnValue 大约需要 ≥{need}MiB。{extra}",
            )
        if pi_on_gpu and free < need:
            return (
                False,
                f"Pi serve（pid={pi_pid}）占着同一张 GPU，RynnValue 无法加载。"
                "请先点「停止 Pi」。",
            )
        return True, f"GPU{idx} 剩余 {free}MiB"

    def _on_mujoco_install_clicked(self) -> None:
        if self._mujoco_launcher.is_running():
            self._on_mujoco_status("MuJoCo 正在运行")
            return
        self.mujoco_log_edit.clear()
        self._mujoco_launcher.start(
            mjcf_path="",
            mujoco_root=self.mujoco_root_edit.text().strip(),
            python_bin=self.mujoco_python_edit.text().strip(),
            mode=str(self.mujoco_mode_combo.currentData() or "python"),
            install=True,
        )
        self._update_mujoco_ui()

