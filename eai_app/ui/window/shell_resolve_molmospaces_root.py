"""ui.window.shell resolve molmospaces root — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class ShellResolveMolmospacesRootMixin:
    def _resolve_molmospaces_root(self) -> str:
        raw = ""
        if hasattr(self, "molmospaces_root_edit"):
            raw = self.molmospaces_root_edit.text().strip()
        backend = self._sim_backend_id() if hasattr(self, "_sim_backend_id") else ""
        if backend == "psi_molmospaces":
            raw = (
                raw
                or _eai_platform_bootstrap.os.environ.get("PSI_MOLMOSPACES_ROOT", "")
                or _eai_platform_bootstrap.os.environ.get("MOLMOSPACES_ROOT", "")
                or _eai_configuration_tabs.PSI_MOLMOSPACES_ROOT_DEFAULT
            )
        else:
            raw = raw or _eai_platform_bootstrap.os.environ.get("MOLMOSPACES_ROOT", "") or _eai_configuration_tabs.MOLMOSPACES_ROOT_DEFAULT
        return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(raw))

    def _sync_spaces_backend_defaults(self, backend: str) -> None:
        """Swap MolmoSpaces / Psi-MolmoSpaces root + default policy when radios flip."""
        if not hasattr(self, "molmospaces_root_edit"):
            return
        cur = self.molmospaces_root_edit.text().strip()
        cur_norm = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(cur)) if cur else ""
        molmo_default = _eai_platform_bootstrap.os.path.abspath(_eai_configuration_tabs.MOLMOSPACES_ROOT_DEFAULT)
        psi_default = _eai_platform_bootstrap.os.path.abspath(_eai_configuration_tabs.PSI_MOLMOSPACES_ROOT_DEFAULT)
        if backend == "psi_molmospaces":
            if (not cur_norm) or cur_norm == molmo_default or cur_norm.rstrip(
                "/"
            ).endswith("/molmospaces"):
                self.molmospaces_root_edit.setText(_eai_configuration_tabs.PSI_MOLMOSPACES_ROOT_DEFAULT)
            if hasattr(self, "molmospaces_recipe_combo"):
                idx = self.molmospaces_recipe_combo.findData("eval_task")
                if idx >= 0:
                    self.molmospaces_recipe_combo.setCurrentIndex(idx)
            if hasattr(self, "molmospaces_robot_combo"):
                idx = self.molmospaces_robot_combo.findData("psi_r1")
                if idx >= 0:
                    self.molmospaces_robot_combo.setCurrentIndex(idx)
            if hasattr(self, "molmospaces_policy_combo"):
                cur_pol = str(self.molmospaces_policy_combo.currentData() or "")
                # Keep an already-selected Psi/WAM remote policy; only fill default
                # when the current choice does not need a deployable server.
                if not _eai_adapters_models_discovery.needs_psi_remote_deploy(cur_pol):
                    idx = self.molmospaces_policy_combo.findData(
                        _eai_configuration_tabs.PSI_MOLMOSPACES_DEFAULT_POLICY
                    )
                    if idx >= 0:
                        self.molmospaces_policy_combo.setCurrentIndex(idx)
            if hasattr(self, "molmospaces_root_edit"):
                self.molmospaces_root_edit.setToolTip(
                    "Psi-MolmoSpaces 仓库路径（默认 "
                    f"{_eai_configuration_tabs.PSI_MOLMOSPACES_ROOT_DEFAULT}）"
                )
        elif backend == "molmospaces":
            if (not cur_norm) or cur_norm == psi_default or cur_norm.rstrip(
                "/"
            ).endswith("/psi-molmospaces"):
                self.molmospaces_root_edit.setText(_eai_configuration_tabs.MOLMOSPACES_ROOT_DEFAULT)
            if hasattr(self, "molmospaces_policy_combo"):
                idx = self.molmospaces_policy_combo.findData(
                    "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig"
                )
                if idx >= 0:
                    self.molmospaces_policy_combo.setCurrentIndex(idx)
            if hasattr(self, "molmospaces_root_edit"):
                self.molmospaces_root_edit.setToolTip(
                    "MolmoSpaces 仓库路径（默认 "
                    f"{_eai_configuration_tabs.MOLMOSPACES_ROOT_DEFAULT}）"
                )
        if hasattr(self, "_refresh_molmospaces_tasks"):
            self._refresh_molmospaces_tasks()
        if hasattr(self, "_on_molmospaces_policy_changed"):
            self._on_molmospaces_policy_changed()

    def _resolve_molmospaces_python(self) -> str:
        # Prefer MuJoCo tab Python if it looks like molmospaces env; else default.
        tab_py = ""
        if hasattr(self, "mujoco_python_edit"):
            tab_py = self.mujoco_python_edit.text().strip()
        for cand in (
            tab_py,
            _eai_platform_bootstrap.os.environ.get("MOLMOSPACES_PYTHON", ""),
            _eai_configuration_tabs.MOLMOSPACES_PYTHON_DEFAULT,
            _eai_adapters_models_discovery.resolve_mujoco_python(),
        ):
            c = (cand or "").strip()
            if c and _eai_platform_bootstrap.os.path.isfile(c):
                return _eai_platform_bootstrap.os.path.abspath(c)
        return "python3"

    def _refresh_molmospaces_tasks(self) -> None:
        if not hasattr(self, "molmospaces_task_combo"):
            return
        root = self._resolve_molmospaces_root() if hasattr(self, "_resolve_molmospaces_root") else ""
        current = self.molmospaces_task_combo.currentData()
        current_id = ""
        if isinstance(current, dict):
            current_id = str(current.get("id") or "")
        backend = self._sim_backend_id() if hasattr(self, "_sim_backend_id") else ""
        psi_only = backend == "psi_molmospaces"
        tasks = _eai_adapters_models_discovery.list_molmospaces_eval_tasks(root, psi_only=psi_only)
        self.molmospaces_task_combo.blockSignals(True)
        self.molmospaces_task_combo.clear()
        ready_n = 0
        for task in tasks:
            label = str(task["label"])
            if task.get("ready"):
                ready_n += 1
            else:
                label = f"{label} · 未下载"
            self.molmospaces_task_combo.addItem(label, task)
        if current_id:
            for i in range(self.molmospaces_task_combo.count()):
                data = self.molmospaces_task_combo.itemData(i)
                if isinstance(data, dict) and data.get("id") == current_id:
                    self.molmospaces_task_combo.setCurrentIndex(i)
                    break
        self.molmospaces_task_combo.blockSignals(False)
        _eai_configuration_models.stamp_eval_task_combo(self.molmospaces_task_combo)
        if hasattr(self, "molmospaces_task_thumb"):
            self.molmospaces_task_thumb.show_combo_current(self.molmospaces_task_combo)
        tag = "psi_molmospaces" if psi_only else "molmospaces"
        self._append_mujoco_log(
            f"[{tag}] 已列出 {len(tasks)} 个评测任务（本地数据就绪 {ready_n}）"
        )

    def _repair_molmospaces_cache_manifest(self) -> _eai_platform_bootstrap.List[str]:
        """Fix cache dirs missing from MolmoSpaces manifest.

        - Non-empty/complete orphans → register in manifest
        - Tiny/incomplete remnants → delete so setup can re-download
        """
        cache = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "molmo-spaces-resources")
        manifest_path = _eai_platform_bootstrap.os.path.join(
            cache, "mjthor_data_type_to_source_to_versions.json"
        )
        if not _eai_platform_bootstrap.os.path.isfile(manifest_path):
            return []
        try:
            import json

            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception as exc:
            return [f"无法读取 manifest: {exc}"]

        notes: _eai_platform_bootstrap.List[str] = []
        for data_type in ("robots", "scenes", "objects", "grasps", "materials"):
            type_dir = _eai_platform_bootstrap.os.path.join(cache, data_type)
            if not _eai_platform_bootstrap.os.path.isdir(type_dir):
                continue
            manifest.setdefault(data_type, {})
            try:
                sources = _eai_platform_bootstrap.os.listdir(type_dir)
            except OSError:
                continue
            for source in sources:
                source_dir = _eai_platform_bootstrap.os.path.join(type_dir, source)
                if not _eai_platform_bootstrap.os.path.isdir(source_dir):
                    continue
                recorded = set(manifest[data_type].setdefault(source, []) or [])
                try:
                    versions = _eai_platform_bootstrap.os.listdir(source_dir)
                except OSError:
                    continue
                for ver in versions:
                    ver_dir = _eai_platform_bootstrap.os.path.join(source_dir, ver)
                    if not _eai_platform_bootstrap.os.path.isdir(ver_dir) or ver in recorded:
                        continue
                    try:
                        names = set(_eai_platform_bootstrap.os.listdir(ver_dir))
                        size = 0
                        for root_d, _dirs, files in _eai_platform_bootstrap.os.walk(ver_dir):
                            for fn in files:
                                try:
                                    size += _eai_platform_bootstrap.os.path.getsize(_eai_platform_bootstrap.os.path.join(root_d, fn))
                                except OSError:
                                    pass
                    except OSError:
                        names, size = set(), 0
                    looks_ok = size > 1_000_000 and (
                        "assets" in names
                        or "mjthor_resources_combined_meta.json.gz" in names
                        or any(
                            n.endswith(".xml") or n.endswith(".urdf") for n in names
                        )
                    )
                    if looks_ok:
                        manifest[data_type][source].append(ver)
                        manifest[data_type][source] = sorted(
                            set(manifest[data_type][source])
                        )
                        notes.append(f"register {data_type}/{source}/{ver}")
                    else:
                        try:
                            _eai_platform_bootstrap.shutil.rmtree(ver_dir, ignore_errors=True)
                            notes.append(f"remove {data_type}/{source}/{ver}")
                        except Exception as exc:
                            notes.append(
                                f"remove-failed {data_type}/{source}/{ver}: {exc}"
                            )

        if not notes:
            return []
        try:
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
                fh.write("\n")
        except Exception as exc:
            return [f"无法写入 manifest: {exc}"]
        return notes

    def _check_molmospaces_resources(self, py: str) -> _eai_platform_bootstrap.Tuple[bool, str]:
        """Return (ok, message). Requires molmospaces-resources >= MOLMOSPACES_RESOURCES_MIN."""
        code = (
            "from importlib.metadata import version\n"
            "from packaging.version import Version\n"
            f"min_v = {_eai_configuration_tabs.MOLMOSPACES_RESOURCES_MIN!r}\n"
            "try:\n"
            "    v = version('molmospaces-resources')\n"
            "except Exception as e:\n"
            "    print('MISSING:' + str(e)); raise SystemExit(2)\n"
            "print(v)\n"
            "raise SystemExit(0 if Version(v) >= Version(min_v) else 3)\n"
        )
        try:
            import subprocess

            r = subprocess.run(
                [py, "-c", code],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception as exc:
            return False, f"无法检测 molmospaces-resources: {exc}"
        ver = (r.stdout or "").strip().splitlines()[-1] if r.stdout else "?"
        if r.returncode == 0:
            return True, ver
        if r.returncode == 2:
            return False, "未安装 molmospaces-resources"
        if r.returncode == 3:
            return (
                False,
                f"molmospaces-resources={ver} < 需要 {_eai_configuration_tabs.MOLMOSPACES_RESOURCES_MIN}",
            )
        err = (r.stderr or r.stdout or "").strip()
        return False, err or f"检测失败 code={r.returncode}"

    def _launch_vpp2_backend(
        self,
        backend: str,
        *,
        task: str = "",
        task_suite: str = "",
        label: str = "",
    ) -> bool:
        """Start VPP2 for any sim backend. Returns True if launched."""
        try:
            _eai_configuration_models.ensure_vpp2_adapter()
        except Exception:
            pass
        ok_v, v_msg = _eai_configuration_models.probe_vpp2()
        self._append_mujoco_log(f"[{backend}/VPP2] {v_msg}")
        if not ok_v:
            self._append_mujoco_log(_eai_configuration_models.vpp2_install_hint())
            self._on_mujoco_status("VPP2 未就绪（见日志）")
            return False
        try:
            argv, cwd, env_extra, note = _eai_configuration_models.build_backend_vpp2_argv(
                backend,
                python_bin=_eai_configuration_models.resolve_vpp2_python(),
                port=_eai_configuration_models.DEFAULT_VPP2_PORT,
                task=task,
                task_suite=task_suite,
            )
        except Exception as exc:  # noqa: BLE001
            self._on_mujoco_status(f"无法组装 VPP2 命令: {exc}")
            self._append_mujoco_log(str(exc))
            return False
        clean_env = {k: v for k, v in env_extra.items() if v is not None}
        clean_env["EAI_DIR"] = _eai_configuration_models.EAI_DIR
        try:
            clean_env.update(self._sim_bridge_child_env())
        except Exception:
            pass
        self._append_mujoco_log(note)
        self._append_mujoco_log("cmd: " + " ".join(argv))
        level = _eai_configuration_models.vpp2_support_level(backend)
        if level == "serve_only":
            self._on_mujoco_status(
                f"VPP2 policy server 启动中（{backend} · serve_only）"
            )
        else:
            self._on_mujoco_status(f"VPP2 评测启动 · {backend}")
        self._mujoco_launcher.start_cwd_command(
            cwd=str(cwd),
            argv=argv,
            label=label or f"vpp2:{backend}",
            env_extra=clean_env,
        )
        self._update_mujoco_ui()
        return True

    def _launch_lingbot_va_backend(
        self,
        backend: str,
        *,
        task: str = "",
        task_suite: str = "",
        checkpoint: str = "",
        label: str = "",
    ) -> bool:
        """Start LingBot-VA for any sim backend. Returns True if launched."""
        ok_va, va_msg = _eai_configuration_models.probe_lingbot_va()
        self._append_mujoco_log(f"[{backend}/LingBot-VA] {va_msg}")
        if not ok_va:
            self._append_mujoco_log(_eai_configuration_models.lingbot_va_install_hint())
            self._on_mujoco_status("LingBot-VA 未就绪（见日志）")
            return False

        va_host = (
            _eai_platform_bootstrap.os.environ.get("LINGBOT_VA_VA_HOST", "").strip()
            or _eai_platform_bootstrap.os.environ.get("VA_SERVER_HOST", "").strip()
            or "127.0.0.1"
        )
        try:
            va_port = int(
                _eai_platform_bootstrap.os.environ.get("LINGBOT_VA_VA_PORT")
                or _eai_platform_bootstrap.os.environ.get("VA_SERVER_PORT")
                or _eai_configuration_models.LINGBOT_VA_PORT_DEFAULT
            )
        except (TypeError, ValueError):
            va_port = int(_eai_configuration_models.LINGBOT_VA_PORT_DEFAULT)
        # Reuse an already-deployed wan_va instead of loading a second copy (OOM).
        if _eai_configuration_models.port_listening(va_host, va_port):
            self._append_mujoco_log(
                f"[{backend}/LingBot-VA] wan_va 已在 "
                f"ws://{va_host}:{va_port}，跳过重复启动"
            )
            self._on_mujoco_status(
                f"LingBot-VA 已部署 ws://{va_host}:{va_port}（复用）"
            )
            self._update_mujoco_ui()
            return True

        try:
            argv, cwd, env_extra, note = _eai_configuration_models.build_backend_va_argv(
                backend,
                checkpoint=checkpoint or _eai_configuration_models.resolve_lingbot_va_checkpoint(""),
                task=task,
                task_suite=task_suite,
                python_bin=_eai_configuration_models.resolve_lingbot_va_python(),
                start_server=True,
                port=va_port,
            )
        except Exception as exc:  # noqa: BLE001
            self._on_mujoco_status(f"无法组装 LingBot-VA 命令: {exc}")
            self._append_mujoco_log(str(exc))
            return False
        clean_env = {k: v for k, v in env_extra.items() if v is not None}
        clean_env["EAI_DIR"] = _eai_configuration_models.EAI_DIR
        try:
            clean_env.update(self._sim_bridge_child_env())
        except Exception:
            pass
        self._append_mujoco_log(note)
        self._append_mujoco_log("cmd: " + " ".join(argv))
        level = _eai_configuration_models.va_support_level(backend)
        if level == "serve_only":
            self._on_mujoco_status(
                f"LingBot-VA wan_va 服务启动中（{backend} · serve_only）"
            )
        else:
            self._on_mujoco_status(f"LingBot-VA 评测启动 · {backend}")
        self._mujoco_launcher.start_cwd_command(
            cwd=str(cwd),
            argv=argv,
            label=label or f"lingbot-va:{backend}",
            env_extra=clean_env,
        )
        self._update_mujoco_ui()
        return True

    def _nvidia_gpu_mem_mib_by_pid(self) -> _eai_platform_bootstrap.Dict[str, str]:
        now = _eai_platform_bootstrap.time.time()
        ts, cache = getattr(self, "_pi_gpu_mem_cache", (0.0, {}))
        if cache and now - float(ts) < 2.0:
            return cache
        out: _eai_platform_bootstrap.Dict[str, str] = {}
        try:
            raw = _eai_platform_bootstrap.subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=1.5,
                stderr=_eai_platform_bootstrap.subprocess.DEVNULL,
            )
            for line in raw.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit():
                    out[parts[0]] = parts[1].replace("MiB", "").strip()
        except Exception:
            out = cache if isinstance(cache, dict) else {}
        self._pi_gpu_mem_cache = (now, out)
        return out

    def _set_libero_pi_status(self, text: str, *, level: str = "muted") -> None:
        if not hasattr(self, "libero_pi_status_label"):
            return
        self.libero_pi_status_label.setText(text)
        self.libero_pi_status_label.setStyleSheet(
            f"color: {self._deploy_status_color(level)};"
        )

    def _set_molmospaces_pi_status(self, text: str, *, level: str = "muted") -> None:
        if not hasattr(self, "molmospaces_pi_status_label"):
            return
        self.molmospaces_pi_status_label.setText(text)
        self.molmospaces_pi_status_label.setStyleSheet(
            f"color: {self._deploy_status_color(level)};"
        )

    def _set_molmospaces_psi_status(self, text: str, *, level: str = "muted") -> None:
        if not hasattr(self, "molmospaces_psi_status_label"):
            return
        self.molmospaces_psi_status_label.setText(text)
        self.molmospaces_psi_status_label.setStyleSheet(
            f"color: {self._deploy_status_color(level)};"
        )

    def _update_molmospaces_pi_deploy_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_pi_policy_launcher", None)
            and self._pi_policy_launcher.is_running()
        )
        host = _eai_configuration_models.PI_SERVER_HOST_DEFAULT
        port = _eai_configuration_tabs.MOLMOSPACES_PI_PORT_DEFAULT
        listening = _eai_configuration_models.port_listening(host, port)
        info = {"pid": "", "config": "", "dir": ""}
        try:
            info = _eai_configuration_models.inspect_pi_server_on_port(port) or info
        except Exception:
            pass
        pid = str(info.get("pid") or "")
        cfg = str(info.get("config") or "")
        gpu_mib = ""
        if pid:
            gpu_mib = (self._nvidia_gpu_mem_mib_by_pid().get(pid) or "").strip()
        if hasattr(self, "molmospaces_pi_start_btn"):
            self.molmospaces_pi_start_btn.setEnabled(not running)
            if running and listening:
                self.molmospaces_pi_start_btn.setText("已就绪")
            elif running:
                self.molmospaces_pi_start_btn.setText("部署中…")
            else:
                self.molmospaces_pi_start_btn.setText("部署启动")
        if hasattr(self, "molmospaces_pi_stop_btn"):
            self.molmospaces_pi_stop_btn.setEnabled(running or listening)
        eval_busy = bool(
            getattr(self, "_mujoco_launcher", None)
            and self._mujoco_launcher.is_running()
        )
        if hasattr(self, "libero_pi_start_btn"):
            if running and not listening:
                self.libero_pi_start_btn.setText("部署中…")
                self.libero_pi_start_btn.setEnabled(False)
            elif listening:
                self.libero_pi_start_btn.setText("已部署")
                self.libero_pi_start_btn.setEnabled(not running and not eval_busy)
            else:
                self.libero_pi_start_btn.setText("部署 Pi")
                self.libero_pi_start_btn.setEnabled(not eval_busy)
        if hasattr(self, "libero_pi_stop_btn"):
            self.libero_pi_stop_btn.setEnabled(running or listening)
        bits = []
        if cfg:
            bits.append(cfg)
        if pid:
            bits.append(f"pid={pid}")
        if gpu_mib:
            bits.append(f"{gpu_mib}MiB")
        extra = (" " + " ".join(bits)) if bits else ""
        if running and not listening:
            status = f"Pi: ◐ 部署中（加载权重）… ws://{host}:{port}{extra}"
            self._set_libero_pi_status(f"◐ 部署中（加载权重）…{extra}", level="warn")
            self._set_molmospaces_pi_status(status, level="warn")
        elif listening:
            owned = "本窗口托管" if running else "可复用"
            status = f"Pi: ● 已就绪 ws://{host}:{port}{extra} — {owned}"
            self._set_libero_pi_status(
                f"● 运行中 ws://{host}:{port}{extra}", level="ok"
            )
            self._set_molmospaces_pi_status(status, level="ok")
        else:
            self._set_libero_pi_status("○ 未部署", level="muted")
            self._set_molmospaces_pi_status("Pi: ○ 未部署", level="muted")

    def _update_molmospaces_psi_deploy_ui(self, *_args) -> None:
        if not hasattr(self, "molmospaces_psi_start_btn"):
            return
        mode = self._molmospaces_psi_mode()
        mode_label = {
            "wam82": "WAM",
            "native6d": "Native6D",
            "psi_policy": "PsiPolicy",
        }.get(mode or "", mode or "")
        host = "localhost"
        port = self._molmospaces_psi_port()
        ckpt = self._molmospaces_psi_ckpt_text()
        ckpt_name = _eai_platform_bootstrap.os.path.basename(ckpt) if ckpt else ""
        gpu = ""
        if hasattr(self, "molmospaces_psi_gpu_spin"):
            gpu = f" gpu={int(self.molmospaces_psi_gpu_spin.value())}"
        running = bool(
            getattr(self, "_psi_remote_launcher", None)
            and self._psi_remote_launcher.is_running()
        )
        listening = _eai_configuration_models.port_listening(host, port) if mode else False
        if running and listening:
            self.molmospaces_psi_start_btn.setText("已就绪")
            self.molmospaces_psi_start_btn.setEnabled(False)
        elif running:
            self.molmospaces_psi_start_btn.setText("部署中…")
            self.molmospaces_psi_start_btn.setEnabled(False)
        else:
            self.molmospaces_psi_start_btn.setText("部署启动")
            self.molmospaces_psi_start_btn.setEnabled(bool(mode))
        if hasattr(self, "molmospaces_psi_stop_btn"):
            self.molmospaces_psi_stop_btn.setEnabled(running or listening)
        if hasattr(self, "molmospaces_psi_probe_btn"):
            self.molmospaces_psi_probe_btn.setEnabled(bool(mode))

        detail_bits = []
        if mode_label:
            detail_bits.append(mode_label)
        detail_bits.append(f":{port}")
        if ckpt_name:
            detail_bits.append(ckpt_name)
        if gpu:
            detail_bits.append(gpu.strip())
        detail = " · ".join(detail_bits)
        if not mode:
            self._set_molmospaces_psi_status(
                "Psi/WAM: ○ 当前策略无需部署", level="muted"
            )
        elif running and listening:
            self._psi_remote_last_error = ""
            self._set_molmospaces_psi_status(
                f"Psi/WAM: ● 已就绪 ws://{host}:{port} — {detail}",
                level="ok",
            )
            if hasattr(self, "_psi_remote_ready_timer") and self._psi_remote_ready_timer.isActive():
                self._psi_remote_ready_timer.stop()
        elif running:
            self._set_molmospaces_psi_status(
                f"Psi/WAM: ◐ 部署中（加载权重）… {detail}",
                level="warn",
            )
        elif listening:
            self._set_molmospaces_psi_status(
                f"Psi/WAM: ● 端口已监听（可复用） ws://{host}:{port} — {detail}",
                level="ok",
            )
        else:
            launcher = getattr(self, "_psi_remote_launcher", None)
            exit_code = getattr(launcher, "last_exit_code", None) if launcher else None
            last_err = str(getattr(self, "_psi_remote_last_error", "") or "").strip()
            user_stopped = bool(getattr(self, "_psi_remote_user_stopped", False))
            if last_err:
                self._set_molmospaces_psi_status(
                    f"Psi/WAM: ✕ 部署失败 — {last_err}",
                    level="err",
                )
            elif (not user_stopped) and exit_code not in (None, 0):
                self._set_molmospaces_psi_status(
                    f"Psi/WAM: ✕ 进程退出 code={exit_code} — {detail}",
                    level="err",
                )
            else:
                self._set_molmospaces_psi_status(
                    f"Psi/WAM: ○ 未部署 — {detail}",
                    level="muted",
                )

    def _warn_if_unlikely_wam_checkpoint(self, ckpt: str, mode: str) -> str:
        """Return a warning string for checkpoints that are almost certainly wrong."""
        if mode != "wam82":
            return ""
        name = _eai_platform_bootstrap.os.path.basename(ckpt or "").lower()
        path_l = (ckpt or "").replace("\\", "/").lower()
        bad_names = ("scheduler.pt", "optimizer.pt", "rng_state.pt", "trainer.pt")
        if name in bad_names or "/checkpoint-" in path_l and name == "scheduler.pt":
            return (
                f"当前文件 {_eai_platform_bootstrap.os.path.basename(ckpt)} 不像 WAM policy 权重；"
                "请选择 step=*.pt（例如 step=00016000-train_loss=0.349923.pt）"
            )
        if "qwen" in path_l and "wam" not in path_l and "native6d" not in path_l:
            return (
                "路径像是 Qwen/LLM 训练目录，不是 WAM finetune checkpoint；"
                "请改选 pick_apple_tomato_*_wam_finetune 下的 step=*.pt"
            )
        return ""

    def _stop_psi_remote_for_relaunch(self, port: int, *, mode: str) -> None:
        """Stop UI-managed and stray listeners on ``port``, then wait until free."""
        if getattr(self, "_psi_remote_launcher", None) is not None:
            self._append_mujoco_log(f"[psi-remote/{mode}] 强制清理 UI 托管进程…")
            self._psi_remote_launcher.force_clear()
        _eai_configuration_models.stop_pi_server_on_port(port, log=self._append_mujoco_log)
        deadline = _eai_platform_bootstrap.time.time() + 8.0
        while _eai_platform_bootstrap.time.time() < deadline:
            _eai_platform_bootstrap.QApplication.processEvents()
            launcher_busy = bool(
                getattr(self, "_psi_remote_launcher", None)
                and self._psi_remote_launcher.is_running()
            )
            if (not launcher_busy) and (not _eai_configuration_models.port_listening("localhost", port, timeout=0.15)):
                break
            _eai_platform_bootstrap.time.sleep(0.2)
        # Last resort: clear launcher state even if QProcess is stuck reporting Running.
        if getattr(self, "_psi_remote_launcher", None) is not None:
            if self._psi_remote_launcher.is_running():
                self._psi_remote_launcher.force_clear()
        self._update_molmospaces_psi_deploy_ui()

    def _start_psi_remote_server_async(
        self, *, prefer_ckpt: str = "", force_restart: bool = False
    ) -> _eai_platform_bootstrap.Tuple[bool, str]:
        """Launch Psi/WAM server without blocking the Qt event loop."""
        mode = self._molmospaces_psi_mode()
        if not mode:
            return False, "当前策略不是 Psi Policy / Native6D / WAM"
        host = "localhost"
        port = self._molmospaces_psi_port()
        ckpt_raw = (prefer_ckpt or "").strip() or self._molmospaces_psi_ckpt_text()
        ckpt = _eai_adapters_models_discovery.resolve_psi_remote_checkpoint(ckpt_raw)
        if hasattr(self, "molmospaces_psi_ckpt_edit") and ckpt:
            self.molmospaces_psi_ckpt_edit.setText(ckpt)
        if not ckpt or not _eai_platform_bootstrap.os.path.isfile(ckpt):
            return False, f"请先选择有效 checkpoint（当前: {ckpt_raw or '<空>'}）"
        warn = self._warn_if_unlikely_wam_checkpoint(ckpt, mode)
        if warn:
            return False, warn
        need_clean = force_restart or (
            mode == "wam82"
            and _eai_configuration_models.port_listening(host, port, timeout=0.3)
            and _eai_adapters_models_discovery.wam82_server_needs_restart(port)
        )
        if need_clean:
            self._append_mujoco_log(
                f"[psi-remote/{mode}] 停止旧服务并用干净环境重启"
                "（env -u LD_PRELOAD）…"
            )
            self._stop_psi_remote_for_relaunch(port, mode=mode)
        elif _eai_configuration_models.port_listening(host, port, timeout=0.3):
            blob = _eai_adapters_models_qwen_local.listener_cmdline(port)
            ckpt_base = _eai_platform_bootstrap.os.path.basename(ckpt.rstrip("/")) if ckpt else ""
            same_ckpt = False
            other_ckpt = False
            if blob and ckpt:
                norm_ckpt = _eai_adapters_models_qwen_local._norm_model_token(ckpt)
                same_ckpt = ckpt in blob or (norm_ckpt and norm_ckpt in blob) or (
                    bool(ckpt_base) and ckpt_base in blob
                )
                other_ckpt = (not same_ckpt) and (
                    ".pt" in blob or "checkpoint" in blob.lower()
                )
            self._update_molmospaces_psi_deploy_ui()
            if same_ckpt:
                return (
                    True,
                    f"同一模型已在运行（:{port} {ckpt_base}），跳过重复部署",
                )
            if other_ckpt:
                return (
                    False,
                    f":{port} 上的服务不是当前 checkpoint"
                    f"（{ckpt_base or ckpt}）。请先「部署停止」再部署。",
                )
            return True, f"端口 :{port} 已在监听，跳过重复部署"
        elif (
            getattr(self, "_psi_remote_launcher", None)
            and self._psi_remote_launcher.is_running()
        ):
            self._update_molmospaces_psi_deploy_ui()
            return True, f"服务已在启动中（:{port}），请等待按钮变为「已就绪」"
        molmo_root = self._resolve_molmospaces_root()
        policy_root = _eai_adapters_models_discovery.resolve_psi_remote_policy_root()
        gpu = "0"
        if hasattr(self, "molmospaces_psi_gpu_spin"):
            gpu = str(int(self.molmospaces_psi_gpu_spin.value()))
        if mode != "wam82" and not policy_root:
            return False, "未找到 psi-policy 仓库（设置 PSI_POLICY_ROOT / PSI_POLICY_DIR）"
        if mode == "wam82" and not _eai_adapters_models_discovery.resolve_wan22_dir():
            self._append_mujoco_log(
                "[psi-remote/wam82] 警告: 未找到 Wan2.2-TI2V-5B，"
                "请设置 WAN22_TI2V_5B_CHECKPOINT_DIR"
            )
        try:
            argv, cwd, env_extra, label = _eai_adapters_models_discovery.build_psi_remote_server_argv(
                mode=mode,
                checkpoint=ckpt,
                molmospaces_root=molmo_root,
                psi_policy_root=policy_root,
                host="0.0.0.0",
                port=port,
                gpu_id=gpu,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"无法组装服务命令: {exc}"
        self._append_mujoco_log(
            f"[psi-remote/{mode}] checkpoint={ckpt} port={port} gpu={gpu}"
        )
        self._append_mujoco_log(f"[psi-remote/{mode}] 后台启动 {label} @ :{port}")
        if env_extra.get("WAN22_TI2V_5B_CHECKPOINT_DIR"):
            self._append_mujoco_log(
                f"[psi-remote/{mode}] WAN={env_extra['WAN22_TI2V_5B_CHECKPOINT_DIR']}"
            )
        if env_extra.get("WAM_PROMPT_EMBEDDING_DIR"):
            self._append_mujoco_log(
                f"[psi-remote/{mode}] prompt_emb={env_extra['WAM_PROMPT_EMBEDDING_DIR']}"
            )
        self._append_mujoco_log(
            f"[psi-remote/{mode}] WAM/Psi 加载可能需数分钟，界面不应卡住；"
            "请等按钮变为「已就绪」后再评测"
        )
        self._psi_remote_launcher.start_cwd_command(
            cwd=cwd,
            argv=argv,
            label=label,
            env_extra=env_extra,
        )
        if hasattr(self, "_psi_remote_ready_timer"):
            self._psi_remote_ready_timer.start()
        self._update_molmospaces_psi_deploy_ui()
        return True, f"已开始部署（:{port}），加载中请稍候"

    def _ensure_psi_remote_policy_ready(
        self,
        *,
        start_if_needed: bool = True,
        wait_s: float = 30.0,
        prefer_ckpt: str = "",
    ) -> _eai_platform_bootstrap.Tuple[bool, str]:
        """Check Psi/WAM readiness; optionally kick off async deploy (no long UI freeze)."""
        mode = self._molmospaces_psi_mode()
        if not mode:
            return False, "当前策略不是 Psi Policy / Native6D / WAM"
        host = "localhost"
        port = self._molmospaces_psi_port()
        ckpt_raw = (prefer_ckpt or "").strip() or self._molmospaces_psi_ckpt_text()
        ckpt = _eai_adapters_models_discovery.resolve_psi_remote_checkpoint(ckpt_raw)
        warn = self._warn_if_unlikely_wam_checkpoint(ckpt, mode)
        if warn:
            return False, warn
        py = _eai_adapters_models_discovery.resolve_psi_remote_python(_eai_adapters_models_discovery.resolve_psi_remote_policy_root())
        force_restart = bool(
            mode == "wam82"
            and _eai_configuration_models.port_listening(host, port, timeout=0.3)
            and _eai_adapters_models_discovery.wam82_server_needs_restart(port)
        )
        if force_restart:
            self._append_mujoco_log(
                f"[psi-remote/{mode}] 预检发现旧/污染的 WAM 服务，强制重启"
                "（env -u LD_PRELOAD，避免 cudnnGetVersion 崩溃）…"
            )
            if not start_if_needed:
                self._stop_psi_remote_for_relaunch(port, mode=mode)
                return False, f"Psi/WAM 需重启: ws://{host}:{port}"
            ok, msg = self._start_psi_remote_server_async(
                prefer_ckpt=ckpt, force_restart=True
            )
            if not ok:
                return False, msg
            return (
                False,
                f"已强制重启部署（:{port}）。"
                "请等按钮变为「已就绪」后再点启动评测（WAM 加载常需数分钟）。",
            )
        if _eai_configuration_models.port_listening(host, port, timeout=0.3):
            ok, msg = _eai_adapters_models_discovery.probe_psi_remote_server(
                mode, host, port, python_bin=py, timeout_s=2.5
            )
            if ok:
                return True, msg
            running = bool(
                getattr(self, "_psi_remote_launcher", None)
                and self._psi_remote_launcher.is_running()
            )
            if running:
                return (
                    False,
                    f"服务仍在加载（:{port} 已监听但握手未完成）。"
                    "请等「已就绪」后再点启动评测。",
                )
            return False, msg
        running = bool(
            getattr(self, "_psi_remote_launcher", None)
            and self._psi_remote_launcher.is_running()
        )
        if running:
            return (
                False,
                f"服务部署中（:{port}）。请等按钮变为「已就绪」后再评测。",
            )
        if not start_if_needed:
            return False, f"Psi/WAM 未就绪: ws://{host}:{port}"
        ok, msg = self._start_psi_remote_server_async(prefer_ckpt=ckpt)
        if not ok:
            return False, msg
        # Brief non-blocking-ish wait: keep processEvents, but never hang for minutes.
        deadline = _eai_platform_bootstrap.time.time() + min(20.0, max(5.0, float(wait_s)))
        last = msg
        while _eai_platform_bootstrap.time.time() < deadline:
            _eai_platform_bootstrap.QApplication.processEvents()
            self._update_molmospaces_psi_deploy_ui()
            if _eai_configuration_models.port_listening(host, port, timeout=0.2):
                ok, last = _eai_adapters_models_discovery.probe_psi_remote_server(
                    mode, host, port, python_bin=py, timeout_s=2.0
                )
                if ok:
                    return True, last
            if not self._psi_remote_launcher.is_running() and not _eai_configuration_models.port_listening(
                host, port, timeout=0.2
            ):
                return False, "Psi/WAM 服务进程已退出，请查看上方日志"
            _eai_platform_bootstrap.time.sleep(0.5)
        return (
            False,
            f"已开始部署但尚未就绪（:{port}）。"
            "WAM 加载常需数分钟：请等「已就绪」后再点启动评测。",
        )

