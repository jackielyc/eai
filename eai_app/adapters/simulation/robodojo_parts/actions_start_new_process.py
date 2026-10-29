"""adapters.simulation.robodojo parts.actions start new process — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class ActionsStartNewProcessMixin:
    def _start_new_process(
        self,
        task: str,
        *,
        bridge_dir: str = "",
        policy_dir: str = "XPolicyLab/policy/demo_policy",
        ckpt: str = "demo",
        action_type: str = "ee",
        policy_env: str = "",
        eval_env: str = "",
        robodojo_root: str = "",
        display: str = "",
        use_gui: bool = True,
        attach_external: bool = False,
        policy_host: str = "",
        policy_port: int = 0,
    ) -> str:
        if self.is_running():
            self.status_message.emit("RoboDojo 评测已在运行")
            return "busy"
        root = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(robodojo_root or _eai_configuration_policies.ROBODOJO_ROOT_DEFAULT)
        )
        eval_env = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(eval_env or _eai_configuration_policies.ROBODOJO_ENV_DEFAULT)
        )
        policy_env = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(policy_env or eval_env)
        )
        bridge_dir = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser((bridge_dir or "").strip()))
        disp = (display or _eai_platform_bootstrap.os.environ.get("DISPLAY") or ":1.0").strip()
        self._use_gui = bool(use_gui)
        self._keep_alive = True
        self._restart_pending = False
        self._adopted = False
        self._external_pids = []
        host = (policy_host or _eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT).strip() or "127.0.0.1"
        port = int(policy_port or _eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT)
        self._last_start_kwargs = {
            "task": task,
            "bridge_dir": bridge_dir,
            "policy_dir": policy_dir,
            "ckpt": ckpt,
            "action_type": action_type,
            "policy_env": policy_env,
            "eval_env": eval_env,
            "robodojo_root": root,
            "display": disp,
            "use_gui": self._use_gui,
            "attach_external": bool(attach_external),
            "policy_host": host,
            "policy_port": port,
        }
        # 「无」：只启评测客户端；仍用 demo_policy 目录提供 client adapter 脚本
        skip_policy = not (policy_dir or "").strip()
        if skip_policy:
            policy_dir = "XPolicyLab/policy/demo_policy"
            self._last_start_kwargs["policy_dir"] = ""
            attach_external = False
        is_starvla = (not skip_policy) and (
            "starVLA" in policy_dir or "starvla" in policy_dir.lower()
        )
        is_lingbot_va = (not skip_policy) and _eai_configuration_models.is_lingbot_va_policy(policy_dir)
        is_vpp2 = (not skip_policy) and _eai_configuration_models.is_vpp2_policy(policy_dir)
        # attach_external：只起 client，连已部署 server。
        # 否则（一体启动）：本进程先起 server 再起 Isaac（与旧 robodojo.sh eval 相同）。
        use_external = bool(attach_external) and (not skip_policy)

        # Viewer (RoboStack) injects ros-humble into PYTHONPATH; that makes
        # transformers see numpy version as None inside starvla/RoboDojo envs.
        # demo_policy's setup_eval_policy_server.sh also needs CONDA_ROOT when
        # `conda` is not on PATH (path-prefix envs still expand conda info --base).
        # /tmp/isaaclab is often root-owned (755) here → IsaacLab FileHandler
        # PermissionError; mirror run_gui_eval.sh and use /dev/shm TMPDIR.
        conda_base = "/home/psibot/miniconda3"
        tmpdir = "/dev/shm/robodojo_tmp"
        eval_num = _eai_configuration_policies.ROBODOJO_EVAL_NUM_KEEP
        exports = [
            "unset PYTHONPATH PYTHONHOME || true",
            "export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1",
            f"export TMPDIR={_eai_platform_bootstrap.shlex.quote(tmpdir)} TEMP={_eai_platform_bootstrap.shlex.quote(tmpdir)} TMP={_eai_platform_bootstrap.shlex.quote(tmpdir)}",
            f"mkdir -p {_eai_platform_bootstrap.shlex.quote(tmpdir)}/isaaclab/logs",
            f"export CONDA_ROOT={_eai_platform_bootstrap.shlex.quote(conda_base)}",
            f"export CONDA_EXE={_eai_platform_bootstrap.shlex.quote(_eai_platform_bootstrap.os.path.join(conda_base, 'bin', 'conda'))}",
            f"export PATH={_eai_platform_bootstrap.shlex.quote(_eai_platform_bootstrap.os.path.join(conda_base, 'bin'))}:\"${{PATH}}\"",
            f"export ROBODOJO_ENV={_eai_platform_bootstrap.shlex.quote(eval_env)}",
            f"export ROBODOJO_POLICY_ENV={_eai_platform_bootstrap.shlex.quote(policy_env)}",
            f"export ROBODOJO_POLICY_DIR={_eai_platform_bootstrap.shlex.quote(policy_dir)}",
            f"export ROBODOJO_CKPT={_eai_platform_bootstrap.shlex.quote(ckpt)}",
            f"export ROBODOJO_ACTION_TYPE={_eai_platform_bootstrap.shlex.quote(action_type)}",
            f"export EVAL_NUM={_eai_platform_bootstrap.shlex.quote(eval_num)}",
            "unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT "
            "GROUP_RANK ROLE_RANK TORCHELASTIC_RUN_ID PMI_RANK PMI_SIZE 2>/dev/null || true",
        ]
        if skip_policy:
            exports.append("export ROBODOJO_SKIP_POLICY_SERVER=1")
            exports.append("export ROBODOJO_FORCE_PROTOCOL=none")
            skip_policy_port = str(port)
            exports.append(f"export ROBODOJO_POLICY_PORT={_eai_platform_bootstrap.shlex.quote(skip_policy_port)}")
        elif use_external:
            exports.append("export ROBODOJO_SKIP_POLICY_SERVER=1")
            exports.append("unset ROBODOJO_FORCE_PROTOCOL || true")
            exports.append(f"export ROBODOJO_POLICY_HOST={_eai_platform_bootstrap.shlex.quote(host)}")
            exports.append(f"export ROBODOJO_POLICY_PORT={_eai_platform_bootstrap.shlex.quote(str(port))}")
        else:
            exports.append("unset ROBODOJO_SKIP_POLICY_SERVER ROBODOJO_FORCE_PROTOCOL || true")
        if bridge_dir:
            exports.append(f"export ISAAC_CAM_BRIDGE_DIR={_eai_platform_bootstrap.shlex.quote(bridge_dir)}")
            exports.append(f"export EAI_DIR={_eai_platform_bootstrap.shlex.quote(_eai_configuration_models.EAI_DIR)}")

        # GUI on TurboVNC: VirtualGL → NVIDIA (else Mesa llvmpipe for viewport).
        if self._use_gui:
            exports.extend(
                [
                    "export ROBODOJO_USE_VGL=${ROBODOJO_USE_VGL:-1}",
                    "export ROBODOJO_VGL_DEVICE=${ROBODOJO_VGL_DEVICE:-egl}",
                    "export ROBODOJO_ISAAC_HEADLESS=0",
                    "export __GLX_VENDOR_LIBRARY_NAME=nvidia",
                    "export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json",
                ]
            )
        lingbot_va_note = ""
        if is_lingbot_va:
            base_model = _eai_configuration_models.resolve_lingbot_va_base_model(
                _eai_platform_bootstrap.os.environ.get("LINGBOT_VA_BASE_MODEL_PATH", "")
            )
            if not base_model or not _eai_platform_bootstrap.os.path.isdir(base_model):
                msg = (
                    "未设置 LingBot-VA base model。"
                    "请 export LINGBOT_VA_BASE_MODEL_PATH=<lingbot-va-base> "
                    f"或放到默认路径。\n{_eai_configuration_models.lingbot_va_install_hint()}"
                )
                self.status_message.emit(
                    "未设置 LINGBOT_VA_BASE_MODEL_PATH（见日志）"
                )
                self.log_line.emit(f"[ERROR] {msg}")
                return "error"
            va_ckpt = _eai_configuration_models.resolve_lingbot_va_checkpoint(ckpt)
            exports.extend(
                [
                    f"export LINGBOT_VA_BASE_MODEL_PATH={_eai_platform_bootstrap.shlex.quote(base_model)}",
                    f"export BASE_MODEL_PATH={_eai_platform_bootstrap.shlex.quote(base_model)}",
                    f"export LINGBOT_VA_CKPT={_eai_platform_bootstrap.shlex.quote(va_ckpt)}",
                    f"export CHECKPOINT_PATH={_eai_platform_bootstrap.shlex.quote(va_ckpt)}",
                    f"export CONFIG_NAME={_eai_platform_bootstrap.shlex.quote(_eai_configuration_models.va_config_for_backend('isaac'))}",
                    f"export LINGBOT_VA_VA_HOST={_eai_platform_bootstrap.shlex.quote(host if use_external else '127.0.0.1')}",
                    f"export LINGBOT_VA_VA_PORT={_eai_platform_bootstrap.shlex.quote(str(port if use_external else _eai_configuration_models.LINGBOT_VA_PORT_DEFAULT))}",
                    "export VA_SERVER_HOST=\"${LINGBOT_VA_VA_HOST}\"",
                    "export VA_SERVER_PORT=\"${LINGBOT_VA_VA_PORT}\"",
                ]
            )
            lingbot_va_note = (
                f"  LINGBOT_VA_BASE_MODEL_PATH={base_model}"
                f"  ckpt={va_ckpt}"
            )
            self.log_line.emit(
                f"[LingBot-VA] RoboDojo adapter · action=joint 推荐 ·{lingbot_va_note}"
            )
        vpp2_note = ""
        if is_vpp2:
            try:
                _eai_configuration_models.ensure_vpp2_adapter()
            except Exception:
                pass
            try:
                vpp2_paths = _eai_configuration_models.resolve_vpp2_paths()
            except Exception as exc:  # noqa: BLE001
                self.status_message.emit("VPP2 路径解析失败（见日志）")
                self.log_line.emit(f"[ERROR] VPP2: {exc}\n{_eai_configuration_models.vpp2_install_hint()}")
                return "error"
            bundle = _eai_configuration_models.resolve_vpp2_bundle(ckpt) if ckpt else vpp2_paths.bundle
            wan = _eai_configuration_models.resolve_vpp2_wan_root()
            weights = str(_eai_configuration_models.resolve_vpp2_weights())
            exports.extend(
                [
                    f"export VPP2_WEIGHTS={_eai_platform_bootstrap.shlex.quote(weights)}",
                    f"export VPP2_BUNDLE={_eai_platform_bootstrap.shlex.quote(bundle)}",
                    f"export VPP2_WAN_ROOT={_eai_platform_bootstrap.shlex.quote(wan)}",
                    f"export VPP2_ROOT={_eai_platform_bootstrap.shlex.quote(str(vpp2_paths.code_root))}",
                    "export VPP2_TORCH_LOAD_MMAP=1",
                    f"export ROBODOJO_POLICY_PORT={_eai_platform_bootstrap.shlex.quote(str(port if use_external else _eai_configuration_models.DEFAULT_VPP2_PORT))}",
                ]
            )
            vpp2_note = f"  VPP2_BUNDLE={bundle}  wan={wan}"
            self.log_line.emit(
                f"[VPP2] RoboDojo adapter · action=ee 必须 ·{vpp2_note}"
            )
        starvla_ckpt_note = ""
        if is_starvla:
            variant = _eai_configuration_policies.resolve_starvla_hf_variant(ckpt)
            ckpt_path = _eai_configuration_policies.resolve_starvla_ckpt_file(variant, ckpt=ckpt)
            if not ckpt_path:
                msg = (
                    f"未找到 starVLA HF 权重（variant={variant}）。"
                    f"请准备 {_eai_configuration_policies.STARVLA_PI_V3_CKPT_FILE_DEFAULT}"
                    " 或设置 STARVLA_CKPT_PATH"
                )
                self.status_message.emit(msg)
                self.log_line.emit(f"[ERROR] {msg}")
                return "error"
            hf_root = (
                _eai_platform_bootstrap.os.environ.get("STARVLA_HF_ROOT", "").strip()
                or _eai_configuration_policies.ROBODOJO_STARVLA_HF_ROOT_DEFAULT
            )
            base_vlm = (
                _eai_platform_bootstrap.os.environ.get("STARVLA_BASE_VLM", "").strip()
                or _eai_configuration_policies.ROBODOJO_STARVLA_BASE_VLM_DEFAULT
            )
            starvla_env = (
                policy_env
                if policy_env and _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(policy_env, "bin", "python"))
                else _eai_configuration_policies.ROBODOJO_STARVLA_ENV_DEFAULT
            )
            exports.extend(
                [
                    f"export STARVLA_ENV={_eai_platform_bootstrap.shlex.quote(starvla_env)}",
                    f"export STARVLA_HF_ROOT={_eai_platform_bootstrap.shlex.quote(hf_root)}",
                    f"export STARVLA_BASE_VLM={_eai_platform_bootstrap.shlex.quote(base_vlm)}",
                    f"export STARVLA_CKPT_PATH={_eai_platform_bootstrap.shlex.quote(ckpt_path)}",
                    "export STARVLA_HF_VERIFY_ONLY=1",
                    "export STARVLA_HF_SKIP_WEIGHT_HASH=1",
                    "export STARVLA_ROBODOJO_NUM_ENVS=1",
                    "export STARVLA_INCLUDE_STATE=True",
                    "export STARVLA_UNNORM_KEY=arx_x5",
                    f"export STARVLA_ROBODOJO_HEADLESS={'1' if not self._use_gui else '0'}",
                ]
            )
            starvla_ckpt_note = (
                f"  STARVLA_CKPT_PATH={ckpt_path}  STARVLA_ENV={starvla_env}"
            )
            # 已单独部署策略时：走通用 GUI/eval（只起 client），勿再跑 run_gui_starvla
            # （后者会再起一套 server）。
            if use_external and self._use_gui:
                gui_script = _eai_platform_bootstrap.os.path.join(root, "scripts", "run_gui_eval.sh")
                if not _eai_platform_bootstrap.os.path.isfile(gui_script):
                    self.status_message.emit(f"未找到脚本: {gui_script}")
                    return "error"
                exports.extend(
                    [
                        f"export ROBODOJO_DISPLAY={_eai_platform_bootstrap.shlex.quote(disp)}",
                        f"export DISPLAY={_eai_platform_bootstrap.shlex.quote(disp)}",
                    ]
                )
                run_cmd = f"exec bash {_eai_platform_bootstrap.shlex.quote(gui_script)} {_eai_platform_bootstrap.shlex.quote(task)}"
                log_cmd = (
                    f"$ bash scripts/run_gui_eval.sh {task}  "
                    f"# GUI client → {host}:{port}"
                )
                status_msg = f"正在启动评测（复用已部署策略 {host}:{port} / GUI）: {task}…"
            elif use_external and not self._use_gui:
                robodojo_sh = _eai_platform_bootstrap.os.path.join(root, "scripts", "robodojo.sh")
                if not _eai_platform_bootstrap.os.path.isfile(robodojo_sh):
                    self.status_message.emit(f"未找到脚本: {robodojo_sh}")
                    return "error"
                run_cmd = (
                    f"exec bash {_eai_platform_bootstrap.shlex.quote(robodojo_sh)} client "
                    f"--policy-dir {_eai_platform_bootstrap.shlex.quote(policy_dir)} "
                    f"--task {_eai_platform_bootstrap.shlex.quote(task)} "
                    f"--ckpt {_eai_platform_bootstrap.shlex.quote(ckpt)} "
                    f"--policy-host {_eai_platform_bootstrap.shlex.quote(host)} "
                    f"--policy-port {int(port)} "
                    f"--eval-num {_eai_platform_bootstrap.shlex.quote(eval_num)} "
                    f"--action-type {_eai_platform_bootstrap.shlex.quote(action_type)} "
                    f"--seed 0"
                )
                log_cmd = (
                    f"$ bash scripts/robodojo.sh client --task {task} "
                    f"--policy-host {host} --policy-port {port}  # headless"
                )
                status_msg = (
                    f"正在启动评测（复用已部署策略 {host}:{port} / headless）: {task}…"
                )
            elif self._use_gui:
                gui_script = _eai_platform_bootstrap.os.path.join(root, "scripts", "run_gui_starvla_pi_v3.sh")
                if not _eai_platform_bootstrap.os.path.isfile(gui_script):
                    self.status_message.emit(f"未找到脚本: {gui_script}")
                    return "error"
                exports.extend(
                    [
                        f"export ROBODOJO_DISPLAY={_eai_platform_bootstrap.shlex.quote(disp)}",
                        f"export DISPLAY={_eai_platform_bootstrap.shlex.quote(disp)}",
                    ]
                )
                run_cmd = f"exec bash {_eai_platform_bootstrap.shlex.quote(gui_script)} {_eai_platform_bootstrap.shlex.quote(task)}"
                log_cmd = (
                    f"$ bash scripts/run_gui_starvla_pi_v3.sh {task}  # GUI starVLA"
                )
                status_msg = f"正在启动 starVLA GUI 评测: {task}…"
            else:
                hf_eval = _eai_platform_bootstrap.os.path.join(
                    root,
                    "XPolicyLab",
                    "policy",
                    "starVLA",
                    "scripts",
                    "eval_hf_robodojo.sh",
                )
                if not _eai_platform_bootstrap.os.path.isfile(hf_eval):
                    self.status_message.emit(f"未找到脚本: {hf_eval}")
                    return "error"
                run_cmd = (
                    f"exec bash {_eai_platform_bootstrap.shlex.quote(hf_eval)} {_eai_platform_bootstrap.shlex.quote(variant)} "
                    f"{_eai_platform_bootstrap.shlex.quote(task)} 0 0 0 "
                    f"{_eai_platform_bootstrap.shlex.quote(policy_env)} {_eai_platform_bootstrap.shlex.quote(eval_env)} "
                    f"{_eai_platform_bootstrap.shlex.quote(eval_num)}"
                )
                log_cmd = (
                    f"$ bash …/eval_hf_robodojo.sh {variant} {task} … "
                    f"# headless starVLA EVAL_NUM={eval_num}"
                )
                status_msg = f"正在启动 starVLA headless 评测: {task}…"
        elif self._use_gui:
            gui_script = _eai_platform_bootstrap.os.path.join(root, "scripts", "run_gui_eval.sh")
            if not _eai_platform_bootstrap.os.path.isfile(gui_script):
                self.status_message.emit(f"未找到脚本: {gui_script}")
                return "error"
            exports.extend(
                [
                    f"export ROBODOJO_DISPLAY={_eai_platform_bootstrap.shlex.quote(disp)}",
                    f"export DISPLAY={_eai_platform_bootstrap.shlex.quote(disp)}",
                ]
            )
            run_cmd = f"exec bash {_eai_platform_bootstrap.shlex.quote(gui_script)} {_eai_platform_bootstrap.shlex.quote(task)}"
            log_cmd = f"$ bash scripts/run_gui_eval.sh {task}  # GUI"
            if skip_policy:
                status_msg = f"正在启动评测（无策略 / GUI）: {task}…"
                log_cmd += "  # skip policy server"
            elif use_external:
                status_msg = (
                    f"正在启动评测（复用已部署策略 {host}:{port} / GUI）: {task}…"
                )
                log_cmd += f"  # client → {host}:{port}"
            else:
                status_msg = f"正在启动 RoboDojo GUI 评测: {task}…"
        else:
            robodojo_sh = _eai_platform_bootstrap.os.path.join(root, "scripts", "robodojo.sh")
            if not _eai_platform_bootstrap.os.path.isfile(robodojo_sh):
                self.status_message.emit(f"未找到脚本: {robodojo_sh}")
                return "error"
            if use_external:
                run_cmd = (
                    f"exec bash {_eai_platform_bootstrap.shlex.quote(robodojo_sh)} client "
                    f"--policy-dir {_eai_platform_bootstrap.shlex.quote(policy_dir)} "
                    f"--task {_eai_platform_bootstrap.shlex.quote(task)} "
                    f"--ckpt {_eai_platform_bootstrap.shlex.quote(ckpt)} "
                    f"--policy-host {_eai_platform_bootstrap.shlex.quote(host)} "
                    f"--policy-port {int(port)} "
                    f"--eval-num {_eai_platform_bootstrap.shlex.quote(eval_num)} "
                    f"--action-type {_eai_platform_bootstrap.shlex.quote(action_type)} "
                    f"--seed 0"
                )
                log_cmd = (
                    f"$ bash scripts/robodojo.sh client --task {task} "
                    f"--policy-host {host} --policy-port {port}"
                )
                status_msg = (
                    f"正在启动评测（复用已部署策略 {host}:{port} / headless）: {task}…"
                )
            else:
                run_cmd = (
                    f"exec bash {_eai_platform_bootstrap.shlex.quote(robodojo_sh)} eval "
                    f"--policy-dir {_eai_platform_bootstrap.shlex.quote(policy_dir)} "
                    f"--task {_eai_platform_bootstrap.shlex.quote(task)} "
                    f"--ckpt {_eai_platform_bootstrap.shlex.quote(ckpt)} "
                    f"--policy-env {_eai_platform_bootstrap.shlex.quote(policy_env)} "
                    f"--eval-env {_eai_platform_bootstrap.shlex.quote(eval_env)} "
                    f"--eval-num {_eai_platform_bootstrap.shlex.quote(eval_num)} "
                    f"--action-type {_eai_platform_bootstrap.shlex.quote(action_type)} "
                    f"--seed 0"
                )
                log_cmd = (
                    f"$ bash scripts/robodojo.sh eval --task {task} "
                    f"--policy-dir {policy_dir} --ckpt {ckpt} "
                    f"--eval-num {eval_num}  # headless keep"
                )
                if skip_policy:
                    status_msg = f"正在启动评测（无策略 / headless）: {task}…"
                    log_cmd += "  # skip policy server"
                else:
                    status_msg = f"正在启动 RoboDojo headless 评测: {task}…"

        cmd = " && ".join(exports) + f" && cd {_eai_platform_bootstrap.shlex.quote(root)} && {run_cmd}"
        # 与策略服务一样：真实评测/Isaac 进独立 session；UI 只 tail 日志。
        # 关闭 EAI 只杀 tail，Isaac 继续跑，下次启动可复用。
        if self._keep_on_exit:
            shell, pid_path = _eai_adapters_models_qwen_local.persistent_model_shell(cmd, self.PERSIST_SLOT)
            self._persist_pid_path = pid_path
            launch_cmd = shell
        else:
            launch_cmd = cmd
        self._stopping = False
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.setWorkingDirectory(root)
        # Drop RoboStack PYTHONPATH at the process boundary (not only in bash).
        qenv = _eai_platform_bootstrap.QProcessEnvironment.systemEnvironment()
        qenv.remove("PYTHONPATH")
        qenv.remove("PYTHONHOME")
        qenv.insert("PYTHONNOUSERSITE", "1")
        qenv.insert("TMPDIR", tmpdir)
        qenv.insert("TEMP", tmpdir)
        qenv.insert("TMP", tmpdir)
        qenv.insert("CONDA_ROOT", conda_base)
        qenv.insert("CONDA_EXE", _eai_platform_bootstrap.os.path.join(conda_base, "bin", "conda"))
        if skip_policy:
            qenv.insert("ROBODOJO_SKIP_POLICY_SERVER", "1")
            qenv.insert("ROBODOJO_FORCE_PROTOCOL", "none")
            qenv.insert("ROBODOJO_POLICY_PORT", str(port))
        elif use_external:
            qenv.insert("ROBODOJO_SKIP_POLICY_SERVER", "1")
            qenv.remove("ROBODOJO_FORCE_PROTOCOL")
            qenv.insert("ROBODOJO_POLICY_HOST", host)
            qenv.insert("ROBODOJO_POLICY_PORT", str(port))
        else:
            qenv.remove("ROBODOJO_SKIP_POLICY_SERVER")
            qenv.remove("ROBODOJO_FORCE_PROTOCOL")
        qenv.insert("EVAL_NUM", eval_num)
        if self._use_gui:
            qenv.insert("ROBODOJO_USE_VGL", qenv.value("ROBODOJO_USE_VGL") or "1")
            qenv.insert("ROBODOJO_VGL_DEVICE", qenv.value("ROBODOJO_VGL_DEVICE") or "egl")
            qenv.insert("ROBODOJO_ISAAC_HEADLESS", "0")
            qenv.insert("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
            if not (qenv.value("VK_ICD_FILENAMES") or "").strip():
                qenv.insert(
                    "VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json"
                )
        conda_bin = _eai_platform_bootstrap.os.path.join(conda_base, "bin")
        path_now = qenv.value("PATH", "")
        if conda_bin not in path_now.split(":"):
            qenv.insert("PATH", f"{conda_bin}:{path_now}" if path_now else conda_bin)
        try:
            _eai_platform_bootstrap.os.makedirs(tmpdir, exist_ok=True)
        except OSError:
            pass
        proc.setProcessEnvironment(qenv)
        # setsid：stop() 可杀进程组；persist 时外层只是 log tail。
        proc.start("setsid", ["bash", "-lc", launch_cmd])
        self._process = proc
        self._pgid = None
        if proc.waitForStarted(5000):
            # persist 模式下真 session pid 在 pid 文件；稍后再读。
            if self._keep_on_exit:
                for _ in range(20):
                    persist_pid = _eai_adapters_models_qwen_local.read_persistent_pid(self._persist_pid_path)
                    if persist_pid > 1:
                        self._pgid = persist_pid
                        break
                    _eai_platform_bootstrap.time.sleep(0.05)
            if self._pgid is None:
                try:
                    self._pgid = int(proc.processId())
                except Exception:
                    self._pgid = None
        self.running_changed.emit(True)
        self.log_line.emit(log_cmd)
        if starvla_ckpt_note:
            self.log_line.emit(starvla_ckpt_note)
        if lingbot_va_note:
            self.log_line.emit(lingbot_va_note)
        if vpp2_note:
            self.log_line.emit(vpp2_note)
        policy_label = "(无 / 进程内零动作，不启策略服务)" if skip_policy else policy_dir
        self.log_line.emit(
            f"  mode={'GUI' if self._use_gui else 'headless'} "
            f"policy={policy_label} ckpt={ckpt} action={action_type} "
            f"EVAL_NUM={eval_num} bridge={bridge_dir or '(unset)'}"
        )
        if self._use_gui:
            self.log_line.emit(
                "  OpenGL: VirtualGL→NVIDIA "
                f"(ROBODOJO_USE_VGL={qenv.value('ROBODOJO_USE_VGL') or '1'}, "
                f"device={qenv.value('ROBODOJO_VGL_DEVICE') or 'egl'})"
            )
        if skip_policy:
            self.log_line.emit(
                "  策略=无：仅启评测；Isaac 使用进程内零动作，不会连接 WebSocket 策略服务"
            )
        self.log_line.emit(
            "  常驻模式：评测/Isaac 保持运行，直到点击「停止评测」；"
            "关闭 EAI 界面不会停止 Isaac Sim"
        )
        self.status_message.emit(status_msg)
        return "started"

    @staticmethod
    def switch_task_request_path(bridge_dir: str = "") -> str:
        bridge = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(
                (bridge_dir or "").strip() or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT
            )
        )
        return _eai_platform_bootstrap.os.path.join(bridge, "switch_task.request")

    def current_task(self) -> str:
        kw = self._last_start_kwargs or {}
        return str(kw.get("task") or "").strip()

    def request_switch_task(self, task: str, bridge_dir: str = "") -> bool:
        """热切换任务：写请求文件；评测在 episode 步进中检测并尽快换场景（不重启 Kit）。"""
        task = (task or "").strip()
        if not task:
            self.status_message.emit("请选择评测任务")
            return False
        if not self.is_running():
            self.status_message.emit("评测未在运行，请先启动")
            return False
        cur = self.current_task()
        if cur and cur == task:
            self.status_message.emit(f"已在任务 {task} 上运行")
            self.log_line.emit(f"切换任务：已是 {task}，忽略")
            return False
        path = self.switch_task_request_path(bridge_dir)
        try:
            _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(path) or ".", exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(task + "\n")
            _eai_platform_bootstrap.os.replace(tmp, path)
        except OSError as exc:
            self.status_message.emit(f"写切换请求失败: {exc}")
            self.log_line.emit(f"[ERROR] switch_task: {exc}")
            return False
        if self._last_start_kwargs is not None:
            self._last_start_kwargs["task"] = task
        self.log_line.emit(
            f"已请求热切换任务: {cur or '?'} → {task}"
            "（不重启 Isaac；数秒内应中止当前 episode 并换场景）"
        )
        self.status_message.emit(f"正在切换任务 → {task}…")
        self._switch_watch_path = path
        self._switch_watch_task = task
        self._switch_watch_tries = 0
        if not hasattr(self, "_switch_watch_timer") or self._switch_watch_timer is None:
            self._switch_watch_timer = _eai_platform_bootstrap.QTimer(self)
            self._switch_watch_timer.setInterval(2000)
            self._switch_watch_timer.timeout.connect(self._on_switch_watch_tick)
        if not self._switch_watch_timer.isActive():
            self._switch_watch_timer.start()
        return True

    def _on_switch_watch_tick(self) -> None:
        path = getattr(self, "_switch_watch_path", "") or ""
        task = getattr(self, "_switch_watch_task", "") or ""
        self._switch_watch_tries = int(getattr(self, "_switch_watch_tries", 0)) + 1
        still = bool(path and _eai_platform_bootstrap.os.path.isfile(path))
        if not still:
            self.log_line.emit(f"切换请求已被评测进程接收 → {task}")
            self.status_message.emit(f"已切换任务 → {task}")
            if getattr(self, "_switch_watch_timer", None) is not None:
                self._switch_watch_timer.stop()
            return
        if self._switch_watch_tries >= 15:
            self.log_line.emit(
                f"[WARN] 切换请求仍未被消费（{path}）。"
                "当前评测进程可能是旧代码：请点「停止评测」后重新「启动评测」再切换。"
            )
            self.status_message.emit("切换未生效：请停止后重开评测")
            if getattr(self, "_switch_watch_timer", None) is not None:
                self._switch_watch_timer.stop()
            return
        if self._switch_watch_tries in (1, 3, 6):
            self.log_line.emit(
                f"仍在等待评测消费切换请求 → {task}（{self._switch_watch_tries * 2}s）…"
            )

    def stop(self) -> None:
        """停止本窗口管理的评测，并清掉系统里所有仿真/eval 残留。"""
        self._keep_alive = False
        self._restart_pending = False
        if getattr(self, "_adopt_watch_timer", None) is not None:
            self._adopt_watch_timer.stop()
        if getattr(self, "_switch_watch_timer", None) is not None:
            self._switch_watch_timer.stop()

        clients = self.list_running_eval_clients()
        for c in clients:
            pid = int(c["pid"])
            if pid not in self._external_pids:
                self._external_pids.append(pid)
        persist_pid = _eai_adapters_models_qwen_local.read_persistent_pid(self._persist_pid_path)
        if persist_pid > 1 and persist_pid not in self._external_pids:
            self._external_pids.append(persist_pid)
        if self._external_pids and self._pgid is None:
            try:
                self._pgid = _eai_platform_bootstrap.os.getpgid(self._external_pids[0])
            except OSError:
                self._pgid = self._external_pids[0]

        if self._stopping:
            self.status_message.emit("正在强制停止全部仿真…")
            self.log_line.emit("--- 再次停止：对全部仿真发 SIGKILL ---")
            self._force_kill_all_sims()
            return

        self._stopping = True
        n = len(clients)
        kits = self.list_running_isaac_sims()
        self.status_message.emit(
            f"正在停止全部仿真（eval_client={n}, IsaacKit={len(kits)}；可再点强制结束）…"
        )
        self.log_line.emit(
            f"--- 请求停止全部仿真（eval_client={n}, IsaacKit={len(kits)}）---"
        )
        for c in clients:
            self.log_line.emit(
                f"  stop target pid={c['pid']} task={c.get('task') or '?'}"
            )
        if persist_pid > 1:
            self.log_line.emit(f"  stop persist session pid={persist_pid}")
            _eai_adapters_models_qwen_local.stop_persistent_model(self._persist_pid_path)
        self.running_changed.emit(True)
        self._signal_process_group(_eai_platform_bootstrap.signal.SIGTERM)
        killed = self._pkill_eval_leftovers(sig="-TERM")
        if killed:
            self.log_line.emit(f"  已对仿真相关进程组发 SIGTERM（patterns hit≈{killed}）")
        _eai_platform_bootstrap.QTimer.singleShot(2000, self._force_kill_all_sims)

    def _force_kill_all_sims(self) -> None:
        """SIGKILL 本窗口进程组 + 全部仿真残留，并复位 UI 状态。"""
        self._keep_alive = False
        self._restart_pending = False
        self._stopping = True
        _eai_adapters_models_qwen_local.stop_persistent_model(self._persist_pid_path)
        self._signal_process_group(_eai_platform_bootstrap.signal.SIGKILL)
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            try:
                self._process.kill()
                self._process.waitForFinished(1500)
            except Exception:
                pass
        killed = self._pkill_eval_leftovers(sig="-KILL")
        for c in self.list_running_eval_clients():
            pid = int(c["pid"])
            try:
                _eai_platform_bootstrap.os.kill(pid, _eai_platform_bootstrap.signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                pass
            try:
                _eai_platform_bootstrap.os.killpg(_eai_platform_bootstrap.os.getpgid(pid), _eai_platform_bootstrap.signal.SIGKILL)
            except Exception:
                pass
        for k in self.list_running_isaac_sims():
            pid = int(k["pid"])
            try:
                _eai_platform_bootstrap.os.kill(pid, _eai_platform_bootstrap.signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                pass
            try:
                _eai_platform_bootstrap.os.killpg(_eai_platform_bootstrap.os.getpgid(pid), _eai_platform_bootstrap.signal.SIGKILL)
            except Exception:
                pass
        self._process = None
        self._pgid = None
        self._external_pids = []
        self._adopted = False
        if getattr(self, "_adopt_watch_timer", None) is not None:
            self._adopt_watch_timer.stop()
        self._stopping = False
        left_clients = self.list_running_eval_clients()
        left_kits = self.list_running_isaac_sims()
        if left_clients or left_kits:
            bits = []
            if left_clients:
                bits.append(
                    "eval="
                    + ", ".join(
                        f"{c['pid']}:{c.get('task') or '?'}" for c in left_clients
                    )
                )
            if left_kits:
                bits.append(
                    "kit=" + ", ".join(str(k["pid"]) for k in left_kits[:6])
                )
            self.log_line.emit("[WARN] 强制停止后仍有残留: " + "; ".join(bits))
            self.status_message.emit(
                f"仍有仿真残留（client={len(left_clients)} kit={len(left_kits)}），请再点一次停止"
            )
            self.running_changed.emit(False)
            return
        self.log_line.emit(f"--- 全部仿真已停止（pkill hits≈{killed}）---")
        self.status_message.emit("全部仿真已停止")
        self.running_changed.emit(False)

    def _signal_process_group(self, sig: int) -> None:
        pgid = self._pgid
        if pgid is None and self._process is not None:
            try:
                pgid = int(self._process.processId())
            except Exception:
                pgid = None
        if pgid is None and self._external_pids:
            try:
                pgid = _eai_platform_bootstrap.os.getpgid(self._external_pids[0])
            except OSError:
                pgid = self._external_pids[0]
        signaled_pgids = set()
        if pgid and pgid > 1:
            try:
                _eai_platform_bootstrap.os.killpg(pgid, sig)
                signaled_pgids.add(pgid)
            except ProcessLookupError:
                pass
            except Exception as exc:
                self.log_line.emit(f"killpg({pgid}, {sig}) 失败: {exc}")
        for pid in list(self._external_pids):
            try:
                child_pgid = _eai_platform_bootstrap.os.getpgid(pid)
                if child_pgid > 1 and child_pgid not in signaled_pgids:
                    _eai_platform_bootstrap.os.killpg(child_pgid, sig)
                    signaled_pgids.add(child_pgid)
            except Exception:
                pass
            try:
                _eai_platform_bootstrap.os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except Exception as exc:
                self.log_line.emit(f"kill({pid}, {sig}) 失败: {exc}")
        if self._process is not None and self._process.state() != _eai_platform_bootstrap.QProcess.NotRunning:
            if sig == _eai_platform_bootstrap.signal.SIGKILL:
                self._process.kill()
            else:
                self._process.terminate()

    def shutdown(self) -> None:
        """EAI 退出：只断开本窗口对评测的管理，不停止 Isaac Sim。

        显式「停止评测」仍走 stop()/_force_kill_all_sims()。
        """
        self._keep_alive = False
        self._restart_pending = False
        self._stopping = False
        if getattr(self, "_adopt_watch_timer", None) is not None:
            self._adopt_watch_timer.stop()
        if getattr(self, "_switch_watch_timer", None) is not None:
            self._switch_watch_timer.stop()
        proc = self._process
        self._process = None
        if proc is not None:
            for signal_name, slot in (
                ("readyReadStandardOutput", self._on_process_output),
                ("finished", self._on_process_finished),
                ("errorOccurred", self._on_process_error),
            ):
                try:
                    getattr(proc, signal_name).disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            if proc.state() != _eai_platform_bootstrap.QProcess.NotRunning:
                # 仅结束日志 tail；persist session / Isaac 继续跑。
                try:
                    proc.terminate()
                    proc.waitForFinished(400)
                except Exception:
                    pass
            try:
                proc.deleteLater()
            except Exception:
                pass
        self._pgid = None
        self._external_pids = []
        self._adopted = False
        if self._keep_on_exit:
            persist_pid = _eai_adapters_models_qwen_local.read_persistent_pid(self._persist_pid_path)
            kits = self.list_running_isaac_sims()
            clients = self.list_running_eval_clients()
            if persist_pid > 1 or kits or clients:
                self.log_line.emit(
                    "--- EAI 退出：保留 Isaac/评测 "
                    f"(persist={persist_pid or '-'}, "
                    f"eval_client={len(clients)}, IsaacKit={len(kits)}) ---"
                )
            return
        # 非保活模式（兼容）：仍强制清理
        self._force_kill_all_sims()

    def _on_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        for line in data.splitlines():
            if line:
                self.log_line.emit(line.rstrip())

    def _schedule_keepalive_restart(self) -> None:
        if not self._keep_alive or self._stopping or not self._last_start_kwargs:
            self._restart_pending = False
            self.running_changed.emit(False)
            return
        self._restart_pending = True
        self.running_changed.emit(True)
        self.log_line.emit("--- 评测进程退出，常驻模式 3s 后自动重启 ---")
        self.status_message.emit("评测常驻：即将自动重启…")
        _eai_platform_bootstrap.QTimer.singleShot(3000, self._restart_keepalive)

