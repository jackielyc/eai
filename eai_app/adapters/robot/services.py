"""adapters.robot.services — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def build_ros_shell_prefix() -> str:
    parts = ["set -e", "source /opt/ros/humble/setup.bash"]
    for setup in (
        "/opt/psi/rt/a2d-tele/install/setup.bash",
        _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/a2d-tele/install/setup.bash"),
        _eai_platform_bootstrap.os.path.expanduser("~/workspace_liyichao/install/setup.bash"),
    ):
        if _eai_platform_bootstrap.os.path.isfile(setup):
            parts.append(f"source {setup}")
    scripts_dir = _eai_adapters_robot_launch_utils.resolve_scripts_pack_dir()
    if scripts_dir:
        parts.append(f"export A2D_SCRIPTS_DIR={scripts_dir}")
    dds = _eai_platform_bootstrap.os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
    if dds and _eai_platform_bootstrap.os.path.isfile(dds):
        parts.append(f"export FASTRTPS_DEFAULT_PROFILES_FILE={dds}")
    parts.append("export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}")
    return " && ".join(parts)


def a2d_docker_container_name(override: str = "") -> str:
    name = (override or "").strip()
    if name:
        return name
    env = _eai_platform_bootstrap.os.environ.get("A2D_DOCKER_CONTAINER", "").strip()
    return env or _eai_configuration_policies.A2D_DOCKER_CONTAINER_DEFAULT


def a2d_docker_cli_available() -> bool:
    return _eai_platform_bootstrap.shutil.which("docker") is not None


def _a2d_local_workspace_ready() -> bool:
    """宿主机/当前环境是否已有 a2d-tele install（可直接 ros2 launch）。"""
    ws = _eai_adapters_robot_launch_utils.resolve_a2d_workspace_dir()
    return _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(ws, "install", "setup.bash"))


def launch_a2d_ros_detached(
    ros_cmd: str,
    *,
    cwd: _eai_platform_bootstrap.Optional[str] = None,
    extra_env: str = "",
    container: str = "",
    force_local: bool = False,
) -> _eai_platform_bootstrap.Tuple[bool, str]:
    """Detached 拉起一条 ros2 命令。

    - force_local=True：始终在本机执行，绝不 docker exec
    - 已在 a2d 容器内，或本机有 a2d-tele install：在当前环境执行
    - 否则（例如 ./run_local.sh 宿主机 viewer 的 docker 模式）：docker exec 进 A2D 容器执行
    """
    env_prefix = f"{extra_env} && " if extra_env.strip() else ""
    use_local = (
        force_local
        or _eai_perception_segmentation.is_running_in_docker()
        or _a2d_local_workspace_ready()
    )
    if use_local:
        workdir = (cwd or "").strip() or _eai_adapters_robot_launch_utils.resolve_a2d_workspace_dir()
        local_setup = _eai_platform_bootstrap.os.path.join(workdir, "install", "setup.bash")

        if force_local:
            # 无 install 时禁止 chdir/启动，避免 Qt 报 failed to chdir + 裸 humble 找不到包
            if not _eai_platform_bootstrap.os.path.isdir(workdir):
                return (
                    False,
                    f"本机 workspace 不存在: {workdir}\n"
                    "请在 A2D→local「本机环境」填写正确路径，或 export A2D_WORKSPACE=…",
                )
            if not _eai_platform_bootstrap.os.path.isfile(local_setup):
                return (
                    False,
                    f"本机缺少 a2d-tele install: {local_setup}\n"
                    "当前机无 /opt/psi/rt/a2d-tele 时，请 colcon install 到本机，"
                    "或改填真实 workspace / 改用 A2D→docker。",
                )
            parts = [
                "set -e",
                "source /opt/ros/humble/setup.bash",
                f"source {_eai_platform_bootstrap.shlex.quote(local_setup)}",
            ]
            scripts_dir = _eai_adapters_robot_launch_utils.resolve_scripts_pack_dir()
            if scripts_dir:
                parts.append(f"export A2D_SCRIPTS_DIR={_eai_platform_bootstrap.shlex.quote(scripts_dir)}")
            dds = _eai_platform_bootstrap.os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
            if dds and _eai_platform_bootstrap.os.path.isfile(dds):
                parts.append(
                    f"export FASTRTPS_DEFAULT_PROFILES_FILE={_eai_platform_bootstrap.shlex.quote(dds)}"
                )
            # 与 run_local.sh 一致：尊重已有 RMW（勿强行覆盖 cyclonedds）
            parts.append(
                "export RMW_IMPLEMENTATION="
                "${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
            )
            prefix = " && ".join(parts)
        else:
            # 非 force_local：若默认路径不存在，落到 HOME，避免 chdir 失败后仍半拉起进程
            if not _eai_platform_bootstrap.os.path.isdir(workdir):
                workdir = _eai_platform_bootstrap.os.environ.get("HOME") or _eai_platform_bootstrap.os.getcwd() or "/"
            prefix = build_ros_shell_prefix()
        full = f"{prefix} && {env_prefix}{ros_cmd}"
        ok = _eai_platform_bootstrap.QProcess.startDetached("bash", ["-lc", full], workdir)
        if ok:
            where = "本机" if force_local else "本地"
            return True, f"已启动({where}): {ros_cmd}"
        return (
            False,
            f"启动失败: {ros_cmd}（workdir={workdir}；"
            "若见 failed to chdir，请检查 workspace 路径）",
        )

    if not a2d_docker_cli_available():
        return (
            False,
            "宿主机无 a2d-tele install，且无 docker CLI，无法启动 GUI。"
            "请用 run_in_docker.sh、切到 A2D→local，或先 docker start A2D 容器。",
        )
    name = a2d_docker_container_name(container)
    try:
        st = _eai_platform_bootstrap.subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return False, f"检查容器失败: {exc}"
    if st.returncode != 0 or (st.stdout or "").strip().lower() != "true":
        return False, f"容器未运行: {name}（请先「开启容器」）"

    # 容器内路径固定；带上宿主 DISPLAY，避免窗口落到错误 X server
    disp = (_eai_platform_bootstrap.os.environ.get("DISPLAY") or ":1").strip() or ":1"
    inner = (
        "set -e && source /opt/ros/humble/setup.bash && "
        "source /opt/psi/rt/a2d-tele/install/setup.bash && "
        f"export DISPLAY={_eai_platform_bootstrap.shlex.quote(disp)} && "
        "export XAUTHORITY=${XAUTHORITY:-/tmp/.Xauthority} && "
        "export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb} && "
        f"{env_prefix}{ros_cmd}"
    )
    # -d：后台；不抢占当前 viewer 终端
    cmd = ["docker", "exec", "-d", "-e", f"DISPLAY={disp}", name, "bash", "-lc", inner]
    try:
        result = _eai_platform_bootstrap.subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return False, f"docker exec 异常: {exc}"
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip() or f"exit={result.returncode}"
        return False, f"docker exec {name} 失败: {err}"
    return True, f"已在容器 {name} 启动: {ros_cmd}"


def run_a2d_docker_action(
    action: str,
    container: str = "",
    *,
    capture: bool = False,
) -> _eai_platform_bootstrap.Tuple[bool, str]:
    """已禁用：本机永远不用 Docker。"""
    _ = (action, container, capture)
    return False, "已禁用 Docker：请用宿主机 forwarder + host_arm_hand_bridge"
    name = a2d_docker_container_name(container)
    if not a2d_docker_cli_available():
        return (
            False,
            "当前环境无 docker CLI（viewer 多半已在容器内）。"
            f"请到宿主机终端执行: docker {action} {name}",
        )
    if action == "ps":
        cmd = ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Image}}"]
    elif action in ("start", "stop", "restart"):
        cmd = ["docker", action, name]
    else:
        return False, f"未知 docker 动作: {action}"
    try:
        result = _eai_platform_bootstrap.subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        return False, f"docker {action} 异常: {exc}"
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    text = out or err or f"docker {action} exit={result.returncode}"
    if result.returncode != 0:
        return False, text
    if capture or action == "ps":
        return True, text
    return True, f"docker {action} {name} 完成"


def _pgrep_pattern(pattern: str) -> bool:
    try:
        result = _eai_platform_bootstrap.subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            timeout=0.5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _tcp_state_json_age_s(
    path: str = "",
) -> _eai_platform_bootstrap.Tuple[float, bool]:
    """返回 (age_s, ready)。文件缺失时 age 很大、ready=False。"""
    p = path or _eai_platform_bootstrap.os.environ.get(
        "EAI_ROBOT_TCP_STATE",
        _eai_platform_bootstrap.os.path.join(_eai_platform_bootstrap.tempfile.gettempdir(), "eai_robot_tcp_state.json"),
    )
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = _eai_platform_bootstrap.json.load(f)
        updated = float(data.get("updated_at") or 0.0)
        ready = bool(data.get("ready")) and bool(data.get("left")) and bool(
            data.get("right")
        )
        if updated <= 0:
            return 1e9, False
        return max(0.0, _eai_platform_bootstrap.time.time() - updated), ready
    except Exception:
        return 1e9, False


def probe_host_min_node_status() -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, bool, str]]:
    """最小栈节点状态：(显示名, 是否运行, 备注)。

    用于「连接机器人」启停的宿主机节点：相机 / 手 / 臂。
    """
    cam = _pgrep_pattern(r"forwarder/app/bin/forwarder|/forwarder/app/bin/forwarder")
    if not cam:
        cam = _pgrep_pattern(r"a2d_sdk/.*/forwarder|bin/forwarder")
    hand = _pgrep_pattern(r"lib/hand/hand_node|/hand/hand_node")
    bridge = _pgrep_pattern(r"host_arm_hand_bridge\.py")
    fk = _pgrep_pattern(r"host_fk_node\.py") or _pgrep_pattern(
        r"(^|/)fk_node( |$)|lib/wbc/fk_node"
    )
    tcp_br = _pgrep_pattern(r"robot_tcp_bridge\.py")
    age, tcp_ready = _tcp_state_json_age_s()
    if tcp_br and tcp_ready and age < 3.0:
        tcp_note = f"TCP {age:.1f}s"
    elif tcp_br and age < 1e8:
        tcp_note = f"TCP 过期 {age:.0f}s"
    elif tcp_br:
        tcp_note = "无 TCP 数据"
    else:
        tcp_note = ""
    return [
        ("相机 forwarder", cam, "看图"),
        ("灵巧手 hand_node", hand, "控手"),
        ("臂/手使能桥", bridge, "使能+指令"),
        ("FK host_fk", fk, "臂 TCP"),
        ("TCP bridge", tcp_br, tcp_note or "写 JSON"),
    ]


def format_host_min_node_status_text(
    rows: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, bool, str]]] = None,
    *,
    compact: bool = False,
) -> str:
    rows = rows if rows is not None else probe_host_min_node_status()
    if compact:
        # 手臂页高度紧，单行摘要，避免撑破手动控制区
        bits: _eai_platform_bootstrap.List[str] = []
        short = {
            "相机 forwarder": "相机",
            "灵巧手 hand_node": "手",
            "臂/手使能桥": "使能桥",
            "FK host_fk": "FK",
            "TCP bridge": "TCP",
        }
        for name, ok, note in rows:
            mark = "●" if ok else "○"
            label = short.get(name, name)
            if name == "TCP bridge" and note and ok:
                bits.append(f"{mark}{label}({note})")
            else:
                bits.append(f"{mark}{label}")
        n_ok = sum(1 for _n, ok, _note in rows if ok)
        return f"节点 {n_ok}/{len(rows)}  " + "  ".join(bits)
    parts: _eai_platform_bootstrap.List[str] = []
    for name, ok, note in rows:
        mark = "●" if ok else "○"
        state = "运行" if ok else "停止"
        extra = f" · {note}" if note else ""
        parts.append(f"{mark} {name}: {state}{extra}")
    return "\n".join(parts)


def _is_stack_services_running() -> bool:
    """手/臂栈是否在跑（永远不用 Docker）。

    正式路径：arm_hand_services.launch.py / base_services.launch.py。
    本机精简路径：scripts/host_arm_hand_bridge.py（使能 + UDP 中继）。
    """
    if _pgrep_pattern(_eai_configuration_policies.STACK_SERVICES_LAUNCH) or _pgrep_pattern(
        _eai_configuration_policies.LEGACY_BASE_SERVICES_LAUNCH
    ):
        return True
    if _pgrep_pattern("host_arm_hand_bridge.py"):
        return True
    # 兼容旧精简栈（若本机曾装 a2d-tele）
    has_router = _pgrep_pattern("topic_router.launch.py") or _pgrep_pattern(
        "topic_router_node"
    )
    has_arm = _pgrep_pattern("arm_control_node") or _pgrep_pattern(
        "wbc/arm_control"
    )
    return bool(has_router and has_arm)


def _ros_service_exists(service: str) -> bool:
    workspace = _eai_adapters_robot_launch_utils.resolve_a2d_workspace_dir()
    dds = _eai_platform_bootstrap.os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
    env_prefix = ""
    if dds and _eai_platform_bootstrap.os.path.isfile(dds):
        env_prefix = f"export FASTRTPS_DEFAULT_PROFILES_FILE={dds} && "
    cmd = (
        f"{env_prefix}source /opt/ros/humble/setup.bash && "
        f"cd {workspace} && source install/setup.bash && "
        "ros2 service list"
    )
    try:
        result = _eai_platform_bootstrap.subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return service in (result.stdout or "")
    except Exception:
        return False


def _ros2_trigger_service(service: str, timeout_sec: float = 15.0) -> tuple[bool, str]:
    """通过 ros2 CLI 调用 Trigger 服务，避免 Qt 线程内 service client 不可靠。"""
    workspace = _eai_adapters_robot_launch_utils.resolve_a2d_workspace_dir()
    dds = _eai_platform_bootstrap.os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "").strip()
    env_prefix = ""
    if dds and _eai_platform_bootstrap.os.path.isfile(dds):
        env_prefix = f"export FASTRTPS_DEFAULT_PROFILES_FILE={dds} && "
    cmd = (
        f"{env_prefix}source /opt/ros/humble/setup.bash && "
        f"cd {workspace} && source install/setup.bash && "
        f"ros2 service call {service} std_srvs/srv/Trigger '{{}}'"
    )
    try:
        result = _eai_platform_bootstrap.subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.returncode != 0:
            return False, f"exit={result.returncode}: {output[:240]}"
        lower = output.lower()
        if "success=false" in lower or "success: false" in lower:
            return False, output[:240]
        if "success=true" in lower or "success: true" in lower:
            return True, output[:240]
        return True, output[:240] or "OK"
    except _eai_platform_bootstrap.subprocess.TimeoutExpired:
        return False, f"超时 ({timeout_sec}s)"
    except Exception as exc:
        return False, str(exc)


def _stop_existing_rrd_replay() -> None:
    """停止已有 rrd_replay 进程，避免 GUI/viewer 重复启动导致回放无效。"""
    _ros2_trigger_service(_eai_configuration_policies.REPLAY_STOP_SERVICE, timeout_sec=5.0)
    for pattern in ("rrd_replay.launch.py", "rrd_replay/rrd_replay_node", "rrd_replay_node"):
        try:
            _eai_platform_bootstrap.subprocess.run(
                ["pkill", "-f", pattern],
                capture_output=True,
                timeout=1,
                check=False,
            )
        except Exception:
            pass
    for _ in range(30):
        if not (
            _pgrep_pattern("rrd_replay.launch.py")
            or _pgrep_pattern("rrd_replay_node")
        ):
            break
        _eai_platform_bootstrap.time.sleep(0.1)

