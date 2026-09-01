#!/usr/bin/env bash
# 在 Docker 容器内运行 viewer（ROS 在 Docker 内时必须用此脚本）
#
# 原因: Docker 内 ROS 使用 SHM 共享内存传输，图像数据无法传到宿主机进程。
#       topic 列表能发现，但收不到图像帧。
#
# 用法:
#   bash run_in_docker.sh
#   A2D_DOCKER_CONTAINER=my-container bash run_in_docker.sh
#
# 环境变量:
#   PSIBOT_HOME              宿主机目录，挂载到容器内同路径（默认 /home/psibot）
#   A2D_DOCKER_CONTAINER     容器名
#   A2D_DOCKER_RECREATE=1    强制重建容器以应用挂载/用户（慎用）
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 解析软链得到真实工程目录（宿主机常为 ~/workspace_liyichao/eai -> share_data/...）
# 容器内 /home/psibot bind 常未生效，必须走已挂载的真实路径，否则 images/ 等资源缺失
EAI_DIR_HOST="$(readlink -f "${SCRIPT_DIR}")"
CONTAINER="${A2D_DOCKER_CONTAINER:-a2d-tele-release-2-1-0rc3-latest}"
PSIBOT_HOME="${PSIBOT_HOME:-/home/psibot}"
PSIBOT_HOME_CONTAINER="${PSIBOT_HOME_CONTAINER:-${PSIBOT_HOME}}"
REMOTE_DIR="${EAI_DIR_HOST}"
A2D_SCRIPTS_DIR="${A2D_SCRIPTS_DIR:-${PSIBOT_HOME}/workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts}"
WORKSPACE_INSTALL="${PSIBOT_HOME}/workspace_liyichao/install/setup.bash"
A2D_SDK_HOME="${A2D_SDK_HOME:-${PSIBOT_HOME}/a2d_sdk}"
FASTDDS_XML="${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"
# 用宿主机 psibot 的 uid/gid 启动（镜像内 psibot 常为 1000，与宿主机不一致）
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
CONTAINER_USER="${HOST_UID}:${HOST_GID}"
# IME：容器禁挂 /tmp，改用 host network 下的 abstract unix socket 转发 fcitx dbus
IME_RUNTIME_DIR="${IME_RUNTIME_DIR:-/tmp/a2d_ime}"
IME_DBUS_ABSTRACT="${IME_DBUS_ABSTRACT:-a2d_ime_dbus}"
SOCAT_BIN="${SOCAT_BIN:-$HOME/.local/bin/socat}"

resolve_socat() {
    if [[ -x "${SOCAT_BIN}" ]]; then printf '%s\n' "${SOCAT_BIN}"; return 0; fi
    if command -v socat >/dev/null 2>&1; then command -v socat; return 0; fi
    return 1
}

container_exists() {
    docker inspect "${CONTAINER}" >/dev/null 2>&1
}

container_has_psibot_mount() {
    docker inspect "${CONTAINER}" --format '{{range .Mounts}}{{if eq .Destination "'"${PSIBOT_HOME_CONTAINER}"'"}}{{.Source}}{{end}}{{end}}' \
        | grep -q .
}


container_runs_as_psibot() {
    local user
    user="$(docker inspect "${CONTAINER}" --format '{{.Config.User}}' 2>/dev/null || true)"
    [[ "${user}" == "${CONTAINER_USER}" || "${user}" == "${HOST_UID}" ]]
}

container_has_ime_mount() {
    docker inspect "${CONTAINER}" --format '{{range .Mounts}}{{if eq .Destination "'"${IME_RUNTIME_DIR}"'"}}{{.Source}}{{end}}{{end}}'         | grep -q .
}

save_container_config() {
    local tmp
    tmp="$(mktemp)"
    docker inspect "${CONTAINER}" > "${tmp}"
    echo "${tmp}"
}

recreate_container_with_psibot_mount() {
    if ! container_exists; then
        echo "错误: 容器 ${CONTAINER} 不存在，无法重建" >&2
        exit 1
    fi

    local inspect_json image workdir entrypoint_json cmd_json privileged
    inspect_json="$(save_container_config)"

    image="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[0]['Config']['Image'])" "${inspect_json}")"
    workdir="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[0]['Config'].get('WorkingDir') or '')" "${inspect_json}")"
    entrypoint_json="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d[0]['Config'].get('Entrypoint') or []))" "${inspect_json}")"
    cmd_json="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d[0]['Config'].get('Cmd') or []))" "${inspect_json}")"
    privileged="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('1' if d[0]['HostConfig'].get('Privileged') else '0')" "${inspect_json}")"

    local -a binds=()
    while IFS= read -r line; do
        [[ -n "${line}" ]] && binds+=("${line}")
    done < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('\n'.join(d[0]['HostConfig'].get('Binds') or []))" "${inspect_json}")

    local -a envs=()
    while IFS= read -r line; do
        [[ -n "${line}" ]] && envs+=("${line}")
    done < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('\n'.join(d[0]['Config'].get('Env') or []))" "${inspect_json}")

    rm -f "${inspect_json}"

    local has_psibot=0
    local b
    for b in "${binds[@]}"; do
        if [[ "${b}" == "${PSIBOT_HOME}:${PSIBOT_HOME_CONTAINER}"* ]]; then
            has_psibot=1
            break
        fi
    done
    if [[ "${has_psibot}" -eq 0 ]]; then
        binds+=("${PSIBOT_HOME}:${PSIBOT_HOME_CONTAINER}:rw")
    fi

    local has_ime=0
    for b in "${binds[@]}"; do
        if [[ "${b}" == "${IME_RUNTIME_DIR}:${IME_RUNTIME_DIR}"* ]]; then
            has_ime=1
            break
        fi
    done
    # /tmp 绑定常被 daemon 拒绝；IME 改走 abstract unix socket，无需额外 bind

    # 去掉旧的 HOME/USER，统一用宿主机 psibot 身份
    local -a filtered_envs=()
    local e
    for e in "${envs[@]}"; do
        case "${e}" in
            HOME=*|USER=*|LOGNAME=*) continue ;;
        esac
        filtered_envs+=("${e}")
    done
    filtered_envs+=("HOME=${PSIBOT_HOME_CONTAINER}")
    filtered_envs+=("USER=psibot")
    filtered_envs+=("LOGNAME=psibot")

    echo ">>> 停止并删除旧容器 ${CONTAINER}（user=${CONTAINER_USER}/psibot，挂载 ${PSIBOT_HOME}:${PSIBOT_HOME_CONTAINER}）"
    docker stop "${CONTAINER}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER}" >/dev/null 2>&1 || true

    local -a run_args=(
        docker run -d
        --name "${CONTAINER}"
        --user "${CONTAINER_USER}"
        --network host
        --ipc host
        --security-opt label=disable
    )
    if [[ "${privileged}" == "1" ]]; then
        run_args+=(--privileged)
    fi
    for b in "${binds[@]}"; do
        run_args+=(-v "${b}")
    done
    for e in "${filtered_envs[@]}"; do
        run_args+=(-e "${e}")
    done
    if [[ -n "${workdir}" ]]; then
        run_args+=(-w "${workdir}")
    fi

    local -a entrypoint cmd
    mapfile -t entrypoint < <(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "${entrypoint_json}")
    mapfile -t cmd < <(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "${cmd_json}")

    if [[ ${#entrypoint[@]} -gt 0 && -n "${entrypoint[0]}" ]]; then
        run_args+=(--entrypoint "${entrypoint[0]}")
        if [[ ${#entrypoint[@]} -gt 1 ]]; then
            run_args+=("${entrypoint[@]:1}")
        fi
    fi
    run_args+=("${image}")
    if [[ ${#cmd[@]} -gt 0 ]]; then
        run_args+=("${cmd[@]}")
    fi

    echo ">>> docker run ${run_args[*]}"
    "${run_args[@]}"
    echo ">>> 容器已以 psibot(${CONTAINER_USER}) 重建并启动"
}

ensure_container_ready() {
    if ! container_exists; then
        echo "错误: 容器 ${CONTAINER} 不存在" >&2
        echo "请设置: A2D_DOCKER_CONTAINER=你的容器名 bash run_in_docker.sh" >&2
        exit 1
    fi

    local force_recreate="${A2D_DOCKER_RECREATE:-0}"
    local need_recreate=0
    if [[ "${force_recreate}" == "1" ]]; then
        need_recreate=1
        echo ">>> A2D_DOCKER_RECREATE=1，重建容器…"
    elif ! container_has_psibot_mount; then
        need_recreate=1
        echo ">>> 容器未挂载 ${PSIBOT_HOME}:${PSIBOT_HOME_CONTAINER}，正在重建…"
    elif ! container_runs_as_psibot; then
        need_recreate=1
        echo ">>> 容器未以 psibot(${CONTAINER_USER}) 运行（当前 User=$(docker inspect -f '{{.Config.User}}' "${CONTAINER}")），正在重建…"
    fi

    if [[ "${need_recreate}" -eq 1 ]]; then
        recreate_container_with_psibot_mount
    elif [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" != "true" ]]; then
        echo ">>> 启动容器 ${CONTAINER}…"
        docker start "${CONTAINER}" >/dev/null
    fi

    if ! container_has_psibot_mount; then
        echo "错误: 仍缺少挂载 ${PSIBOT_HOME}:${PSIBOT_HOME_CONTAINER}" >&2
        exit 1
    fi
    if ! container_runs_as_psibot; then
        echo "错误: 容器仍未以 psibot(${CONTAINER_USER}) 运行" >&2
        exit 1
    fi
}

ensure_container_ready

echo ">>> 容器: ${CONTAINER}"
echo ">>> 挂载: ${PSIBOT_HOME} → ${PSIBOT_HOME_CONTAINER}"
echo ">>> 工作目录: ${REMOTE_DIR}"
if [[ "${REMOTE_DIR}" != "${PSIBOT_HOME}/workspace_liyichao/eai" ]]; then
    echo ">>> 已解析软链: ${PSIBOT_HOME}/workspace_liyichao/eai → ${REMOTE_DIR}"
fi

if [[ ! -f "${SCRIPT_DIR}/show_camera_topics.py" ]]; then
    echo "错误: 未找到 ${SCRIPT_DIR}/show_camera_topics.py" >&2
    exit 1
fi

# 必须能看到完整工程（含 images/）；勿回退到只 cp 单个 py（会导致预览图缺失）
if ! docker exec -u "${CONTAINER_USER}" "${CONTAINER}" test -f "${REMOTE_DIR}/show_camera_topics.py"; then
    echo "错误: 容器内未看到 ${REMOTE_DIR}/show_camera_topics.py" >&2
    echo "请确认该路径已挂载进容器（当前依赖 share_data 挂载）" >&2
    exit 1
fi
if ! docker exec -u "${CONTAINER_USER}" "${CONTAINER}" test -d "${REMOTE_DIR}/images"; then
    echo "错误: 容器内未看到 ${REMOTE_DIR}/images（场景预览图目录）" >&2
    echo "宿主机有: ${EAI_DIR_HOST}/images" >&2
    exit 1
fi

if [[ -f "${FASTDDS_XML}" ]]; then
    docker exec -u "${CONTAINER_USER}" "${CONTAINER}" mkdir -p "${PSIBOT_HOME}/.cache/a2d_dds" 2>/dev/null || true
    if docker exec -u "${CONTAINER_USER}" "${CONTAINER}" test -f "${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"; then
        FASTDDS_IN_CONTAINER="${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"
    else
        docker exec -u "${CONTAINER_USER}" "${CONTAINER}" mkdir -p /tmp/a2d_dds
        docker cp "${FASTDDS_XML}" "${CONTAINER}:/tmp/a2d_dds/fastdds_profiles_a2d.xml"
        FASTDDS_IN_CONTAINER="/tmp/a2d_dds/fastdds_profiles_a2d.xml"
    fi
else
    FASTDDS_IN_CONTAINER="/tmp/a2d_dds/fastdds_profiles_a2d.xml"
fi

DISPLAY="${DISPLAY:-:0}"
echo ">>> DISPLAY=${DISPLAY}"

# Docker 内 GUI 需要宿主机 X11 授权。GDM 常用 /run/user/$UID/gdm/Xauthority，
# 而容器常挂载的 ~/.Xauthority 可能过期或被误建成目录。
if command -v xhost >/dev/null 2>&1; then
    if xhost +local: >/dev/null 2>&1; then
        echo ">>> 已执行: xhost +local: （允许容器访问本机显示）"
    else
        echo "警告: xhost +local: 失败，若出现 Qt/xcb 无法连接 DISPLAY，请在宿主机终端手动执行:" >&2
        echo "  xhost +local:" >&2
    fi
fi

XAUTH_HOST="${XAUTHORITY:-}"
if [[ -z "${XAUTH_HOST}" || ! -f "${XAUTH_HOST}" ]]; then
    for cand in "/run/user/$(id -u)/gdm/Xauthority" "${HOME}/.Xauthority"; do
        if [[ -f "${cand}" ]]; then
            XAUTH_HOST="${cand}"
            break
        fi
    done
fi
if [[ -d "${HOME}/.Xauthority" ]]; then
    echo "警告: ${HOME}/.Xauthority 是目录（应为文件），容器内 X cookie 挂载无效。" >&2
    echo "  可在宿主机用 sudo 修复: sudo rm -rf ${HOME}/.Xauthority" >&2
    echo "  然后: cp \"\${XAUTHORITY:-/run/user/\$(id -u)/gdm/Xauthority}\" ${HOME}/.Xauthority && chmod 600 ${HOME}/.Xauthority" >&2
fi

DOCKER_XAUTH_ENV=()
if [[ -n "${XAUTH_HOST}" && -f "${XAUTH_HOST}" ]]; then
    # 复制到容器可读写路径，避免仅 root 可读或路径未挂载
    XAUTH_COPY="${PSIBOT_HOME}/.cache/a2d_xauth"
    mkdir -p "${PSIBOT_HOME}/.cache"
    cp -f "${XAUTH_HOST}" "${XAUTH_COPY}" 2>/dev/null || true
    chmod 644 "${XAUTH_COPY}" 2>/dev/null || true
    if [[ -f "${XAUTH_COPY}" ]]; then
        DOCKER_XAUTH_ENV+=(-e "XAUTHORITY=${XAUTH_COPY}")
        echo ">>> XAUTHORITY=${XAUTH_HOST} → 容器内 ${XAUTH_COPY}"
    fi
fi

# 宿主机侧 Qwen 启停控制（Docker viewer 经 127.0.0.1:18101 调用；需 host 网络）
ensure_local_qwen_hostctl() {
    local ctl_py="${SCRIPT_DIR}/local_qwen_hostctl.py"
    local ctl_log="${SCRIPT_DIR}/log/local_qwen_hostctl.log"
    local ctl_port="${LOCAL_QWEN_CTL_PORT:-18101}"
    mkdir -p "${SCRIPT_DIR}/log"
    if curl -fsS --max-time 1 "http://127.0.0.1:${ctl_port}/health" >/dev/null 2>&1; then
        echo ">>> 本地 Qwen hostctl 已在运行 (:${ctl_port})"
        return 0
    fi
    if [[ ! -f "${ctl_py}" ]]; then
        echo "警告: 未找到 ${ctl_py}，测试 Tab 启动推理服务可能失败" >&2
        return 0
    fi
    echo ">>> 启动本地 Qwen hostctl (127.0.0.1:${ctl_port})"
    nohup python3 "${ctl_py}" >>"${ctl_log}" 2>&1 &
    disown || true
    sleep 0.4
    if curl -fsS --max-time 1 "http://127.0.0.1:${ctl_port}/health" >/dev/null 2>&1; then
        echo ">>> hostctl 就绪。测试 Tab 可点「启动推理服务」"
    else
        echo "警告: hostctl 未能就绪，可手动: python3 ${ctl_py}" >&2
    fi
}
ensure_local_qwen_hostctl

# 宿主机侧远程 Qwen 部署（Docker 内不要直接 ssh 隧道，避免 PID/端口冲突）
remote_hostctl_has_list_models() {
    local port="$1"
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
        "http://127.0.0.1:${port}/list_models?host=psi_motus_2_for_liyichao" 2>/dev/null || echo 000)"
    [[ "${code}" == "200" ]]
}

stop_remote_qwen_hostctl() {
    local ctl_port="$1"
    pkill -f "remote_qwen_hostctl.py" 2>/dev/null || true
    fuser -k "${ctl_port}/tcp" 2>/dev/null || true
    sleep 0.3
}

ensure_remote_qwen_hostctl() {
    local ctl_py="${SCRIPT_DIR}/remote_qwen_hostctl.py"
    local ctl_log="${SCRIPT_DIR}/log/remote_qwen_hostctl.log"
    local ctl_port="${REMOTE_QWEN_CTL_PORT:-18103}"
    mkdir -p "${SCRIPT_DIR}/log"
    if curl -fsS --max-time 1 "http://127.0.0.1:${ctl_port}/health" >/dev/null 2>&1; then
        if remote_hostctl_has_list_models "${ctl_port}"; then
            echo ">>> 远程 Qwen hostctl 已在运行 (:${ctl_port})"
            return 0
        fi
        echo ">>> 远程 Qwen hostctl 版本过旧（缺少 list_models），重启…"
        stop_remote_qwen_hostctl "${ctl_port}"
    fi
    if [[ ! -f "${ctl_py}" ]]; then
        echo "警告: 未找到 ${ctl_py}" >&2
        return 0
    fi
    echo ">>> 启动远程 Qwen hostctl (127.0.0.1:${ctl_port})"
    nohup python3 "${ctl_py}" >>"${ctl_log}" 2>&1 &
    disown || true
    sleep 0.4
    if curl -fsS --max-time 1 "http://127.0.0.1:${ctl_port}/health" >/dev/null 2>&1; then
        echo ">>> remote hostctl 就绪（测试 Tab 远程部署走宿主机 SSH）"
    else
        echo "警告: remote hostctl 未能就绪，可手动: python3 ${ctl_py}" >&2
    fi
}
ensure_remote_qwen_hostctl

# 清理旧版错误 SSH 隧道（进程在但 18100/18102 未监听），勿调用 stop（会停远程模型）
cleanup_stale_remote_qwen_tunnels() {
    if curl -fsS --max-time 1 "http://127.0.0.1:18100/health" >/dev/null 2>&1; then
        echo ">>> 远程 Qwen 隧道已可用 (:18100)"
        return 0
    fi
    if curl -fsS --max-time 1 "http://127.0.0.1:18102/health" >/dev/null 2>&1; then
        echo ">>> 远程 Qwen 隧道已可用 (:18102)"
        return 0
    fi
    echo ">>> 清理无效 SSH 隧道进程"
    pkill -f 'ssh .* -L 18100:127.0.0.1:8100' 2>/dev/null || true
    pkill -f 'ssh .* -L 18102:127.0.0.1:8100' 2>/dev/null || true
    fuser -k 18100/tcp 2>/dev/null || true
    fuser -k 18102/tcp 2>/dev/null || true
}
cleanup_stale_remote_qwen_tunnels

# 把宿主机 fcitx session dbus 转到 abstract socket，供容器（host network）访问
# 不能依赖 /tmp 或 /home/psibot bind：前者常被 daemon 拒绝，后者在本环境常不生效
dbus_addr_has_fcitx() {
    local addr="$1"
    [[ -n "${addr}" ]] || return 1
    DBUS_SESSION_BUS_ADDRESS="${addr}" \
        timeout 2 dbus-send --session --print-reply \
        --dest=org.freedesktop.DBus /org/freedesktop/DBus \
        org.freedesktop.DBus.ListNames 2>/dev/null \
        | grep -qi fcitx
}

parse_dbus_unix_path() {
    local addr="$1"
    if [[ "${addr}" =~ unix:path=([^,]+) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    fi
}

resolve_host_fcitx_bus() {
    local addr path pid sock
    for pid in $(pgrep -x fcitx 2>/dev/null; pgrep -x fcitx5 2>/dev/null); do
        addr="$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | sed -n 's/^DBUS_SESSION_BUS_ADDRESS=//p' | head -1)"
        path="$(parse_dbus_unix_path "${addr}")"
        if [[ -n "${path}" ]] && dbus_addr_has_fcitx "unix:path=${path}"; then
            printf '%s\n' "${path}"
            return 0
        fi
    done
    path="$(parse_dbus_unix_path "${DBUS_SESSION_BUS_ADDRESS:-}")"
    if [[ -n "${path}" ]] && dbus_addr_has_fcitx "unix:path=${path}"; then
        printf '%s\n' "${path}"
        return 0
    fi
    for path in \
        "${XDG_RUNTIME_DIR:-}/bus" \
        "/tmp/runtime-${USER:-psibot}/bus" \
        "/run/user/$(id -u)/bus"
    do
        if dbus_addr_has_fcitx "unix:path=${path}"; then
            printf '%s\n' "${path}"
            return 0
        fi
    done
    for sock in /tmp/dbus-*; do
        [[ -S "${sock}" ]] || continue
        if dbus_addr_has_fcitx "unix:path=${sock}"; then
            printf '%s\n' "${sock}"
            return 0
        fi
    done
    return 1
}

stop_dbus_proxy() {
    local pidfile="$1"
    local pid=""
    # 只用 pidfile，避免 pkill -f 匹配到本脚本命令行而自杀
    if [[ -f "${pidfile}" ]]; then
        pid="$(cat "${pidfile}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]]; then
            kill "${pid}" 2>/dev/null || true
            # socat fork 子进程
            pkill -P "${pid}" 2>/dev/null || true
        fi
        rm -f "${pidfile}"
    fi
    rm -f "${IME_RUNTIME_DIR}/bus"
    sleep 0.2
}

ensure_host_dbus_proxy_for_ime() {
    local runtime_proxy="${IME_RUNTIME_DIR}"
    local pidfile="${runtime_proxy}/dbus_proxy.pid"
    local logfile="${runtime_proxy}/dbus_proxy.log"
    local hostbus_file="${runtime_proxy}/host_bus.path"
    local abstract_addr="unix:abstract=${IME_DBUS_ABSTRACT}"
    local host_bus=""
    local prev_bus=""
    local need_restart=0
    local socat_cmd=""

    mkdir -p "${runtime_proxy}"
    chmod 700 "${runtime_proxy}" 2>/dev/null || true

    if ! host_bus="$(resolve_host_fcitx_bus)"; then
        echo "警告: 未找到带 fcitx 的宿主机 dbus，中文输入可能不可用" >&2
        echo "  请确认桌面已启动 fcitx（当前用户），且能在本机终端输入中文" >&2
        return 0
    fi
    echo ">>> 宿主机 fcitx dbus: ${host_bus}"

    if [[ -f "${hostbus_file}" ]]; then
        prev_bus="$(cat "${hostbus_file}" 2>/dev/null || true)"
    fi
    if [[ "${prev_bus}" != "${host_bus}" ]]; then
        need_restart=1
    fi
    if ! dbus_addr_has_fcitx "${abstract_addr}"; then
        need_restart=1
    fi

    if [[ "${need_restart}" -eq 0 ]]; then
        echo ">>> dbus abstract 代理已可用 (${abstract_addr})"
        return 0
    fi

    echo ">>> 重启 dbus 代理 (fcitx → ${abstract_addr})"
    stop_dbus_proxy "${pidfile}"

    if ! socat_cmd="$(resolve_socat)"; then
        echo "警告: 未找到 socat（试 ${HOME}/.local/bin/socat），无法建立 abstract dbus 代理" >&2
        return 0
    fi

    nohup "${socat_cmd}" "ABSTRACT-LISTEN:${IME_DBUS_ABSTRACT},fork,reuseaddr" \
        "UNIX-CONNECT:${host_bus}" </dev/null >>"${logfile}" 2>&1 &
    echo $! >"${pidfile}"
    printf '%s\n' "${host_bus}" >"${hostbus_file}"
    sleep 0.35

    if dbus_addr_has_fcitx "${abstract_addr}"; then
        echo ">>> dbus 代理就绪: ${abstract_addr}"
    else
        echo "错误: dbus 代理未连通 fcitx（${abstract_addr}）。" >&2
        echo "  常见原因: 代理在 Cursor 沙箱 netns 启动，宿主机/容器连不上。" >&2
        echo "  请在宿主机普通终端执行:" >&2
        echo "    bash ${SCRIPT_DIR}/fix_ime_dbus_proxy.sh" >&2
        if [[ -f "${logfile}" ]]; then
            tail -n 20 "${logfile}" >&2 || true
        fi
        exit 1
    fi
}
ensure_host_dbus_proxy_for_ime

# 确保 workspace 内有 fcitx Qt5 插件副本
ensure_fcitx_qt_plugin() {
    local dst_dir="${SCRIPT_DIR}/qt_plugins/platforminputcontexts"
    local dst="${dst_dir}/libfcitxplatforminputcontextplugin.so"
    local src="/usr/lib/x86_64-linux-gnu/qt5/plugins/platforminputcontexts/libfcitxplatforminputcontextplugin.so"
    mkdir -p "${dst_dir}"
    if [[ -f "${src}" && ( ! -f "${dst}" || "$(stat -c%s "${src}" 2>/dev/null)" != "$(stat -c%s "${dst}" 2>/dev/null)" ) ]]; then
        cp -f "${src}" "${dst}"
        echo ">>> 已复制 fcitx Qt 插件到 ${dst}"
    fi
}
ensure_fcitx_qt_plugin

DBUS_PROXY_ADDR="unix:abstract=${IME_DBUS_ABSTRACT}"

# 必须用宿主机用户跑 GUI：root 无法通过 dbus EXTERNAL 认证连接 fcitx
docker exec \
    -u "${HOST_UID}:${HOST_GID}" \
    -e HOME="${PSIBOT_HOME_CONTAINER}" \
    -e USER="${USER:-psibot}" \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -e "XDG_RUNTIME_DIR=/tmp/a2d_runtime" \
    -e "DBUS_SESSION_BUS_ADDRESS=${DBUS_PROXY_ADDR}" \
    -e "LANG=zh_CN.UTF-8" \
    -e "LC_ALL=zh_CN.UTF-8" \
    -e "QT_IM_MODULE=${QT_IM_MODULE:-fcitx}" \
    -e "GTK_IM_MODULE=${GTK_IM_MODULE:-fcitx}" \
    -e "XMODIFIERS=${XMODIFIERS:-@im=fcitx}" \
    "${DOCKER_XAUTH_ENV[@]}" \
    -e FASTRTPS_DEFAULT_PROFILES_FILE="${FASTDDS_IN_CONTAINER}" \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_DOMAIN_ID=0 \
    -e ROS_LOCALHOST_ONLY=1 \
    -e A2D_SCRIPTS_DIR="${A2D_SCRIPTS_DIR}" \
    -e PSIBOT_HOME="${PSIBOT_HOME_CONTAINER}" \
    -e LOCAL_QWEN_CTL_URL="http://127.0.0.1:${LOCAL_QWEN_CTL_PORT:-18101}" \
    -e LOCAL_QWEN_API_BASE="http://127.0.0.1:${LOCAL_QWEN_PORT:-8100}/v1" \
    -e REMOTE_QWEN_CTL_URL="http://127.0.0.1:${REMOTE_QWEN_CTL_PORT:-18103}" \
    -it "${CONTAINER}" \
    bash -lc "
set -eo pipefail
mkdir -p /tmp/a2d_runtime ${PSIBOT_HOME_CONTAINER}/.cache/a2d_runtime
chmod 700 /tmp/a2d_runtime ${PSIBOT_HOME_CONTAINER}/.cache/a2d_runtime 2>/dev/null || true
source /opt/ros/humble/setup.bash
if [[ -f /opt/psi/rt/a2d-tele/install/setup.bash ]]; then
    source /opt/psi/rt/a2d-tele/install/setup.bash
fi
if [[ -f ${WORKSPACE_INSTALL} ]]; then
    source ${WORKSPACE_INSTALL}
fi
export A2D_SCRIPTS_DIR=\"${A2D_SCRIPTS_DIR}\"
export PSIBOT_HOME=\"${PSIBOT_HOME_CONTAINER}\"
export HOME=\"${PSIBOT_HOME_CONTAINER}\"
if locale -a 2>/dev/null | grep -qi zh_CN; then
    export LANG=\"zh_CN.UTF-8\"
    export LC_ALL=\"zh_CN.UTF-8\"
else
    echo \"警告: 容器无 zh_CN.UTF-8，回退 C.UTF-8\" >&2
    export LANG=\"C.UTF-8\"
    export LC_ALL=\"C.UTF-8\"
fi
export QT_X11_NO_MITSHM=1
export QT_IM_MODULE=\"fcitx\"
export XMODIFIERS=\"@im=fcitx\"
export GTK_IM_MODULE=\"fcitx\"
export DBUS_SESSION_BUS_ADDRESS=\"${DBUS_PROXY_ADDR}\"
export XDG_RUNTIME_DIR=\"/tmp/a2d_runtime\"

unset QT_PLUGIN_PATH

python3 -m pip install -q --user \
    'PyQt5==5.15.10' \
    'opencv-python-headless==4.10.0.84' \
    'numpy>=1.23.5,<2.0.0' \
    'pyqtgraph==0.13.7' \
    'PyOpenGL==3.1.7' 2>/dev/null || true
python3 -m pip uninstall -y opencv-python 2>/dev/null || true

python3 - <<'PY'
import pathlib, shutil
src = pathlib.Path('${REMOTE_DIR}/qt_plugins/platforminputcontexts/libfcitxplatforminputcontextplugin.so')
if not src.is_file():
    raise SystemExit(0)
import PyQt5
dst_dir = pathlib.Path(PyQt5.__file__).resolve().parent / 'Qt5' / 'plugins' / 'platforminputcontexts'
dst_dir.mkdir(parents=True, exist_ok=True)
dst = dst_dir / src.name
if (not dst.exists()) or dst.stat().st_size != src.stat().st_size:
    shutil.copy2(src, dst)
    print(f'>>> 已安装 fcitx Qt 插件到 {dst}')
PY

# pip PyQt5 自带 Qt；fcitx 插件链到系统 Qt 会 Cannot mix incompatible Qt。
# 必须把 PyQt5/Qt5/lib 放在 LD_LIBRARY_PATH 最前。
PYQT_QT_LIB=\"\$(python3 -c 'import PyQt5, pathlib; print(pathlib.Path(PyQt5.__file__).resolve().parent / \"Qt5\" / \"lib\")' 2>/dev/null || true)\"
if [[ -n \"\${PYQT_QT_LIB}\" && -d \"\${PYQT_QT_LIB}\" ]]; then
    export LD_LIBRARY_PATH=\"\${PYQT_QT_LIB}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"
    echo \">>> PyQt5 Qt lib: \${PYQT_QT_LIB}\"
fi

echo \">>> IME QT_IM_MODULE=\$QT_IM_MODULE DBUS=\$DBUS_SESSION_BUS_ADDRESS\"
echo '>>> 预检 topic...'
python3 - <<'PY'
import rclpy
rclpy.init()
from rclpy.node import Node
n = Node('preflight')
topics = [t for t in dict(n.get_topic_names_and_types()) if t.startswith('/camera')]
print(f'发现 {len(topics)} 个 /camera topic')
n.destroy_node()
rclpy.shutdown()
PY

exec python3 ${REMOTE_DIR}/show_camera_topics.py \"\$@\"
" -- "$@"
