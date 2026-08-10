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
#   A2D_DOCKER_RECREATE=1    强制重建容器以应用挂载（慎用）
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${A2D_DOCKER_CONTAINER:-a2d-tele-release-2-1-0rc3-latest}"
PSIBOT_HOME="${PSIBOT_HOME:-/home/psibot}"
PSIBOT_HOME_CONTAINER="${PSIBOT_HOME_CONTAINER:-${PSIBOT_HOME}}"
REMOTE_DIR="${PSIBOT_HOME}/workspace_liyichao/eai"
A2D_SCRIPTS_DIR="${A2D_SCRIPTS_DIR:-${PSIBOT_HOME}/workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts}"
WORKSPACE_INSTALL="${PSIBOT_HOME}/workspace_liyichao/install/setup.bash"
A2D_SDK_HOME="${A2D_SDK_HOME:-${PSIBOT_HOME}/a2d_sdk}"
FASTDDS_XML="${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"

container_exists() {
    docker inspect "${CONTAINER}" >/dev/null 2>&1
}

container_has_psibot_mount() {
    docker inspect "${CONTAINER}" --format '{{range .Mounts}}{{if eq .Destination "'"${PSIBOT_HOME_CONTAINER}"'"}}{{.Source}}{{end}}{{end}}' \
        | grep -q .
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

    local inspect_json image workdir entrypoint_json cmd_json
    inspect_json="$(save_container_config)"

    image="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[0]['Config']['Image'])" "${inspect_json}")"
    workdir="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[0]['Config'].get('WorkingDir') or '')" "${inspect_json}")"
    entrypoint_json="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d[0]['Config'].get('Entrypoint') or []))" "${inspect_json}")"
    cmd_json="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d[0]['Config'].get('Cmd') or []))" "${inspect_json}")"

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

    echo ">>> 停止并删除旧容器 ${CONTAINER}（添加挂载 ${PSIBOT_HOME}:${PSIBOT_HOME_CONTAINER}）"
    docker stop "${CONTAINER}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER}" >/dev/null 2>&1 || true

    local -a run_args=(
        docker run -d
        --name "${CONTAINER}"
        --network host
        --privileged
        --ipc host
        --security-opt label=disable
    )
    for b in "${binds[@]}"; do
        run_args+=(-v "${b}")
    done
    for e in "${envs[@]}"; do
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
    echo ">>> 容器已重建并启动"
}

ensure_container_ready() {
    if ! container_exists; then
        echo "错误: 容器 ${CONTAINER} 不存在" >&2
        echo "请设置: A2D_DOCKER_CONTAINER=你的容器名 bash run_in_docker.sh" >&2
        exit 1
    fi

    local force_recreate="${A2D_DOCKER_RECREATE:-0}"
    if [[ "${force_recreate}" == "1" ]] || ! container_has_psibot_mount; then
        if ! container_has_psibot_mount; then
            echo ">>> 容器未挂载 ${PSIBOT_HOME}:${PSIBOT_HOME_CONTAINER}，正在重建…"
        else
            echo ">>> A2D_DOCKER_RECREATE=1，重建容器…"
        fi
        recreate_container_with_psibot_mount
    elif [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" != "true" ]]; then
        echo ">>> 启动容器 ${CONTAINER}…"
        docker start "${CONTAINER}" >/dev/null
    fi

    if ! container_has_psibot_mount; then
        echo "错误: 仍缺少挂载 ${PSIBOT_HOME}:${PSIBOT_HOME_CONTAINER}" >&2
        exit 1
    fi
}

ensure_container_ready

echo ">>> 容器: ${CONTAINER}"
echo ">>> 挂载: ${PSIBOT_HOME} → ${PSIBOT_HOME_CONTAINER}"
echo ">>> 工作目录: ${REMOTE_DIR}"

if [[ ! -f "${SCRIPT_DIR}/show_camera_topics.py" ]]; then
    echo "错误: 未找到 ${SCRIPT_DIR}/show_camera_topics.py" >&2
    exit 1
fi

# 挂载后容器内路径与宿主机一致，无需 docker cp
if ! docker exec "${CONTAINER}" test -f "${REMOTE_DIR}/show_camera_topics.py"; then
    echo ">>> 容器内未看到 ${REMOTE_DIR}/show_camera_topics.py，回退 docker cp …"
    docker exec "${CONTAINER}" mkdir -p "${REMOTE_DIR}"
    docker cp "${SCRIPT_DIR}/show_camera_topics.py" "${CONTAINER}:${REMOTE_DIR}/"
fi

if [[ -f "${FASTDDS_XML}" ]]; then
    docker exec "${CONTAINER}" mkdir -p "${PSIBOT_HOME}/.cache/a2d_dds" 2>/dev/null || true
    if docker exec "${CONTAINER}" test -f "${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"; then
        FASTDDS_IN_CONTAINER="${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"
    else
        docker exec "${CONTAINER}" mkdir -p /tmp/a2d_dds
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

docker exec \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -e XDG_RUNTIME_DIR=/tmp/runtime-psibot \
    "${DOCKER_XAUTH_ENV[@]}" \
    -e FASTRTPS_DEFAULT_PROFILES_FILE="${FASTDDS_IN_CONTAINER}" \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_DOMAIN_ID=0 \
    -e ROS_LOCALHOST_ONLY=1 \
    -e A2D_SCRIPTS_DIR="${A2D_SCRIPTS_DIR}" \
    -e PSIBOT_HOME="${PSIBOT_HOME_CONTAINER}" \
    -it "${CONTAINER}" \
    bash -lc "
set -eo pipefail
mkdir -p /tmp/runtime-psibot
source /opt/ros/humble/setup.bash
if [[ -f /opt/psi/rt/a2d-tele/install/setup.bash ]]; then
    source /opt/psi/rt/a2d-tele/install/setup.bash
fi
if [[ -f ${WORKSPACE_INSTALL} ]]; then
    source ${WORKSPACE_INSTALL}
fi
export A2D_SCRIPTS_DIR=\"${A2D_SCRIPTS_DIR}\"
export PSIBOT_HOME=\"${PSIBOT_HOME_CONTAINER}\"

unset QT_PLUGIN_PATH

python3 -m pip install -q --user \
    'PyQt5==5.15.10' \
    'opencv-python-headless==4.10.0.84' \
    'numpy>=1.23.5,<2.0.0' \
    'pyqtgraph==0.13.7' \
    'PyOpenGL==3.1.7' 2>/dev/null || true
python3 -m pip uninstall -y opencv-python 2>/dev/null || true

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
