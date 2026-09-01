#!/usr/bin/env bash
# 必须在宿主机普通终端运行（不要在 Cursor Agent 里跑）
set -eo pipefail
IME_RUNTIME_DIR="${IME_RUNTIME_DIR:-/tmp/a2d_ime}"
IME_DBUS_ABSTRACT="${IME_DBUS_ABSTRACT:-a2d_ime_dbus}"
SOCAT_BIN="${SOCAT_BIN:-$HOME/.local/bin/socat}"
PIDFILE="${IME_RUNTIME_DIR}/dbus_proxy.pid"
LOGFILE="${IME_RUNTIME_DIR}/dbus_proxy.log"
HOSTBUS_FILE="${IME_RUNTIME_DIR}/host_bus.path"
mkdir -p "${IME_RUNTIME_DIR}"
chmod 700 "${IME_RUNTIME_DIR}" 2>/dev/null || true
[[ -x "${SOCAT_BIN}" ]] || { echo "错误: 无 ${SOCAT_BIN}" >&2; exit 1; }

parse_dbus_unix_path() {
  local addr="$1"
  if [[ "${addr}" =~ unix:path=([^,]+) ]]; then printf '%s\n' "${BASH_REMATCH[1]}"; fi
}
dbus_has_fcitx() {
  local out
  out="$(dbus-send --bus="$1" --print-reply --dest=org.freedesktop.DBus /org/freedesktop/DBus org.freedesktop.DBus.ListNames 2>/dev/null || true)"
  grep -qi fcitx <<<"${out}"
}
resolve_host_fcitx_bus() {
  local addr path pid sock
  for pid in $(pgrep -x fcitx 2>/dev/null; pgrep -x fcitx5 2>/dev/null); do
    addr="$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | sed -n 's/^DBUS_SESSION_BUS_ADDRESS=//p' | head -1 || true)"
    [[ -z "${addr}" ]] && addr="$(sudo tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | sed -n 's/^DBUS_SESSION_BUS_ADDRESS=//p' | head -1 || true)"
    path="$(parse_dbus_unix_path "${addr}")"
    if [[ -n "${path}" && -S "${path}" ]] && dbus_has_fcitx "unix:path=${path}"; then
      printf '%s\n' "${path}"; return 0
    fi
  done
  for sock in /tmp/dbus-*; do
    [[ -S "${sock}" ]] || continue
    if dbus_has_fcitx "unix:path=${sock}"; then printf '%s\n' "${sock}"; return 0; fi
  done
  return 1
}
stop_proxy() {
  local pid=""
  if [[ -f "${PIDFILE}" ]]; then
    pid="$(cat "${PIDFILE}" 2>/dev/null || true)"
    [[ -n "${pid}" ]] && { kill "${pid}" 2>/dev/null || true; pkill -P "${pid}" 2>/dev/null || true; }
    rm -f "${PIDFILE}"
  fi
  for pid in $(pgrep -u "${USER}" -f "${SOCAT_BIN} ABSTRACT-LISTEN:${IME_DBUS_ABSTRACT}" 2>/dev/null || true); do
    kill "${pid}" 2>/dev/null || true
  done
  sleep 0.2
}

ABSTRACT_ADDR="unix:abstract=${IME_DBUS_ABSTRACT}"
echo ">>> 解析宿主机 fcitx dbus…"
HOST_BUS="$(resolve_host_fcitx_bus)" || { echo "错误: 未找到 fcitx dbus" >&2; exit 1; }
echo ">>> 宿主机 fcitx dbus: ${HOST_BUS}"
printf '%s' "${HOST_BUS}" > "${HOSTBUS_FILE}"

if dbus_has_fcitx "${ABSTRACT_ADDR}"; then
  echo ">>> abstract 代理已可用: ${ABSTRACT_ADDR}"
else
  echo ">>> 重启 abstract 代理…"
  stop_proxy
  nohup "${SOCAT_BIN}" "ABSTRACT-LISTEN:${IME_DBUS_ABSTRACT},fork,reuseaddr" \
    "UNIX-CONNECT:${HOST_BUS}" >"${LOGFILE}" 2>&1 &
  echo $! > "${PIDFILE}"
  disown || true
  sleep 0.5
fi

if ! dbus_has_fcitx "${ABSTRACT_ADDR}"; then
  echo "错误: ${ABSTRACT_ADDR} 仍无 fcitx。请在宿主机普通终端执行本脚本（不要在 Cursor Agent 里跑）。" >&2
  cat "${LOGFILE}" 2>/dev/null || true
  exit 1
fi
echo ">>> dbus 代理就绪: ${ABSTRACT_ADDR} → ${HOST_BUS}"
echo ">>> 然后重启 viewer:"
echo "    docker exec a2d-tele-release-2-1-0rc3-latest pkill -f show_camera_topics.py || true"
echo "    ~/workspace_liyichao/run_in_docker.sh"
