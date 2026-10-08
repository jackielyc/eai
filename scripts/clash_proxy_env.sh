#!/usr/bin/env bash
# Clash Verge (Sidecar) 代理环境：本机无 systemd，TUN/Service 不可用，
# 终端与 Python 需显式走 mixed-port（默认 127.0.0.1:7897）。
#
# 用法:
#   source scripts/clash_proxy_env.sh          # 端口在听则 export
#   source scripts/clash_proxy_env.sh --force  # 不探测，直接 export
#   source scripts/clash_proxy_env.sh --off    # 取消代理变量
#   bash scripts/clash_proxy_env.sh --print    # 仅打印 export 行

CLASH_PROXY_HOST="${CLASH_PROXY_HOST:-127.0.0.1}"
CLASH_MIXED_PORT="${CLASH_MIXED_PORT:-7897}"
_CLASH_MODE="apply"
for _clash_arg in "$@"; do
  case "${_clash_arg}" in
    --force) _CLASH_MODE="force" ;;
    --off|unset) _CLASH_MODE="off" ;;
    --print) _CLASH_MODE="print" ;;
    -h|--help)
      sed -n '2,11p' "${BASH_SOURCE[0]:-${(%):-%N}}" 2>/dev/null || sed -n '2,11p' "$0"
      unset _CLASH_MODE _clash_arg
      return 0 2>/dev/null || exit 0
      ;;
  esac
done

_clash_port_listening() {
  local host="$1" port="$2"
  if command -v ss >/dev/null 2>&1; then
    ss -lnt "sport = :${port}" 2>/dev/null | grep -q LISTEN && return 0
  fi
  if command -v nc >/dev/null 2>&1; then
    nc -z "${host}" "${port}" >/dev/null 2>&1 && return 0
  fi
  if [[ -n "${BASH_VERSION:-}" ]]; then
    (echo >/dev/tcp/"${host}"/"${port}") >/dev/null 2>&1 && return 0
  fi
  return 1
}

_clash_proxy_url="http://${CLASH_PROXY_HOST}:${CLASH_MIXED_PORT}"
_clash_no_proxy="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"

_clash_do_export() {
  export http_proxy="${_clash_proxy_url}"
  export https_proxy="${_clash_proxy_url}"
  export HTTP_PROXY="${_clash_proxy_url}"
  export HTTPS_PROXY="${_clash_proxy_url}"
  export all_proxy="${_clash_proxy_url}"
  export ALL_PROXY="${_clash_proxy_url}"
  export no_proxy="${_clash_no_proxy}"
  export NO_PROXY="${_clash_no_proxy}"
}

_clash_do_unset() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
}

_clash_print_exports() {
  printf 'export http_proxy=%s\n' "${_clash_proxy_url}"
  printf 'export https_proxy=%s\n' "${_clash_proxy_url}"
  printf 'export HTTP_PROXY=%s\n' "${_clash_proxy_url}"
  printf 'export HTTPS_PROXY=%s\n' "${_clash_proxy_url}"
  printf 'export all_proxy=%s\n' "${_clash_proxy_url}"
  printf 'export ALL_PROXY=%s\n' "${_clash_proxy_url}"
  printf 'export no_proxy=%s\n' "${_clash_no_proxy}"
  printf 'export NO_PROXY=%s\n' "${_clash_no_proxy}"
}

case "${_CLASH_MODE}" in
  off)
    _clash_do_unset
    echo "[clash-proxy] cleared proxy env"
    ;;
  print)
    if ! _clash_port_listening "${CLASH_PROXY_HOST}" "${CLASH_MIXED_PORT}"; then
      echo "[clash-proxy] ${CLASH_PROXY_HOST}:${CLASH_MIXED_PORT} not listening" >&2
      unset _CLASH_MODE _clash_arg _clash_proxy_url _clash_no_proxy
      unset -f _clash_port_listening _clash_do_export _clash_do_unset _clash_print_exports 2>/dev/null
      return 1 2>/dev/null || exit 1
    fi
    _clash_print_exports
    ;;
  force)
    _clash_do_export
    echo "[clash-proxy] forced ${_clash_proxy_url}"
    ;;
  *)
    _existing="${HTTPS_PROXY:-${https_proxy:-}}"
    if [[ -n "${_existing}" ]]; then
      echo "[clash-proxy] keep existing proxy=${_existing}"
    elif ! _clash_port_listening "${CLASH_PROXY_HOST}" "${CLASH_MIXED_PORT}"; then
      echo "[clash-proxy] skip: ${CLASH_PROXY_HOST}:${CLASH_MIXED_PORT} not listening (start Clash Verge Sidecar)"
    else
      _clash_do_export
      echo "[clash-proxy] using ${_clash_proxy_url}"
    fi
    unset _existing
    ;;
esac

unset _CLASH_MODE _clash_arg _clash_proxy_url _clash_no_proxy
unset -f _clash_port_listening _clash_do_export _clash_do_unset _clash_print_exports 2>/dev/null
return 0 2>/dev/null || exit 0
