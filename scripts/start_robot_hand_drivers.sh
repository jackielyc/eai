#!/usr/bin/env bash
# 已废弃 Docker 手驱动。永远用宿主机：使能桥 + Ruiyan serial hand_node。
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="${SCRIPT_DIR}/start_host_arm_hand.sh"
RUIYAN="${SCRIPT_DIR}/start_host_ruiyan.sh"
echo ">>> 不用 Docker：宿主机手/臂桥 + Ruiyan" >&2
case "${1:-}" in
  --status|status)
    bash "${BRIDGE}" status
    bash "${RUIYAN}" status
    ;;
  --stop|stop)
    bash "${RUIYAN}" stop || true
    bash "${BRIDGE}" stop || true
    ;;
  --force)
    bash "${RUIYAN}" stop || true
    bash "${BRIDGE}" stop || true
    bash "${BRIDGE}" start
    bash "${RUIYAN}" start
    ;;
  *)
    bash "${BRIDGE}" start
    bash "${RUIYAN}" start
    ;;
esac
