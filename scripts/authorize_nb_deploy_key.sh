#!/usr/bin/env bash
# Run ON the target notebook (TI-ONE web terminal as root) to allow deploy SSH.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PUB="$HERE/psibot_deploy.pub"
[ -f "$PUB" ] || PUB=/share_data/projects/mahjong/share/personal/liyichao/eai/scripts/psibot_deploy.pub
[ -f "$PUB" ] || { echo "missing pubkey file"; exit 1; }
KEY=$(cat "$PUB")
authorize() {
  local home="$1"
  [ -d "$home" ] || return 0
  mkdir -p "$home/.ssh"
  chmod 700 "$home/.ssh"
  touch "$home/.ssh/authorized_keys"
  chmod 600 "$home/.ssh/authorized_keys"
  if grep -qxF "$KEY" "$home/.ssh/authorized_keys" 2>/dev/null; then
    echo "already authorized in $home/.ssh/authorized_keys"
  else
    echo "$KEY" >> "$home/.ssh/authorized_keys"
    echo "authorized in $home/.ssh/authorized_keys"
  fi
}
authorize /root
authorize /home/psibot || true
authorize "${HOME:-}" || true
echo "done. From viewer host test:"
echo "  ssh nb-1668047664206989312-cp0nfah4b37k 'hostname; nvidia-smi -L'"
