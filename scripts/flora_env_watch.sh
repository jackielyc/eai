#!/usr/bin/env bash
# Watch Flora env install; status files stay under eai/log/.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DONE="$ROOT/log/flora_env.DONE"
LOG="$ROOT/log/flora_env_install.log"
NOTIFY="$ROOT/log/flora_env_notify.txt"
PIDF="$ROOT/log/flora_env_install.pid"
mkdir -p "$ROOT/log"
alive() {
  local p
  p=$(cat "$PIDF" 2>/dev/null || true)
  [[ -n "$p" && -d "/proc/$p" ]] && return 0
  ps -eo cmd | grep -q '[e]nvs/flora/bin/python -m pip' && return 0
  return 1
}
for i in $(seq 1 240); do
  if [[ -f "$DONE" ]]; then
    echo "FLORA env DONE at $(date)" | tee "$NOTIFY"
    cat "$DONE" >>"$NOTIFY"
    exit 0
  fi
  if ! alive; then
    sleep 5
    if [[ -f "$DONE" ]]; then
      echo "FLORA env DONE at $(date)" | tee "$NOTIFY"
      cat "$DONE" >>"$NOTIFY"
      exit 0
    fi
    if ! alive; then
      echo "installer exited without DONE at $(date)" | tee "$NOTIFY"
      tail -40 "$LOG" >>"$NOTIFY"
      exit 1
    fi
  fi
  sleep 15
done
echo "watch timeout at $(date)" | tee "$NOTIFY"
exit 2
