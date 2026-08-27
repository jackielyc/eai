#!/usr/bin/env bash
# Shared torchrun env for lake_qwen35 launchers. Source from run_train_*.sh.
# Single machine (default): torchrun --standalone, same as before.
# Multi-node: HOSTS / HOSTFILE on one machine SSHs to each host; or set
# NNODES + NODE_RANK + MASTER_ADDR on every machine yourself.

_dist_trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

_dist_bare_host() {
  local h="$1"
  h="${h##*@}"
  printf '%s' "${h}"
}

_dist_ssh_target() {
  local host="$1"
  if [[ "${host}" == *@* ]]; then
    printf '%s' "${host}"
  elif [[ -n "${SSH_USER:-}" ]]; then
    printf '%s@%s' "${SSH_USER}" "${host}"
  else
    printf '%s' "${host}"
  fi
}

_dist_local_addrs() {
  {
    hostname -I 2>/dev/null | tr ' ' '\n'
    hostname -f 2>/dev/null
    hostname -s 2>/dev/null
    hostname 2>/dev/null
    if command -v ip >/dev/null 2>&1; then
      ip -4 -o addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1
    fi
  } | awk 'NF && $0 != "127.0.0.1"'
}

_dist_host_is_local() {
  local host="$(_dist_bare_host "$1")"
  local addr
  if [[ "${host}" == "localhost" || "${host}" == "127.0.0.1" ]]; then
    return 0
  fi
  while read -r addr; do
    [[ -z "${addr}" ]] && continue
    if [[ "${addr}" == "${host}" ]]; then
      return 0
    fi
  done < <(_dist_local_addrs)
  return 1
}

_dist_load_hosts() {
  DIST_HOSTS=()
  local line host
  if [[ -n "${HOSTFILE:-}" ]]; then
    if [[ ! -f "${HOSTFILE}" ]]; then
      echo "[error] HOSTFILE not found: ${HOSTFILE}" >&2
      exit 1
    fi
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="${line%%#*}"
      line="$(_dist_trim "${line}")"
      [[ -z "${line}" ]] && continue
      host="${line%%[[:space:]]*}"
      host="$(_dist_trim "${host}")"
      [[ -n "${host}" ]] && DIST_HOSTS+=("${host}")
    done < "${HOSTFILE}"
  elif [[ -n "${HOSTS:-}" ]]; then
    local IFS=','
    read -r -a DIST_HOSTS <<< "${HOSTS}"
    local i
    for i in "${!DIST_HOSTS[@]}"; do
      DIST_HOSTS[$i]="$(_dist_trim "${DIST_HOSTS[$i]}")"
    done
  fi
  local cleaned=()
  for host in "${DIST_HOSTS[@]+"${DIST_HOSTS[@]}"}"; do
    [[ -n "${host}" ]] && cleaned+=("${host}")
  done
  DIST_HOSTS=("${cleaned[@]+"${cleaned[@]}"}")
}

_dist_match_rank() {
  local addr i bare
  while read -r addr; do
    [[ -z "${addr}" ]] && continue
    for i in "${!DIST_HOSTS[@]}"; do
      bare="$(_dist_bare_host "${DIST_HOSTS[$i]}")"
      if [[ "${bare}" == "${addr}" ]]; then
        printf '%s' "${i}"
        return 0
      fi
    done
  done < <(_dist_local_addrs)
  return 1
}

_dist_from_slurm() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi
  if [[ "${_nnodes_provided}" -eq 0 && -n "${SLURM_NNODES:-}" ]]; then
    NNODES="${SLURM_NNODES}"
  fi
  if [[ "${_node_rank_provided}" -eq 0 && -n "${SLURM_NODEID:-}" ]]; then
    NODE_RANK="${SLURM_NODEID}"
  fi
  if [[ "${_master_addr_provided}" -eq 0 && -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"
  fi
}

_dist_is_coordinator() {
  [[ -z "${DIST_REMOTE:-}" && "${#DIST_HOSTS[@]}" -gt 1 ]]
}

_dist_ssh() {
  # shellcheck disable=SC2206
  local opts=( ${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o LogLevel=ERROR} )
  ssh -n "${opts[@]}" "$@"
}

_dist_worker_script() {
  local rank="$1"
  local entry="$2"
  local name
  printf 'set -euo pipefail\n'
  printf 'trap '\''kill 0'\'' HUP INT TERM\n'
  printf 'cd %q\n' "${ROOT}"
  printf 'export DIST_REMOTE=1\n'
  printf 'export PYTHONUNBUFFERED=1\n'
  printf 'unset HOSTS HOSTFILE\n'
  printf 'export NODE_RANK=%q\n' "${rank}"
  printf 'export NNODES=%q\n' "${NNODES}"
  printf 'export MASTER_ADDR=%q\n' "${MASTER_ADDR}"
  printf 'export MASTER_PORT=%q\n' "${MASTER_PORT}"
  for name in \
    PYTHON CONFIG CUDA_VISIBLE_DEVICES NPROC DEVICE_MAP DDP_TIMEOUT DATASET_WAIT_SEC \
    TOKENIZERS_PARALLELISM PYTHONUNBUFFERED NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_ASYNC_ERROR_HANDLING \
    NCCL_SOCKET_IFNAME NCCL_IB_DISABLE \
    NCCL_DEBUG NCCL_P2P_DISABLE NCCL_IB_HCA HF_HOME TRANSFORMERS_CACHE \
    HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE PYTHONPATH CUDA_HOME LD_LIBRARY_PATH \
    http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY; do
    if [[ -n "${!name+x}" ]]; then
      printf 'export %s=%q\n' "${name}" "${!name}"
    fi
  done
  printf 'bash %q\n' "${entry}"
}

_dist_child_pids=()
_dist_on_signal() {
  echo "[info] stopping workers..." >&2
  local pid
  for pid in "${_dist_child_pids[@]+"${_dist_child_pids[@]}"}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  wait || true
  exit 130
}

# Launch the same entry script on every host in HOSTS/HOSTFILE, then wait.
# No-op on a single machine, or when already running as a remote worker.
dist_spawn_workers() {
  local entry="$1"
  local host rank target pid status rc local_log
  if [[ -n "${DIST_REMOTE:-}" ]]; then
    return 0
  fi
  if [[ "${#DIST_HOSTS[@]}" -le 1 ]]; then
    return 0
  fi
  if [[ "${entry}" != /* ]]; then
    entry="$(cd "$(dirname "${entry}")" && pwd)/$(basename "${entry}")"
  fi
  if [[ ! -f "${entry}" ]]; then
    echo "[error] launch script not found: ${entry}" >&2
    exit 1
  fi

  echo "[info] coordinator: ssh/local launch on ${#DIST_HOSTS[@]} hosts"
  echo "[info] hosts=${DIST_HOSTS[*]}"
  echo "[info] master=${MASTER_ADDR}:${MASTER_PORT} nproc=${NPROC}"

  for rank in "${!DIST_HOSTS[@]}"; do
    host="${DIST_HOSTS[$rank]}"
    if _dist_host_is_local "${host}"; then
      continue
    fi
    target="$(_dist_ssh_target "${host}")"
    echo "[info] checking ssh ${target} ..."
    if ! _dist_ssh "${target}" "test -f $(printf '%q' "${entry}") && test -d $(printf '%q' "${ROOT}")"; then
      echo "[error] cannot ssh to ${target}, or ${entry} / ${ROOT} missing on that host." >&2
      echo "[error] need passwordless ssh and a shared path (e.g. NFS)." >&2
      exit 1
    fi
  done

  trap '_dist_on_signal' INT TERM
  mkdir -p "${ROOT}/output"
  for rank in "${!DIST_HOSTS[@]}"; do
    host="${DIST_HOSTS[$rank]}"
    if [[ "${rank}" -eq 0 ]]; then
      # Rank 0 trains on the launch terminal so [train] lines show up immediately.
      if _dist_host_is_local "${host}"; then
        echo "[info] local rank=0 host=${host} (progress on this terminal)"
        (
          set -euo pipefail
          export DIST_REMOTE=1 PYTHONUNBUFFERED=1
          export NODE_RANK="${rank}"
          export NNODES MASTER_ADDR MASTER_PORT
          unset HOSTS HOSTFILE
          bash "${entry}"
        ) &
      else
        target="$(_dist_ssh_target "${host}")"
        echo "[info] ssh rank=0 host=${target} (progress on this terminal)"
        (
          set -euo pipefail
          _dist_ssh "${target}" "bash -c $(printf '%q' "$(_dist_worker_script "${rank}" "${entry}")")"
        ) &
      fi
    else
      local_log="${ROOT}/output/train_n${rank}_$(date +%Y%m%d_%H%M%S).log"
      echo "[info] rank=${rank} host=${host} log=${local_log}"
      if _dist_host_is_local "${host}"; then
        (
          set -euo pipefail
          export DIST_REMOTE=1 PYTHONUNBUFFERED=1
          export NODE_RANK="${rank}"
          export NNODES MASTER_ADDR MASTER_PORT
          unset HOSTS HOSTFILE
          bash "${entry}"
        ) >"${local_log}" 2>&1 &
      else
        target="$(_dist_ssh_target "${host}")"
        (
          set -euo pipefail
          _dist_ssh "${target}" "bash -c $(printf '%q' "$(_dist_worker_script "${rank}" "${entry}")")"
        ) >"${local_log}" 2>&1 &
      fi
    fi
    _dist_child_pids+=("$!")
  done

  status=0
  for pid in "${_dist_child_pids[@]}"; do
    rc=0
    wait "${pid}" || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
      status="${rc}"
    fi
  done
  trap - INT TERM
  if [[ "${status}" -ne 0 ]]; then
    echo "[error] one or more workers exited with status ${status}" >&2
    exit "${status}"
  fi
  echo "[info] all workers finished"
  exit 0
}

dist_ensure_dataset() {
  local train_jsonl="${ROOT}/data/hermas_sys2_train.jsonl"
  local train_json="${ROOT}/data/hermas_sys2_train_20k.json"
  local wait_sec="${DATASET_WAIT_SEC:-1800}"

  if [[ -f "${train_jsonl}" || -f "${train_json}" ]]; then
    return 0
  fi

  if [[ "${NODE_RANK}" -eq 0 ]]; then
    echo "[info] dataset missing, running convert first..."
    bash "${ROOT}/scripts/run_convert.sh"
    return 0
  fi

  echo "[info] node_rank=${NODE_RANK}: waiting up to ${wait_sec}s for dataset..."
  local elapsed=0
  while [[ "${elapsed}" -lt "${wait_sec}" ]]; do
    if [[ -f "${train_jsonl}" || -f "${train_json}" ]]; then
      echo "[info] dataset ready"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "[error] dataset still missing after ${wait_sec}s; convert on node 0 or share data/" >&2
  exit 1
}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

if [[ -z "${NPROC:-}" ]]; then
  NPROC="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
fi

_node_rank_provided=0
[[ -n "${NODE_RANK+x}" ]] && _node_rank_provided=1
_nnodes_provided=0
[[ -n "${NNODES+x}" ]] && _nnodes_provided=1
_master_addr_provided=0
[[ -n "${MASTER_ADDR+x}" ]] && _master_addr_provided=1

DIST_HOSTS=()
_dist_load_hosts

if [[ "${#DIST_HOSTS[@]}" -gt 0 ]]; then
  if [[ "${_nnodes_provided}" -eq 0 ]]; then
    NNODES="${#DIST_HOSTS[@]}"
  elif [[ "${NNODES}" -ne "${#DIST_HOSTS[@]}" ]]; then
    echo "[warn] NNODES=${NNODES} != host list length ${#DIST_HOSTS[@]}; using host list" >&2
    NNODES="${#DIST_HOSTS[@]}"
  fi
  if [[ "${_master_addr_provided}" -eq 0 ]]; then
    MASTER_ADDR="$(_dist_bare_host "${DIST_HOSTS[0]}")"
  fi
  if _dist_is_coordinator; then
    NODE_RANK="${NODE_RANK:-0}"
  elif [[ "${_node_rank_provided}" -eq 0 ]]; then
    if [[ "${#DIST_HOSTS[@]}" -eq 1 ]]; then
      NODE_RANK=0
    elif ! NODE_RANK="$(_dist_match_rank)"; then
      echo "[error] this machine is not in HOSTS/HOSTFILE; set NODE_RANK explicitly" >&2
      echo "[error] hosts: ${DIST_HOSTS[*]}" >&2
      exit 1
    fi
  fi
else
  _dist_from_slurm
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] NNODES must be a positive integer, got: ${NNODES}" >&2
  exit 1
fi
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || [[ "${NODE_RANK}" -ge "${NNODES}" ]]; then
  echo "[error] NODE_RANK must be in [0, NNODES), got NODE_RANK=${NODE_RANK} NNODES=${NNODES}" >&2
  exit 1
fi
if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] NPROC must be a positive integer, got: ${NPROC}" >&2
  exit 1
fi

export MASTER_ADDR MASTER_PORT
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

DIST_LAUNCH=( "${PYTHON}" -u -m torch.distributed.run --nproc_per_node="${NPROC}" )
if [[ "${NNODES}" -eq 1 ]]; then
  DIST_LAUNCH+=( --standalone )
else
  DIST_LAUNCH+=(
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )
fi
