#!/usr/bin/env bash
# 启动 RynnValue（视频推理 / Reward 服务 / Robometer 评测）。
# 由 eai viewer「RynnValue」页调用；也可手动：
#   bash run_rynnvalue.sh --job infer --model-path /path/to/RynnValue-8B \
#     --video-path /path/to/demo.mp4 --instruction "Put the box in the drawer"
#   bash run_rynnvalue.sh --job reward_server --model-path /path/to/RynnValue-8B
set -euo pipefail

DEFAULT_ROOT="/share_data/projects/mahjong/share/personal/liyichao/RynnValue"
ROOT="${RYNNVALUE_ROOT:-${DEFAULT_ROOT}}"
PY="${RYNNVALUE_PYTHON:-}"
JOB="infer"
MODEL_PATH=""
CHECKPOINT_PATH=""
VIDEO_PATH=""
INSTRUCTION=""
ROBOT_DESCRIPTION="${RYNNVALUE_ROBOT_DESCRIPTION:-}"
CAMERA_DESCRIPTION="${RYNNVALUE_CAMERA_DESCRIPTION:-}"
OUTPUT_PATH=""
# 64 frames @640px → ~15k tokens; pred_slot_isolated_eager softmax needs ~28GiB alone.
NUM_FRAMES="16"
# Upstream defaults (num_steps=0 → every frame, batch_size=4) OOM on ~44GB with
# pred_slot_isolated_eager attention; keep denser sampling opt-in.
NUM_STEPS="32"
BATCH_SIZE="1"
MAX_IMAGE_SIDE="640"
MAX_NEW_TOKENS="128"
FPS="30"
PORT="8001"
GPU="0"
DATASETS_PATH=""
EXTRA_ARGS=()
GUI=0
# Released RynnValue-* have use_meta=True; example clip is ALOHA top-down.
DEFAULT_ROBOT_DESCRIPTION="an ALOHA dual-arm robot"
DEFAULT_CAMERA_DESCRIPTION="the top-down camera"

usage() {
  cat <<'EOF'
用法:
  bash run_rynnvalue.sh [选项]

选项:
  --job NAME            infer | reward_server | policy_ranking | confusion_matrix
  --root PATH           RynnValue 仓库根
  --python PATH         Python（建议仓库 .venv；空则优先 uv run）
  --model-path PATH     HuggingFace 模型目录或 repo id
  --checkpoint-path PATH  训练 checkpoint 目录（含 model.pt；可选）
  --video-path PATH     推理输入视频（infer）
  --instruction TEXT    任务指令（infer）
  --robot-description TEXT   机器人描述（use_meta=True 必需；默认 ALOHA）
  --camera-description TEXT  相机视角描述（use_meta=True 必需；默认 top-down）
  --output-path PATH    推理输出目录（默认 <root>/rynn_infer/outputs）
  --num-frames N        每个 prefix 采样帧数（默认 16；64 在 44GB 上易 OOM）
  --num-steps N         时间轴采样点数（默认 32；0=每帧，很慢且更吃显存）
  --batch-size N        每步 batch（默认 1；>1 易 OOM）
  --max-image-side N    送入模型前长边上限（默认 640；可降到 448 省显存）
  --max-new-tokens N    Analysis 生成长度（默认 128）
  --fps N               趋势视频帧率（默认 30）
  --port N              reward_server 端口（默认 8001）
  --gpu N               CUDA_VISIBLE_DEVICES（默认 0）
  --datasets-path PATH  ROBOMETER_PROCESSED_DATASETS_PATH（评测）
  --gui                 图形化标记（UI 播放趋势视频；评测 save_videos=true）
  -h, --help            显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job) JOB="${2:-}"; shift 2 ;;
    --root) ROOT="${2:-}"; shift 2 ;;
    --python) PY="${2:-}"; shift 2 ;;
    --model-path) MODEL_PATH="${2:-}"; shift 2 ;;
    --checkpoint-path) CHECKPOINT_PATH="${2:-}"; shift 2 ;;
    --video-path) VIDEO_PATH="${2:-}"; shift 2 ;;
    --instruction) INSTRUCTION="${2:-}"; shift 2 ;;
    --robot-description) ROBOT_DESCRIPTION="${2:-}"; shift 2 ;;
    --camera-description) CAMERA_DESCRIPTION="${2:-}"; shift 2 ;;
    --output-path) OUTPUT_PATH="${2:-}"; shift 2 ;;
    --num-frames) NUM_FRAMES="${2:-}"; shift 2 ;;
    --num-steps) NUM_STEPS="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --max-image-side) MAX_IMAGE_SIDE="${2:-}"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="${2:-}"; shift 2 ;;
    --fps) FPS="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --gpu) GPU="${2:-}"; shift 2 ;;
    --datasets-path) DATASETS_PATH="${2:-}"; shift 2 ;;
    --gui) GUI=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

ROOT="$(cd "${ROOT}" && pwd)"
if [[ ! -d "${ROOT}/rynn_infer" ]] || [[ ! -d "${ROOT}/rynn_value" ]]; then
  echo "错误: 无效仓库（缺少 rynn_infer/ 或 rynn_value/）: ${ROOT}" >&2
  exit 1
fi

JOB_NORM="$(echo "${JOB}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

if [[ -z "${PY}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PY="${ROOT}/.venv/bin/python"
  elif ! command -v uv >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
      PY="$(command -v python3)"
    else
      echo "错误: 找不到 Python。请设置 RYNNVALUE_PYTHON 或 --python" >&2
      exit 1
    fi
  fi
fi

export RYNNVALUE_ROOT="${ROOT}"
export RYNNVALUE_GUI="${GUI}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
# Viewer/RoboStack injects ros-humble into PYTHONPATH; rynnvalue (3.10) must not
# import that env's numpy/site-packages (3.11) or imports abort.
unset PYTHONHOME PYTHONPATH AMENT_PREFIX_PATH ROS_DISTRO ROS_VERSION COLCON_PREFIX_PATH || true
# 始终 Agg：趋势视频离屏渲染；「图形化」由 UI 播放输出 mp4
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

if [[ -n "${PY}" ]]; then
  export RYNNVALUE_PYTHON="${PY}"
  PY_BIN_DIR="$(cd "$(dirname "${PY}")" && pwd)"
  export PATH="${PY_BIN_DIR}:${PATH}"
  # pip-installed torch+cu124 keeps cudnn under site-packages/nvidia/*/lib;
  # without these on LD_LIBRARY_PATH, conv/attention can abort with
  # "Unable to load ... libcudnn_graph.so".
  SITE_PACKAGES="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths().get("purelib",""))' 2>/dev/null || true)"
  if [[ -n "${SITE_PACKAGES}" && -d "${SITE_PACKAGES}/nvidia" ]]; then
    NVIDIA_LIB_DIRS=()
    shopt -s nullglob
    for d in "${SITE_PACKAGES}"/nvidia/*/lib "${SITE_PACKAGES}"/nvidia/*/lib64; do
      [[ -d "${d}" ]] && NVIDIA_LIB_DIRS+=("${d}")
    done
    shopt -u nullglob
    if [[ ${#NVIDIA_LIB_DIRS[@]} -gt 0 ]]; then
      NVIDIA_LIBS="$(IFS=:; echo "${NVIDIA_LIB_DIRS[*]}")"
      export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      echo "[rynnvalue] nvidia_libs=${#NVIDIA_LIB_DIRS[@]} dirs prepended to LD_LIBRARY_PATH"
    fi
  fi
fi

LOG_DIR="${ROOT}/rynn_infer/outputs"
mkdir -p "${LOG_DIR}"
echo "[rynnvalue] log_dir=${LOG_DIR}"
echo "[rynnvalue] job=${JOB_NORM} gui=${GUI}"
echo "[rynnvalue] root=${ROOT}"
echo "[rynnvalue] python=${PY:-uv run python}"

_run_in() {
  local cwd="$1"; shift
  if [[ -n "${PY}" ]]; then
    (cd "${cwd}" && "${PY}" "$@")
  else
    (cd "${cwd}" && uv run python "$@")
  fi
}

case "${JOB_NORM}" in
  infer|inference|demo)
    if [[ -z "${MODEL_PATH}" ]]; then
      echo "错误: infer 需要 --model-path" >&2
      exit 1
    fi
    if [[ -z "${VIDEO_PATH}" ]]; then
      VIDEO_PATH="${ROOT}/example/Put_the_box_in_the_drawer_and_close_it.mp4"
    fi
    if [[ ! -f "${VIDEO_PATH}" ]]; then
      echo "错误: 视频不存在: ${VIDEO_PATH}" >&2
      exit 1
    fi
    if [[ -z "${INSTRUCTION}" ]]; then
      INSTRUCTION="Put the box in the drawer and close it"
    fi
    # Released checkpoints ship with use_meta=True and refuse empty meta.
    if [[ -z "${ROBOT_DESCRIPTION}" ]]; then
      ROBOT_DESCRIPTION="${DEFAULT_ROBOT_DESCRIPTION}"
    fi
    if [[ -z "${CAMERA_DESCRIPTION}" ]]; then
      CAMERA_DESCRIPTION="${DEFAULT_CAMERA_DESCRIPTION}"
    fi
    OUTPUT_PATH="${OUTPUT_PATH:-${LOG_DIR}}"
    mkdir -p "${OUTPUT_PATH}"
    echo "[rynnvalue] model=${MODEL_PATH}"
    echo "[rynnvalue] video=${VIDEO_PATH}"
    echo "[rynnvalue] robot=${ROBOT_DESCRIPTION}"
    echo "[rynnvalue] camera=${CAMERA_DESCRIPTION}"
    echo "[rynnvalue] num_frames=${NUM_FRAMES} num_steps=${NUM_STEPS} batch_size=${BATCH_SIZE} max_image_side=${MAX_IMAGE_SIDE}"
    CMD=(
      inference.py
      --model_path "${MODEL_PATH}"
      --video_path "${VIDEO_PATH}"
      --instruction "${INSTRUCTION}"
      --robot_description "${ROBOT_DESCRIPTION}"
      --camera_description "${CAMERA_DESCRIPTION}"
      --num_frames "${NUM_FRAMES}"
      --num_steps "${NUM_STEPS}"
      --batch_size "${BATCH_SIZE}"
      --max_image_side "${MAX_IMAGE_SIDE}"
      --max_new_tokens "${MAX_NEW_TOKENS}"
      --fps "${FPS}"
      --output_path "${OUTPUT_PATH}"
    )
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      CMD+=("${EXTRA_ARGS[@]}")
    fi
    echo "$ cd ${ROOT}/rynn_infer && ${PY:-uv run python} ${CMD[*]}"
    # Tee output so we can surface the Saved mp4 path for the UI player.
    OUT_LOG="$(mktemp)"
    set +e
    if [[ -n "${PY}" ]]; then
      (cd "${ROOT}/rynn_infer" && "${PY}" "${CMD[@]}" 2>&1 | tee "${OUT_LOG}")
    else
      (cd "${ROOT}/rynn_infer" && uv run python "${CMD[@]}" 2>&1 | tee "${OUT_LOG}")
    fi
    RC=${PIPESTATUS[0]}
    set -e
    SAVED="$(grep -E '^Saved .+\.mp4$' "${OUT_LOG}" | tail -n1 | sed 's/^Saved //' || true)"
    rm -f "${OUT_LOG}"
    if [[ -n "${SAVED}" ]]; then
      echo "[rynnvalue] output_video=${SAVED}"
      if [[ "${GUI}" == "1" && -f "${SAVED}" ]]; then
        # CLI --gui：推理本身用 Agg 离屏渲染；完成后打开趋势视频。
        if command -v xdg-open >/dev/null 2>&1; then
          echo "[rynnvalue] opening video with xdg-open…"
          xdg-open "${SAVED}" >/dev/null 2>&1 || true
        elif command -v ffplay >/dev/null 2>&1; then
          echo "[rynnvalue] opening video with ffplay…"
          ffplay -autoexit -loglevel error "${SAVED}" >/dev/null 2>&1 || true
        else
          echo "[rynnvalue] 无播放器（xdg-open/ffplay），请手动打开: ${SAVED}"
        fi
      fi
    fi
    exit "${RC}"
    ;;

  reward_server|server|serve)
    if [[ -z "${MODEL_PATH}" && -z "${CHECKPOINT_PATH}" ]]; then
      echo "错误: reward_server 需要 --model-path 或 --checkpoint-path" >&2
      exit 1
    fi
    SERVER_SH="${ROOT}/robometer/rynnvalue_eval/start_server.sh"
    if [[ ! -f "${SERVER_SH}" ]]; then
      echo "错误: 未找到 ${SERVER_SH}" >&2
      exit 1
    fi
    ARGS=(--port "${PORT}" --gpu "${GPU}" --num-frames "${NUM_FRAMES}")
    if [[ -n "${MODEL_PATH}" ]]; then
      ARGS+=(--model-path "${MODEL_PATH}")
    fi
    if [[ -n "${CHECKPOINT_PATH}" ]]; then
      ARGS+=(--checkpoint-path "${CHECKPOINT_PATH}")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      ARGS+=("${EXTRA_ARGS[@]}")
    fi
    echo "$ bash ${SERVER_SH} ${ARGS[*]}"
    cd "${ROOT}/robometer"
    exec bash "${SERVER_SH}" "${ARGS[@]}"
    ;;

  policy_ranking|ranking)
    if [[ -z "${MODEL_PATH}" ]]; then
      echo "错误: policy_ranking 需要 --model-path" >&2
      exit 1
    fi
    if [[ -n "${DATASETS_PATH}" ]]; then
      export ROBOMETER_PROCESSED_DATASETS_PATH="${DATASETS_PATH}"
    fi
    SAVE_VIDEOS="false"
    if [[ "${GUI}" == "1" ]]; then
      SAVE_VIDEOS="true"
    fi
    HYDRA=(
      "reward_model=rynnvalue"
      "model_path=${MODEL_PATH}"
      "custom_eval.eval_types=[policy_ranking]"
      "custom_eval.policy_ranking=[rbm-1m-ood]"
      "custom_eval.use_frame_steps=false"
      "custom_eval.pad_frames=false"
      "custom_eval.num_examples_per_quality_pr=1000"
      "max_frames=${NUM_FRAMES}"
      "save_videos=${SAVE_VIDEOS}"
    )
    if [[ -n "${CHECKPOINT_PATH}" ]]; then
      HYDRA+=("model_config.checkpoint_path=${CHECKPOINT_PATH}")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      HYDRA+=("${EXTRA_ARGS[@]}")
    fi
    echo "$ python robometer/evals/run_baseline_eval.py ${HYDRA[*]}"
    _run_in "${ROOT}/robometer" robometer/evals/run_baseline_eval.py "${HYDRA[@]}"
    ;;

  confusion_matrix|confusion)
    if [[ -z "${MODEL_PATH}" ]]; then
      echo "错误: confusion_matrix 需要 --model-path" >&2
      exit 1
    fi
    if [[ -n "${DATASETS_PATH}" ]]; then
      export ROBOMETER_PROCESSED_DATASETS_PATH="${DATASETS_PATH}"
    fi
    SAVE_VIDEOS="false"
    if [[ "${GUI}" == "1" ]]; then
      SAVE_VIDEOS="true"
    fi
    HYDRA=(
      "reward_model=rynnvalue"
      "model_path=${MODEL_PATH}"
      "custom_eval.eval_types=[confusion_matrix]"
      "max_frames=${NUM_FRAMES}"
      "save_videos=${SAVE_VIDEOS}"
    )
    if [[ -n "${CHECKPOINT_PATH}" ]]; then
      HYDRA+=("model_config.checkpoint_path=${CHECKPOINT_PATH}")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      HYDRA+=("${EXTRA_ARGS[@]}")
    fi
    echo "$ python robometer/evals/run_baseline_eval.py ${HYDRA[*]}"
    _run_in "${ROOT}/robometer" robometer/evals/run_baseline_eval.py "${HYDRA[@]}"
    ;;

  *)
    echo "错误: 未知 --job: ${JOB}" >&2
    usage
    exit 1
    ;;
esac
