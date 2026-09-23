#!/usr/bin/env bash
# 下载 RISE online RL 所需的动力学权重：
#   1) Lightricks/LTX-Video 的 tokenizer / text_encoder / vae
#   2) OpenDriveLab-org/RISE_Assets 的 pretrained diffusion
# 用法:
#   bash scripts/setup_rise_dynamics_ckpts.sh
#   RISE_DYNAMICS_ROOT=/path bash scripts/setup_rise_dynamics_ckpts.sh
set -euo pipefail

ROOT="${RISE_DYNAMICS_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/models/rise_dynamics}"
BACKBONE="${ROOT}/checkpoints"
PRETRAINED="${ROOT}/pretrained"
HF="${HF_CLI:-}"
if [[ -z "${HF}" ]]; then
  if [[ -x /home/psibot/miniconda3/envs/RLinf/bin/huggingface-cli ]]; then
    HF=/home/psibot/miniconda3/envs/RLinf/bin/huggingface-cli
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF="$(command -v huggingface-cli)"
  elif command -v hf >/dev/null 2>&1; then
    HF="$(command -v hf)"
  else
    echo "错误: 未找到 huggingface-cli / hf" >&2
    exit 1
  fi
fi

mkdir -p "${BACKBONE}" "${PRETRAINED}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "[setup] backbone -> ${BACKBONE}"
"${HF}" download Lightricks/LTX-Video \
  --include "tokenizer/*" "text_encoder/*" "vae/*" \
  --local-dir "${BACKBONE}"

echo "[setup] pretrained dynamics -> ${PRETRAINED}"
TMP="${ROOT}/pretrained_tmp"
rm -rf "${TMP}"
"${HF}" download OpenDriveLab-org/RISE_Assets \
  --include "dynamics_model/pretrained/*" \
  --local-dir "${TMP}"
if [[ -f "${TMP}/dynamics_model/pretrained/diffusion_pytorch_model.safetensors" ]]; then
  mv "${TMP}/dynamics_model/pretrained/"* "${PRETRAINED}/"
fi
rm -rf "${TMP}"

INFER="${ROOT}/infer.yaml"
RISE_DYN_MODELS="${RISE_DYNAMICS_MODELS:-/share_data/projects/mahjong/share/personal/liyichao/RoboDojo/XPolicyLab/policy/RISE/RISE/dynamics/dynamics_model/models}"
cat > "${INFER}" <<EOF
pretrained_model_name_or_path: ${BACKBONE}

tokenizer_class_path: transformers
tokenizer_class: T5Tokenizer
textenc_class_path: transformers
textenc_class: T5EncoderModel
vae_class_path: ${RISE_DYN_MODELS}/ltx_models/autoencoder_kl_ltx.py
vae_class: AutoencoderKLLTXVideo

diffusion_model_class_path: ${RISE_DYN_MODELS}/ltx_models/action_encoder_control.py
diffusion_model_class: LTXVideoTransformer3DModel
diffusion_scheduler_class_path: diffusers
diffusion_scheduler_class: FlowMatchEulerDiscreteScheduler

pipeline_class_path: ${RISE_DYN_MODELS}/pipeline/custom_pipeline.py
pipeline_class: CustomPipeline

return_video: true
mixed_precision: bf16
allow_tf32: False
nccl_timeout: 600
seed: 57

diffusion_model:
  model_path: ${PRETRAINED}/diffusion_pytorch_model.safetensors
  config:
    activation_fn: gelu-approximate
    attention_bias: true
    attention_head_dim: 64
    attention_out_bias: true
    caption_channels: 4096
    cross_attention_dim: 2048
    in_channels: 128
    norm_elementwise_affine: false
    norm_eps: 1.0e-6
    num_attention_heads: 32
    num_layers: 28
    out_channels: 128
    patch_size: 1
    patch_size_t: 1
    qk_norm: rms_norm_across_heads
    action_expert: false
    action_in_channels: 14
    action_num_attention_heads: 16
    action_attention_head_dim: 32

use_color_jitter: true
num_inference_step: 50
noisy_video: true
load_weights: true
EOF
echo "[setup] wrote ${INFER}"

echo "[setup] verifying..."
test -f "${BACKBONE}/tokenizer/tokenizer_config.json"
test -f "${BACKBONE}/text_encoder/config.json"
test -f "${BACKBONE}/vae/config.json"
test -f "${PRETRAINED}/diffusion_pytorch_model.safetensors"
test -f "${INFER}"
echo "[setup] rise dynamics ckpts ready under ${ROOT}"
