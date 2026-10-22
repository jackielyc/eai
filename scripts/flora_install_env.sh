#!/usr/bin/env bash
# Install full FLORA runtime into miniconda3/envs/flora
# Avoid bare `import torch` during install — CUDA init can hang under contention.
# Logs: eai/log/flora_env_install.log  Done marker: eai/log/flora_env.DONE
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="${FLORA_CONDA_ENV_DIR:-/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/flora}"
PY="$ENV/bin/python"
LOG="$ROOT/log/flora_env_install.log"
DONE="$ROOT/log/flora_env.DONE"
mkdir -p "$ROOT/log"
# Douban ~20-70MB/s on this host; Tsinghua ~4MB/s; Aliyun often 404/slow for large wheels
IDX=(-i https://pypi.doubanio.com/simple --trusted-host pypi.doubanio.com)
IDX_FALLBACK=(-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn)

pkg_ok() { "$PY" -m pip show "$1" >/dev/null 2>&1; }
timed_import() {
  # $1 = module, timeout seconds
  timeout "${2:-60}" "$PY" -c "import $1; print($1.__version__ if hasattr($1,'__version__') else 'ok')" 2>/dev/null
}

exec >>"$LOG" 2>&1
echo "==== $(date) flora_install_env.sh start ===="
echo "python=$PY index=${IDX[*]}"
"$PY" -V

pip_ok() { "$PY" -m pip -V >/dev/null 2>&1; }
if ! pip_ok; then
  echo "repair pip via ensurepip"
  "$PY" -m ensurepip --upgrade || true
fi
# Skip upgrading setuptools every run — uninstall on shared FS is very slow (ptlrpc).
echo "pip=$("$PY" -m pip -V 2>/dev/null || echo missing)"

echo "---- torch ----"
if ! pkg_ok torch; then
  if ! "$PY" -m pip install --no-cache-dir --no-compile \
      torch==2.11.0 torchvision==0.26.0 \
      --index-url https://download.pytorch.org/whl/cu128; then
    echo "cu128 pin failed; try latest cu124 from pytorch"
    if ! "$PY" -m pip install --no-cache-dir --no-compile \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124; then
      echo "fallback douban torch"
      "$PY" -m pip install --no-cache-dir --no-compile torch torchvision "${IDX[@]}" || \
        "$PY" -m pip install --no-cache-dir --no-compile torch torchvision "${IDX_FALLBACK[@]}"
    fi
  fi
else
  echo "torch already present: $("$PY" -m pip show torch | awk '/^Version:/{print $2}')"
fi

echo "---- annotate stack ----"
"$PY" -m pip install --no-cache-dir --no-compile \
  "huggingface_hub>=1.0" "transformers==5.8.1" "peft==0.18.1" \
  "accelerate==1.12.0" "safetensors==0.7.0" "av>=15" "numpy<3" "Pillow>=11,<13" \
  "${IDX[@]}" || \
"$PY" -m pip install --no-cache-dir --no-compile \
  "huggingface_hub>=1.0" "transformers==5.8.1" "peft==0.18.1" \
  "accelerate==1.12.0" "safetensors==0.7.0" "av>=15" "numpy<3" "Pillow>=11,<13" \
  "${IDX_FALLBACK[@]}"

echo "---- rewarder stack (vllm) ----"
if ! pkg_ok vllm; then
  if ! "$PY" -m pip install --no-cache-dir --no-compile \
      "vllm==0.24.0" "qwen-vl-utils==0.0.14" "decord==0.6.0" "${IDX[@]}"; then
    echo "douban vllm failed; try tsinghua"
    if ! "$PY" -m pip install --no-cache-dir --no-compile \
        "vllm==0.24.0" "qwen-vl-utils==0.0.14" "decord==0.6.0" "${IDX_FALLBACK[@]}"; then
      echo "vllm==0.24.0 failed; try unpinned vllm on douban"
      "$PY" -m pip install --no-cache-dir --no-compile \
        vllm "qwen-vl-utils==0.0.14" "decord==0.6.0" "${IDX[@]}" || \
      "$PY" -m pip install --no-cache-dir --no-compile \
        vllm "qwen-vl-utils==0.0.14" "decord==0.6.0" "${IDX_FALLBACK[@]}" || true
    fi
  fi
else
  echo "vllm already present: $("$PY" -m pip show vllm | awk '/^Version:/{print $2}')"
fi

echo "---- verify (pip show + timed import) ----"
ok=1
for pkg in torch transformers peft accelerate safetensors av; do
  if pkg_ok "$pkg"; then
    echo "pip OK $pkg=$("$PY" -m pip show "$pkg" | awk '/^Version:/{print $2}')"
  else
    echo "MISSING $pkg"; ok=0
  fi
done
if pkg_ok vllm; then
  echo "pip OK vllm=$("$PY" -m pip show vllm | awk '/^Version:/{print $2}')"
else
  echo "MISSING vllm"; ok=0
fi

# Best-effort imports with timeout (do not block forever)
for mod in torch transformers peft vllm; do
  ver=$(timed_import "$mod" 90) && echo "import OK $mod=$ver" || echo "import SKIP/TIMEOUT $mod (pip present is enough)"
done

if [[ "$ok" -eq 1 ]]; then
  date >"$DONE"
  echo "==== $(date) ALL ENV DONE ====" | tee -a "$DONE"
  "$PY" -m pip list 2>/dev/null | rg -i 'torch|transformers|peft|vllm|accelerate|safetensors|av |huggingface' | tee -a "$DONE"
  exit 0
fi
echo "==== $(date) ENV INCOMPLETE ===="
exit 1
