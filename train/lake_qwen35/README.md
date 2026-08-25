# share_data_lake × Qwen3.5 Multimodal LoRA SFT

用 `/share_data_lake` 的 clip index + zarr RGB 帧，微调 Qwen3.5-4B / 35B-A3B（图像 + 任务文本 → Skill/Subtask/Memory）。

目录结构对齐 `eai/train/cortex_qwen35`：

```text
lake_qwen35/
  scripts/
    convert_lake_to_sft.py   # lake clip → JSON + jpg
    train_lora_sft_mm.py     # 多模态 LoRA 训练
    run_convert.sh
    run_smoke_4b.sh
    run_train_4b.sh
    run_train_35b.sh
  configs/
  data/
  output/
```

## 环境

```bash
# 训练
export PYTHON=/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/Qwen2.5-VL/bin/python
# 数据转换（需要 zarr）
export CONVERT_PYTHON=/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/psi-policy/bin/python
cd /share_data/projects/mahjong/share/personal/liyichao/eai/train/lake_qwen35
```

## 1. 转换数据

默认从 `hermes-human-ego-10029` / `10029-hermes-data-3_VA48DX` 导出（Hermes 人形 ego 数据，图像在 zarr 的 `image_rectified` JPEG 字段）：

```bash
bash scripts/run_convert.sh
```

该 view 约有 **535 万 train / 59 万 val** clip，默认只导出 subset（`SKIP_FULL=1`），不会全量扫描。

自定义来源：

```bash
DATAHOUSE_ID=hermes-human-ego-10029 \
VIEW_ID=10029-hermes-data-3_VA48DX \
CAMERA=auto \
SUBSET_TRAIN=20000 SUBSET_VAL=2000 \
bash scripts/run_convert.sh
```

机器人 RGB 数据（如 box-bag）示例：

```bash
DATAHOUSE_ID=box-bag-pick-place-research-10014 \
VIEW_ID=box-pick-place-001_B16IKQ \
CAMERA=rgb_head SKIP_FULL=1 \
bash scripts/run_convert.sh
```

产物：

- `data/lake_sys2_train_20k.json` / `lake_sys2_val_2k.json`
- `data/images/<datahouse>/<view>/<split>/*.jpg`

### 按任务（task）导出

Hermes 当前 view 约有 **1484 种 task**（`clip_index_*.parquet` 的 `task` 列，中文任务名）。

**每个 task 导出 N 条（均衡采样，推荐）：**

```bash
# 每个 task 随机取 50 条 train + 50 条 val → 约 1484×50 ≈ 7.4 万 train
MAX_PER_TASK=50 SKIP_FULL=1 bash scripts/run_convert.sh
```

**只导出指定 task：**

```bash
TASKS='安装电动牙刷刷头,将回形针放回回形针盒中' \
MAX_PER_TASK=100 \
SKIP_FULL=1 \
bash scripts/run_convert.sh
```

**从文件读取 task 列表（一行一个）：**

```bash
TASK_LIST_FILE=/path/to/tasks.txt MAX_PER_TASK=200 SKIP_FULL=1 bash scripts/run_convert.sh
```

**某 task 全量导出：**

```bash
TASKS='安装电动牙刷刷头' SKIP_FULL=0 bash scripts/run_convert.sh
# 产出 data/lake_sys2_train.json（该 task 在 train split 的全部 clip）
```

**查看 task 种类与数量：**

```bash
/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/psi-policy/bin/python - <<'PY'
import pandas as pd
p="/share_data_lake/hermes-human-ego-10029/views/10029-hermes-data-3_VA48DX/clip_index_train.parquet"
print(pd.read_parquet(p, columns=["task"])["task"].value_counts().head(20))
PY
```

| 环境变量 | 含义 |
|---|---|
| `MAX_PER_TASK` | 每个 task 最多导出 N 条（train/val 各自独立采样） |
| `TASKS` | 逗号分隔的 task 名，精确匹配 |
| `TASK_LIST_FILE` | 每行一个 task 名的文本文件 |
| `SUBSET_TRAIN` / `SUBSET_VAL` | 全局随机上限（与 `MAX_PER_TASK` 同时设时，以 `MAX_PER_TASK` 为准） |
| `SKIP_FULL=0` | 不做全局/每 task 上限，导出筛选后的全部 clip |

### 全量流式导出 + 断点续传

全量导出（`SKIP_FULL=0`）默认：
- **流式写** `data/lake_sys2_train.jsonl` / `lake_sys2_val.jsonl`（不把所有样本堆进内存）
- **断点续传** `data/.convert_checkpoint/{train,val}.clip_ids` + 已有 jpg/jsonl
- 分块读取 parquet（535 万 clip 不会一次性载入 RAM）

```bash
# 全量 Hermes（可 tmux 后台跑，中断后原命令重跑即可续传）
SKIP_FULL=0 RESUME=1 SKIP_EXISTING=1 bash scripts/run_convert.sh

# 从头重导（清空 checkpoint 与 jsonl）
SKIP_FULL=0 RESUME=0 bash scripts/run_convert.sh
```

全量训练请指向 jsonl：

```yaml
dataset_path: .../data/lake_sys2_train.jsonl
eval_dataset_path: .../data/lake_sys2_val.jsonl
```

子集兼容文件仍会自动生成：`lake_sys2_train_20k.json`（从 jsonl 流式截取前 N 行）。

### 加速导出

默认已开启：
- **`NUM_WORKERS=$(nproc)`**：默认占用全部 CPU 核并行解码/写盘
- **`SKIP_EXISTING=1`**：jpg 已存在则跳过 zarr 读取（断点续跑）
- Hermes **`image_rectified`**：直接写 JPEG 字节，不再 decode→re-encode

```bash
# 默认全核 + 断点续跑
SKIP_EXISTING=1 MAX_PER_TASK=50 bash scripts/run_convert.sh

# 限制并行度（例如只用 16 核）
NUM_WORKERS=16 SKIP_EXISTING=1 bash scripts/run_convert.sh

# 强制重导全部图片
SKIP_EXISTING=0 bash scripts/run_convert.sh
```

> assistant 标签目前由 `task` 名模板生成；若有平台语义标注，可在 `convert_lake_to_sft.py` 中替换 `_build_assistant()`。

## 2. 训练

```bash
# 冒烟（1 卡，16 条）
GPU=0 bash scripts/run_smoke_4b.sh

# 4B 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NPROC=6 bash scripts/run_train_4b.sh

# 35B-A3B（device_map=auto）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_train_35b.sh
```

## 与 cortex_qwen35 的区别

| | cortex_qwen35 | lake_qwen35 |
|---|---|---|
| 数据 | Cortex JSONL 文本 | share_data_lake zarr + RGB |
| convert | `convert_cortex_to_sft.py` | `convert_lake_to_sft.py` |
| train | `train_lora_sft.py`（纯文本） | `train_lora_sft_mm.py`（processor + 图像） |

## 说明

- 多模态训练默认 `per_device_train_batch_size=1`（Qwen3-VL batch 需合并 `pixel_values`）。
- 脚本会自动计算 `position_ids` / `mm_token_type_ids`（Qwen3.5 必需）。
- 若 OOM，调低 `image_max_pixels`（yaml 或 `--image_max_pixels`）。
