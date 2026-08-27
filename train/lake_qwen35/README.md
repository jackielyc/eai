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

- `data/hermas_sys2_train_20k.json` / `hermas_sys2_val_2k.json`
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
# 产出 data/hermas_sys2_train.json（该 task 在 train split 的全部 clip）
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
- **流式写** `data/hermas_sys2_train.jsonl` / `hermas_sys2_val.jsonl`（不把所有样本堆进内存）
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
dataset_path: .../data/hermas_sys2_train.jsonl
eval_dataset_path: .../data/hermas_sys2_val.jsonl
```

子集兼容文件仍会自动生成：`hermas_sys2_train_20k.json`（从 jsonl 流式截取前 N 行）。

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

> assistant 预测目标包含：**所有子任务**（同一 `upper_clip_id` 下按时间排序的全部 layer-2 中文列表）、
> **技能 / 当前子任务 / 上一个子任务 / 下一个子任务**。子任务文案来自官方 layer-2 中文（`data_hub.clip_description.description_zh`），
> 本地缓存为 `data/layer2_clip_id2zh.json`、`data/layer2_en2zh.json`、`data/layer2_all_subtasks_by_clip.json`。
> user 只给 **任务** + **上一个子任务**，不把全部子任务当作提示。
> assistant 的 **上一个子任务** 与 user 输入相同（第一步为尚未完成）；**下一个子任务** 为列表中当前步的下一步（最后一步为尚未有下一步）。
> **任务** 使用 layer-1 中文任务名（`task`）。训练样本的 system / user / assistant 模板均为中文。

## 2. 训练

```bash
# 冒烟（1 卡，16 条）
GPU=0 bash scripts/run_smoke_4b.sh

# 4B 单机多卡（默认 8 卡；不设 NNODES 时与原来相同）
bash scripts/run_train_4b.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NPROC=6 bash scripts/run_train_4b.sh

# 4B 多机：在一台机器上执行即可，脚本按 HOSTS ssh 到各节点。
# 需要免密 ssh，且工程路径在各机相同（NFS）。第一个 IP 是 master。
HOSTS=<node0-ip>,<node1-ip> bash scripts/run_train_4b.sh
HOSTFILE=/path/to/hosts.txt bash scripts/run_train_4b.sh   # 每行一个 IP

# 也可以不用 HOSTS，在每台机器上自己起（手动指定 rank）
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> bash scripts/run_train_4b.sh   # 机器 0
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0-ip> bash scripts/run_train_4b.sh   # 机器 1

# 35B-A3B 单机（device_map=auto，单进程切多卡）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_train_35b.sh

# 35B 多机需 DEVICE_MAP=none（auto 不能和 torchrun 一起用）
HOSTS=<node0-ip>,<node1-ip> DEVICE_MAP=none bash scripts/run_train_35b.sh
```

多机环境变量：`HOSTS` / `HOSTFILE`（推荐，一台机器启动）、`SSH_USER`、`SSH_OPTS`、`NNODES`（默认 1）、`NODE_RANK`（默认 0）、`MASTER_ADDR`（默认 `127.0.0.1`）、`MASTER_PORT`（默认 29500）、`NPROC`（默认等于 `CUDA_VISIBLE_DEVICES` 个数）。缺数据时只在 `NODE_RANK=0` 上 convert，其它节点等待共享 `data/`。

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
