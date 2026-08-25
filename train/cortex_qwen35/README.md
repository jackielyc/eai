# Cortex System-2 × Qwen3.5 LoRA SFT

用 [Steinate/Cortex](https://huggingface.co/datasets/Steinate/Cortex) 的语言记忆/子任务标注，微调：

- `models/Qwen/Qwen3.5-4B`
- `models/Qwen/Qwen3.5-35B-A3B`

训练任务对齐 Cortex System-2：给定任务指令 + language memory，预测 `Skill / Subtask / Memory`。

> 本地 `LLaMA-Factory` 尚未注册 `qwen3_5` / `qwen3_5_moe`，且当前环境 `trl` 与 LF 不兼容。  
> **推荐直接跑本目录的 HuggingFace + PEFT 脚本**（接口与数据格式仍按 LLaMA-Factory ShareGPT 组织）。  
> `configs/llamafactory_*.yaml` 留给 LF 支持 Qwen3.5 后使用。

## 目录

```text
cortex_qwen35/
  data/                 # ShareGPT JSON + dataset_info.json（已转换）
  configs/              # 4B / 35B 训练配置
  scripts/
    convert_cortex_to_sft.py
    train_lora_sft.py
    run_smoke_4b.sh
    run_train_4b.sh
    run_train_35b.sh
  output/
```

## 环境

```bash
# 使用已有环境（transformers 5.15 + peft，可加载 Qwen3.5）
export PYTHON=/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/Qwen2.5-VL/bin/python
```

## 数据

原始 JSONL：`dataset/Steinate/Cortex/*.jsonl`  
已转换好的 SFT 数据在 `data/`（全量约 170 万条；默认训练用 20k 子集）。

重新转换：

```bash
$PYTHON scripts/convert_cortex_to_sft.py
# 只要子集：
$PYTHON scripts/convert_cortex_to_sft.py --skip-full
```

## 训练

当前机器 GPU 全满时请先空出卡，或指定空闲卡：

```bash
# 冒烟（1 卡，32 条）
GPU=0 bash scripts/run_smoke_4b.sh

# Qwen3.5-4B LoRA（默认 8 卡 DDP，20k 数据）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 bash scripts/run_train_4b.sh

# Qwen3.5-35B-A3B LoRA（默认 device_map=auto 切分多卡）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_train_35b.sh
```

全量数据训练时，把对应 yaml 里的 `dataset_path` 改成：

```yaml
dataset_path: .../data/cortex_sys2_train.json
```

## 输出

LoRA adapter 保存在：

- `output/qwen35-4b-lora-cortex`
- `output/qwen35-35b-a3b-lora-cortex`

## 说明

- 当前标注 JSONL **不含真实视频帧**（只有 `video_path` 模板）。本流水线做 **文本 System-2 SFT**（记忆跟踪 + 子任务规划）。
- 若后续挂上原始视频，可再扩展为多模态 SFT（与 Cortex 论文中的 visual stream 一致）。
- transformers 5.x 的 `TrainingArguments` 已移除 `warmup_ratio`，脚本统一使用 `warmup_steps`。
