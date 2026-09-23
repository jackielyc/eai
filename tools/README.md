# GPU Stress Test

在多张 GPU 上持续执行 GEMM（矩阵乘法），把 **SM 利用率**拉高，同时默认只占少量显存。用于压测、预热或验证 CUDA/PyTorch 环境。可用 `-u` / `--max-util` 设定最高利用率；要用更大矩阵占满显存时再加大 `-s` 或 `--mem-fraction`。

## 依赖

- 已安装 **PyTorch + CUDA** 的 Python 解释器
- 可见的 NVIDIA GPU（`nvidia-smi` 能列出设备）

## 快速开始

在 `eai` 仓库根目录下：

```bash
# 默认：自动检测空闲 GPU 并压测，直到 Ctrl+C
bash tools/train.sh
```

另开终端观察利用率：

```bash
watch -n1 nvidia-smi
```

## 推荐用法（Shell 封装）

`train.sh` 会调用 `train.py`，并默认使用：

`miniconda3/envs/Qwen2.5-VL/bin/python`

| 场景 | 命令 |
|------|------|
| 自动检测空闲 GPU 并压测 | `bash tools/train.sh` |
| 最多使用 2 张空闲 GPU | `bash tools/train.sh -n 2` |
| 强制压测所有可见 GPU | `bash tools/train.sh --all-gpus` |
| 指定卡号（仍跳过繁忙卡） | `bash tools/train.sh --gpu-ids 0,2,4` |
| 多机远程压测（Ctrl+C 会同步停远程） | `bash tools/train.sh --hosts gpu-a,gpu-b gpu-c` |
| 从文件读 hosts | `bash tools/train.sh --hosts-file tools/hosts.txt` |
| 跑 5 分钟后自动停止 | `bash tools/train.sh -d 300` |
| 把利用率压到约 70% | `bash tools/train.sh -u 70` |
| 故意占更多显存 | `bash tools/train.sh -s 16384 --mem-fraction 0.85` |
| 利用率 ≤20% 的卡视为空闲 | `bash tools/train.sh -t 20` |
| 限制可见设备后再压测 | `CUDA_VISIBLE_DEVICES=2,3 bash tools/train.sh` |
| 查看帮助 | `bash tools/train.sh -h` |

### 环境变量

| 变量 | 说明 |
|------|------|
| `CUDA_VISIBLE_DEVICES` | 限制 PyTorch 可见的 GPU（逗号分隔），例如 `0,1` |
| `PYTHON` | 指定带 CUDA 的 Python，例如 `PYTHON=/path/to/python bash tools/train.sh` |

## 直接调用 Python

```bash
PYTHON=/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/Qwen2.5-VL/bin/python

# 自动检测空闲 GPU（默认）
$PYTHON tools/train.py

# 强制使用所有可见 GPU
$PYTHON tools/train.py --all-gpus

# 最多 2 张空闲 GPU，bf16，跑 600 秒
$PYTHON tools/train.py -n 2 --dtype bf16 -d 600

# 最高利用率 70%
$PYTHON tools/train.py -u 70

# 闲置门槛 20%（利用率不超过 20% 的卡才压）
$PYTHON tools/train.py -t 20

# 手动指定更大矩阵（会明显占显存）
$PYTHON tools/train.py --gpu-ids 0,1 -s 16384 --streams 8
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-n`, `--num-gpus` | 全部空闲 | 在已选 GPU 中最多使用 N 张 |
| `--gpu-ids` | — | 逗号分隔的 GPU 索引；默认仍会跳过繁忙卡 |
| `--all-gpus` | 关闭 | 跳过空闲检测，使用所有可见 GPU |
| `-t`, `--idle-util` | `5` | 闲置卡利用率门槛（%）。`nvidia-smi` 利用率不超过该值才视为空闲。`--idle-util-max` 为同义参数 |
| `--idle-mem-mib` | `512` | 判定空闲的最大已用显存（MiB，与显存占比联合判断） |
| `--idle-mem-frac-max` | `0.05` | 判定空闲的最大显存占用比例 |
| `--allow-compute-procs` | 关闭 | 允许已有 compute 进程的 GPU 参与压测 |
| `-u`, `--max-util` | `100` | 压测时的最高 GPU 利用率（%）。小于 100 时按实际 kernel 时间补偿休眠 |
| `--hosts` | — | 远程主机列表（空格或逗号分隔）。对本机名会本地执行，其余走 ssh |
| `--hosts-file` | — | 每行一个 host，`#` 开头为注释 |
| `--ssh-user` | — | 未写 `user@host` 时使用的 SSH 用户 |
| `--ssh-opts` | — | 额外 ssh 参数，例如 `'-p 2222 -i /path/key'` |
| `-d`, `--duration` | `0` | 运行秒数；`0` 表示直到 Ctrl+C |
| `-s`, `--matrix-size` | `0` | GEMM 维度 N×N；`0` 按空闲显存自动估算，且自动上限为 8192（约 1 GB 级显存） |
| `--dtype` | `fp16` | 计算精度：`fp16` / `bf16` / `fp32`（fp16/bf16 在支持的卡上会走 Tensor Core） |
| `--streams` | `4` | 每张 GPU 的 CUDA stream 数，用于重叠计算 |
| `--mem-fraction` | `0.08` | 自动估算矩阵大小时，使用的空闲显存比例（默认偏低，避免占满显存） |
| `--report-interval` | `10` | 每张 GPU 打印吞吐日志的间隔（秒）；`0` 关闭 |

## 运行时会看到什么

启动示例：

```text
[info] GPU 0 (torch 0): busy | util=98%, mem=72.1/80.0 GB (90.1%), procs=2
[info] GPU 1 (torch 1): idle | util=0%, mem=0.4/80.0 GB (0.5%)
[info] visible=(all) mode=idle auto-detect using GPUs: [1, 3]
[info] running until Ctrl+C
[gpu 1] NVIDIA A100-SXM4-80GB | size=8192 dtype=fp16 streams=4
```

- 每张 GPU 独立进程（`spawn`），互不阻塞
- 显存不足时会自动缩小矩阵并重试
- `Ctrl+C` 或到达 `-d` 时长后会优雅退出

## 多机远程执行

需要各机器能 **免密 SSH**（`BatchMode`），并且能访问同一份脚本 / Python（例如共享盘）。

```bash
# 逗号或空格均可
bash tools/train.sh --hosts gpu-a,gpu-b gpu-c

# hosts 文件
cat > /tmp/gpu_hosts.txt <<'EOF'
gpu-a
gpu-b
# gpu-d   skipped
user@gpu-c
EOF
bash tools/train.sh --hosts-file /tmp/gpu_hosts.txt --all-gpus -d 300
```

本地日志会带主机前缀：

```text
[info] remote hosts=['gpu-a', 'gpu-b']
[gpu-a] [info] GPU 0 (torch 0): idle | util=0%, mem=0.4/80.0 GB (0.5%)
[gpu-b] [info] using GPUs: [1, 2]
```

**退出同步**：本地 `Ctrl+C`、收到 `SIGTERM`、或进程退出时，会先给各 ssh/本地子进程发 `SIGINT`/`SIGTERM`。远端用 `exec python ...` 跑，SSH 断开后会收到 `SIGHUP`，压测进程一并退出。

远程机需要与当前机器相同的 `PYTHON` 路径和脚本路径（共享 NFS 时通常已满足）。

## 注意事项

1. **默认只压空闲 GPU**（通过 `nvidia-smi` 看利用率、显存占用、compute 进程）；训练中的卡会被跳过。若要强制占满所有卡，加 `--all-gpus`。
2. **`--max-util` 按忙碌时间补偿休眠**。大矩阵单次 GEMM 可能超过 1 秒，旧的固定窗口会来不及休眠、看起来像没生效；现在会缩小计算 tile，并把 `nvidia-smi` 利用率打在日志里。
3. 自动矩阵大小基于**当前空闲显存的一小部分**（默认 8%，N 上限 8192），优先打满利用率而不是占满显存。需要更大占用来验证 OOM/显存带宽时，用 `-s 16384` 或 `--mem-fraction 0.85`。
4. 若提示找不到 Python，设置 `PYTHON` 指向已安装 PyTorch CUDA 版的环境。
5. 本工具只做计算压测，**不读写磁盘、不涉及模型权重**。

## 文件

| 文件 | 作用 |
|------|------|
| `train.py` | 核心逻辑：多进程 GEMM 压测 |
| `train.sh` | 便捷入口，设置默认 Python 并转发参数 |
