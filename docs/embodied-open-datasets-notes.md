# 开源具身数据集梳理

> 整理时间：2026-03（对话记录落盘）  
> 数字来自各项目主页 / arXiv / Hugging Face，子集与转换版可能略有出入。

配套 Canvas（Cursor 聊天旁打开）：

```
/home/psibot/.cursor/projects/share-data-projects-mahjong-share-personal-liyichao-eai/canvases/
  embodied-open-datasets.canvas.tsx              # 总览
  embodied-dataset-by-embodiment.canvas.tsx      # 单臂/双臂/移动 + Mix
  embodied-dataset-sizes.canvas.tsx              # 体积对照
```

---

## 1. 怎么选（结论）

| 目标 | 优先数据 |
|------|----------|
| VLA 预训练 | OXE（广）+ DROID/Bridge（质）+ AgiBot / Galaxea / RoboMIND（长程·双臂·移动） |
| 评测 | LIBERO + SimplerEnv；长程看 CALVIN / RoboCasa |
| 格式 | OXE → RLDS；训练管线优先 **LeRobot**（HF 转换多） |
| 许可 | AgiBot、Galaxea 多为 **CC BY-NC-SA**（非商用）；OXE 各子集独立 |

落地顺序：定格式与 camera/action 约定 → 只下 P0 跑通 → 加 P1 扩泛化 → 自有数据 fine-tune。双臂/移动勿与单臂 EE 动作硬拼，先统一 schema 或加 embodiment tag。

---

## 2. 数据集一览

| 数据集 | 年份 | 规模 | 本体 | 格式 | 用途 | 要点 | 链接 |
|--------|------|------|------|------|------|------|------|
| Open X-Embodiment (OXE) | 2023 | 1M+ traj / 60 源集 | 22 robots | RLDS | 预训练优先 | 跨本体预训练事实标准 | https://robotics-transformer-x.github.io/ |
| ARIO | 2024 | ~3M episodes | 多本体统一 | ARIO | 预训练优先 | 整合 OXE、RH20T 等 | https://imaei.github.io/project_pages/ario/ |
| AgiBot World (α/β/2026) | 2024–2026 | α 9.2万 / β ~100万 | G1/G2 双臂+灵巧手 | LeRobot | 预训练优先 | 长程双臂；GO-1 | https://agibot-world.com/ |
| RoboMIND 2.0 | 2025+ | 31万+ 双臂 | 6 embodiments | project | 预训练优先 | 触觉 12k + 移动 20k | https://log2r.github.io/RoboMIND2.0/ |
| Galaxea Open-World | 2025 | 500h+ / ~100k | R1-Lite | LeRobot / RLDS | 预训练优先 | 开放世界移动操作 | https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset |
| DROID | 2024 | 76k / 350h | Franka | RLDS + LeRobot | 预训练优先 | 564 scenes | https://droid-dataset.github.io/ |
| BridgeData V2 | 2023 | 60k | WidowX | raw + RLDS + LeRobot | 预训练优先 | 开放词汇多任务 | https://rail-berkeley.github.io/bridgedata/ |
| RT-1 / Fractal | 2022 | ~130k | Google Robot | RLDS + LeRobot | 预训练优先 | OXE 重要子集 | https://robotics-transformer.github.io/ |
| RH20T | 2023 | 110k+ | 多臂 | custom / ARIO | 补充 | 力/声多模态 | https://rh20t.github.io/ |
| RoboSet | 2023 | ~98k | Franka | project | 补充 | 厨房多技能 | https://robopen.github.io/ |
| BC-Z | 2022 | ~26k | 单臂 | Kaggle | 补充 | 早期 zero-shot | https://sites.google.com/view/bc-z/ |
| LIBERO | 2023 | 130 tasks / 4 suites | Franka sim | HDF5 / LeRobot | 评测 | VLA FT 默认榜 | https://libero-project.github.io/ |
| RoboCasa | 2024 | 100+ atomic tasks | mobile sim | LeRobot | 预训练/仿真 | 厨房可扩展 | https://robocasa.ai/ |
| CALVIN | 2022 | 长程语言链 | Franka sim | custom | 评测 | 长程经典 | http://calvin.cs.uni-freiburg.de/ |
| ManiSkill / 3 | 2021–2025 | GPU 并行大规模 | 多臂 sim | project | 补充 | 高速仿真 | https://www.maniskill.ai/ |
| BEHAVIOR-1K | 2024 | 1000 activities | mobile sim | assets | 评测 | 日常活动广度 | https://behavior.stanford.edu/ |
| SimplerEnv | 2024 | eval suite | WidowX / Google sim | eval | 评测 | 与真机 OXE 相关较好 | https://simpler-env.github.io/ |
| Ego4D / EPIC | 2020– | 数千小时 | 人类 | video | 辅训 | 无机器人动作标签 | https://ego4d-data.org/ |

### 按类别

- **聚合**：OXE、ARIO  
- **真机桌面**：DROID、Bridge、Fractal、RoboSet、BC-Z、RH20T  
- **长程 / 双臂 / 移动**：AgiBot、Galaxea、RoboMIND  
- **仿真评测**：LIBERO、SimplerEnv、RoboCasa、CALVIN、ManiSkill、BEHAVIOR-1K  
- **人类视频**：Ego4D / EPIC（视觉先验）

---

## 3. 按本体：下载清单 + Mix

### 3.1 单臂

**P0**：BridgeData V2（`bridge_orig`）、DROID、RT-1 Fractal  
**P1**：OXE Magic Soup 其余子集、RH20T  
**P2**：LIBERO + SimplerEnv  

| 阶段 | 数据 | 相对权重 | 说明 |
|------|------|----------|------|
| 预训练主 mix | bridge_orig | 1.0（锚点） | 用官网版，勿用过期 OXE bridge |
| 预训练主 mix | fractal20220817_data | ~0.54 | OpenVLA Magic Soup++ |
| 预训练主 mix | droid | 0.06–0.10 | 后期可降权或去掉 |
| 预训练主 mix | OXE 其余过滤子集 | 见 mixtures.py | 单臂 + 第三人称 + EE |
| 微调 | 自有 demos | 100% 或与 Bridge 9:1 | — |

下载：

- Bridge: https://rail-berkeley.github.io/bridgedata/ · HF `nvidia/BridgeData2_LeRobot_v3`
- DROID: https://droid-dataset.github.io/ · HF `lerobot/droid_1.0.1`
- Fractal: OXE GCS · HF `IPEC-COMMUNITY/fractal20220817_data_lerobot`

### 3.2 双臂

**P0**：AgiBot Alpha → Beta/2026、RoboMIND 2.0  
**P1**：OXE 单臂（Stage1 先验）  
**P2**：RoboTwin / LIBERO(sim)  

| 阶段 | 数据 | 相对权重 | 说明 |
|------|------|----------|------|
| Stage1 跨本体 | OXE 单臂 (+Bridge/DROID) | Magic Soup | 动作维 pad/掩码 |
| Stage2 双臂专训 | AgiBot / RoboMIND | ≈ 0.7 / 0.3 | 先统一 action schema |
| Stage3 任务微调 | 目标任务 | 100% 或 8:2 | 保留子任务语言 |

下载：

- Alpha: https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha  
- 2026: https://huggingface.co/datasets/agibot-world/AgiBotWorld2026  
- RoboMIND: https://log2r.github.io/RoboMIND2.0/

### 3.3 移动

**P0**：Galaxea Open-World、RoboMIND mobile  
**P1**：AgiBot 移动底座子集、Fractal  
**P2**：RoboCasa / BEHAVIOR-1K  

| 阶段 | 数据 | 相对权重 | 说明 |
|------|------|----------|------|
| Stage1 操作先验 | Bridge + Fractal (+DROID) | 1.0 / 0.5 / 0.05 | 先学抓放 |
| Stage2 移动主训 | Galaxea | 0.8 | 单本体主数据 |
| Stage2 移动主训 | RoboMIND mobile + AgiBot | 0.2 | 补长尾 |
| Stage3 微调 | 目标场地 | 100% 或 9:1 | 可用 RoboCasa 增广 |

下载：https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset（按 task tar）

---

## 4. 体积（约数）

同一语料往往：**raw ≫ RLDS ≫ LeRobot（视频压缩）**。训练优先下 LeRobot。

| 数据集 | 约体积 | 来源/备注 |
|--------|--------|-----------|
| LIBERO (LeRobot) | ~2 GB | HF `lerobot/libero` |
| DROID-100 样例 | ~2 GB | 跑通管线 |
| AgiBot sample | ~7 GB | 结构预览 |
| Fractal (LeRobot) | ~22 GB | HF IPEC |
| Bridge V2 (LeRobot) | ~91 GB | HF nvidia；原图可达 ~1 TB |
| DROID (LeRobot) | ~412 GB | HF `lerobot/droid_1.0.1` |
| DROID (RLDS) | ~1.7 TB | 官方；raw ~8.7 TB |
| OXE HF 镜像 | ~1.1 TB | 非全集；全集多 TB |
| AgiBot α 社区 LeRobot | ~1.8 TB | 非官方转换 |
| Galaxea | ~2.9 TB | 227 task archives；HF used 有报更大 |
| RH20T 640 RGB | ~5 TB | +depth ~10 TB；原分辨率 ~40 TB |
| AgiBot Alpha 官方 | ~8.5 TB | OpenDriveLab README |
| RoboMIND (HF v1) | ~12 TB | v2.0 预计更大 |
| AgiBot World 2026 | ~14 TB | HF Total file size |
| AgiBot Beta | ~44 TB | README ~43.8 TB |

### 磁盘预算

| 场景 | 建议 |
|------|------|
| 试跑 | ≤ 100 GB（样例 + Bridge 或 Fractal） |
| 单臂 P0（LeRobot） | Bridge+Fractal+DROID ≈ **0.5 TB** |
| 双臂先跑通 | Alpha **~8.5 TB**；Beta 需 **~50 TB** 级 |
| 移动 | Galaxea **~3 TB**，可按 task 分批 |
| OXE | 按 `mixtures.py` 子集按需拉，勿一次拉满 |

---

## 5. 格式与许可提醒

- **RLDS**：OXE 官方，GCS/TFDS，偏 TF/Jax  
- **LeRobot**：Parquet + MP4 + meta；DROID、Bridge、AgiBot、Galaxea、RoboCasa 等均有  
- **社区转换**：IPEC-COMMUNITY / any4lerobot 等  

许可：AgiBot、Galaxea 为 NC；RH20T 含人脸/语音；正式训练前核对各 README。

权重参考：OpenVLA `prismatic/vla/datasets/rlds/oxe/mixtures.py`（Magic Soup++）、Galaxea G0 三段 curriculum。
