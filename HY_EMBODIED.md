# Hy-Embodied 接入说明

在 `show_camera_topics.py` 的 **AI 对话** 面板中可选用：

| 预设 | 模型 | 默认 API | 启动脚本 |
|------|------|----------|----------|
| Hy-Embodied-VLM-1.0 | `hy_a3b` (~30B MoE / ~3B 激活) | `http://127.0.0.1:8080/v1` | `run_hy_embodied_vlm.sh` |
| Hy-Embodied-RxBrain-1.0 | `hy-rxbrain` (~6.2B) | `http://127.0.0.1:8090/v1` | `run_hy_rxbrain.sh` |

官方链接：

- VLM: [GitHub HY-Embodied](https://github.com/Tencent-Hunyuan/HY-Embodied) · [HF Hy-Embodied-VLM-1.0](https://huggingface.co/tencent/Hy-Embodied-VLM-1.0)
- RxBrain: [GitHub Hy-Embodied-RxBrain-1.0](https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0) · [HF Hy-Embodied-RxBrain-1.0](https://huggingface.co/tencent/Hy-Embodied-RxBrain-1.0)

## 在 Viewer 中使用

1. 用脚本在**宿主机**启动对应服务（Docker 内 viewer 请加 `--host 0.0.0.0`，API 填宿主机 IP）。
2. 对话面板选择对应预设，Key 填 `EMPTY`。
3. 勾选 **附带相机图**（多模态必须），可选 **thinking**（仅 VLM）。
4. 发送关于当前相机画面的问题。

## Hy-Embodied-VLM-1.0（推荐 vLLM）

硬件：约 4×80GB GPU（`TP=4`），权重 BF16 ~86GB。

```bash
bash run_hy_embodied_vlm.sh --clone
bash run_hy_embodied_vlm.sh --install   # 需 uv
TP=4 bash run_hy_embodied_vlm.sh
```

环境变量：`MODEL_PATH`、`TP`、`PORT`、`HY_EMBODIED_VLM_API_BASE`。

## Hy-Embodied-RxBrain-1.0（本仓库 worker，VQA）

当前接入为 **理解/VQA**（图像问答）。文生图 / 多帧想象 / 交错规划需官方脚本 + FLUX VAE，未接到 UI。

```bash
bash run_hy_rxbrain.sh --clone
bash run_hy_rxbrain.sh --download
bash run_hy_rxbrain.sh --install
bash run_hy_rxbrain.sh
# Docker viewer:
# bash run_hy_rxbrain.sh --host 0.0.0.0
```

环境变量：`HY_RXBRAIN_REPO`、`HY_RXBRAIN_CKPT`、`HY_RXBRAIN_API_BASE`。

## 说明

- 权重与仓库默认落在 `eai/third_party/`、`eai/weights/`，体积很大，勿提交 git。
- Viewer 只做 OpenAI 兼容客户端；推理在独立服务进程中运行。
