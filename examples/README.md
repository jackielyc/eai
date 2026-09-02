# 远程 API 调用示例 — 使用说明

独立示例，对齐仓库 `show_camera_topics.py` 中的 `LlmChatClient`：用标准库 `urllib` 调用 **OpenAI 兼容** `POST {api_base}/chat/completions`，无需 `requests` / `openai` SDK。

路径：`eai/examples/`。

## 文件

| 文件 | 作用 |
|------|------|
| `remote_api_chat_example.py` | 可运行客户端 + CLI |
| `README.md` | 本说明 |

## 依赖

- Python 3.10+（仅标准库即可跑通文本对话）
- 可选：`opencv-python` / `opencv-python-headless`（`--image` 时自动缩放 JPEG）

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLM_API_BASE` | `https://api.openai.com/v1` | 必须以 `/v1` 结尾（与 Viewer 一致） |
| `LLM_MODEL` | `gpt-4o-mini` | 模型名 |
| `LLM_API_KEY` | （空） | 云端填真实 Key；本地服务填 `EMPTY` 或 `ollama` |
| `LLM_MAX_TOKENS` | `1536` | 最大生成长度 |
| `LLM_ENABLE_THINKING` | 关 | `1/true` 开启 thinking |
| `LLM_SYSTEM_PROMPT` | 简短助手提示 | system 角色 |
| `LLM_MAX_RETRIES` | `2` | 网络 / 5xx / 429 重试次数 |
| `LLM_CHAT_TIMEOUT_S` | `120` | 纯文本超时（秒） |
| `LLM_CHAT_VISION_TIMEOUT_S` | `300` | 带图超时（秒） |

认证头统一为：

```http
Authorization: Bearer <key或EMPTY>
Content-Type: application/json
```

## 内置预设（`--preset`）

| 预设 | API Base | Model | Key |
|------|----------|-------|-----|
| `local-qwen` | `http://127.0.0.1:8100/v1` | `qwen3.5-4b` | `EMPTY` |
| `remote-qwen` | `http://127.0.0.1:18100/v1` | `qwen3.5-35b-a3b` | `EMPTY` |
| `ollama` | `http://127.0.0.1:11434/v1` | `qwen2.5` | `ollama` |
| `hy-vlm` | `http://127.0.0.1:8080/v1` | `hy_a3b` | `EMPTY` |
| `hy-rxbrain` | `http://127.0.0.1:8090/v1` | `hy-rxbrain` | `EMPTY` |
| `dashscope` | 百炼兼容模式 | `qwen-plus` | 需真实 Key |
| `openai` | OpenAI 官方 | `gpt-4o-mini` | 需真实 Key |

`local-qwen` / `remote-qwen` / Hy 系列的 base 也可被对应环境变量覆盖（与主仓库相同）。

## 快速开始

```bash
cd /share_data/projects/mahjong/share/personal/liyichao/eai/examples

# 1) 先探测服务是否可达
python3 remote_api_chat_example.py --preset ollama --probe

# 2) 纯文本问答
python3 remote_api_chat_example.py --preset ollama "用一句话解释 ROS2 topic"

# 3) 多模态（附带图片）
python3 remote_api_chat_example.py --preset local-qwen \
  --image /path/to/frame.jpg "描述画面中的物体"

# 4) 云端百炼
export LLM_API_KEY=sk-xxxx
python3 remote_api_chat_example.py --preset dashscope "你好"
```

### 在代码中复用

```python
from remote_api_chat_example import LlmChatClient, LlmChatConfig

cfg = LlmChatConfig.from_env(preset="local-qwen")
# 或手动：
# cfg = LlmChatConfig(
#     api_base="http://127.0.0.1:8100/v1",
#     model="qwen3.5-4b",
#     api_key="EMPTY",
#     max_retries=2,
# )
client = LlmChatClient(cfg)

# 探测
print(client.probe())

# 单轮
print(client.ask("你好"))

# 带图
print(client.ask("图里有什么？", image_path="/tmp/frame.jpg"))

# 自定义多轮 messages（与 Viewer 相同结构）
reply = client.chat(
    [
        {"role": "system", "content": "你是机器人助手"},
        {"role": "user", "content": "当前任务是什么？"},
    ]
)
print(reply)
```

## 稳定调用建议

1. **先 `--probe` 再 chat**  
   检查 `/health`、`/models`（或 Ollama `/api/tags`），确认 `configured_model` 在列表中。

2. **超时按场景分开**  
   文本默认 120s；带图默认 300s。大模型 / 慢链路用 `--timeout 600` 或环境变量放宽。

3. **自动重试**  
   默认对「网络错误 / HTTP 5xx / 429」重试 2 次（指数退避）。业务 4xx（如 Key 错、模型名错）不重试。可用 `--retries 0` 关闭。

4. **Key 约定**  
   - 本地 worker（Qwen / Hy-Embodied）：`EMPTY`  
   - Ollama：`ollama`  
   - 云端：真实 Key，不要提交进 git

5. **关闭无用 thinking**  
   Qwen3.5 等默认可能进入长思考，挤占 `max_tokens`。不需要时不要加 `--thinking`，客户端会显式传 `enable_thinking: false`。

6. **图片体积**  
   有 OpenCV 时会把长边缩到 ≤1280 再 JPEG；避免原图过大导致超时或 413。

7. **远程 Qwen**  
   Viewer / Docker 场景需先在宿主机拉起 tunnel（`remote_qwen_ctl` / hostctl），本示例只访问本机映射端口（默认 `18100`），不负责 SSH。

8. **api_base 格式**  
   必须是 `http(s)://host:port/v1`，脚本会拼 `/chat/completions`。不要多写一层 `/v1`。

## 与主仓库的关系

| 能力 | 本示例 | `show_camera_topics.py` |
|------|--------|-------------------------|
| Chat Completions | ✅ | ✅ `LlmChatClient` |
| 探测 `/health` `/models` | ✅ | ✅ `probe_llm_endpoint` |
| 相机帧自动附带 | ❌（用 `--image`） | ✅ |
| Qt 对话面板 / 历史 | ❌ | ✅ |
| SAM3 / FoundationPose HTTP | ❌ | ✅ |

需要 GUI 或相机流时仍用主 Viewer；本示例适合脚本、联调、CI 探测与二次开发。

## CLI 参数摘要

```text
python3 remote_api_chat_example.py [prompt]
  --preset {local-qwen,remote-qwen,ollama,hy-vlm,hy-rxbrain,dashscope,openai}
  --api-base URL   --model NAME   --api-key KEY
  --image PATH     --system TEXT  --thinking
  --max-tokens N   --timeout SEC  --retries N
  --probe          --dump-request
```
