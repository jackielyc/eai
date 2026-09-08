# 测试工作室（test_studio）

**独立前端窗口**（不是相机 Viewer 里的 Tab）。界面全新；推理启停 / 模型扫描 / 对话组件复用 `show_camera_topics` 与 `QwenDeployController`。

## 启动

```bash
bash test_studio/run_test_studio.sh
```

容器内也可：

```bash
python3 -m test_studio.main
```

主窗口 `run_in_docker.sh` **不会**再出现「测试工作室」Tab；原「测试」Tab 仍可用。

## 结构

| 文件 | 作用 |
|------|------|
| `run_test_studio.sh` | Docker 启动脚本 |
| `main.py` | 独立进程入口 |
| `window.py` | `TestStudioWindow` 主窗口 |
| `page.py` | 新 UI（`TestStudioPage`） |
| `deploy_controller.py` | 推理部署控制器（可多 View；Viewer「测试」Tab 与工作室各自独立进程各用一份） |
| `__init__.py` | 导出 |

## 复用关系

- **对话**：`ChatPanelWidget`（`show_camera_topics`）
- **启停 / 扫权重 / 上次配置**：`QwenDeployController` + `DeployView`
- **场景图路径 / 样式 / Combo IME**：从 `show_camera_topics` 导入
