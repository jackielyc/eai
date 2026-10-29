# EAI 机器人开发界面

从本目录运行 `python -m eai_app`，或使用主入口 `bash run.sh`。宿主机启动用 `bash scripts/launch/run_local.sh`，Docker 启动用 `bash scripts/launch/run_in_docker.sh`。

| 目录 | 内容 |
| --- | --- |
| `eai_app/` | 分层实现代码 |
| `compat/` | 旧 Python 模块转发入口 |
| `scripts/launch/` | 界面、模型服务与仿真启动脚本 |
| `scripts/setup/` | 环境安装和资源下载 |
| `scripts/maintenance/` | 系统维护脚本 |
| `scripts/` | 设备运维、执行与回归脚本 |
| `config/` | 标定、任务、界面偏好和本地榜单 |
| `requirements/` | 抓取 IK、网格、分割等可选依赖清单 |
| `tests/`、`docs/`、`tools/` | 测试、文档和命令行工具 |

根目录保留主依赖清单 `requirements.txt`。旧入口的文件路径已迁移，完整对应表见 [迁移清单](docs/root-layout-migration.json)。外部脚本请相应更新路径。旧模块导入方式见 [兼容说明](compat/README.md)。

- [代码结构、依赖边界和热更新](docs/architecture.md)
- [分阶段抓取](docs/grasp_execution.md)
- [手机盒装配](docs/phone_assembly.md)

界面启动后，通过 **开发 → 空闲时自动应用代码更改** 开启/关闭代码监视，或按 **Ctrl+Shift+R** 检查并应用。兼容的方法体修改可直接作用于现有按钮回调；结构性修改会明确提示重启。接入这套新目录和更新机制需要先重启一次旧界面。

离线回归：`bash scripts/run_offline_tests.sh`。脚本分别使用已有的 UI/ROS Python 和规划/仿真环境，不控制真机。
