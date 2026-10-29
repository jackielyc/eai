# EAI 机器人开发界面

也可从本目录运行 `python -m eai_app`。启动方式保持兼容：继续使用原来的启动脚本或 `show_camera_topics.py`。实现代码现集中在 `eai_app/`；根目录同名 Python 文件是兼容入口，请修改其文档字符串指向的实现文件。

- [代码结构、依赖边界和热更新](docs/architecture.md)
- [分阶段抓取](docs/grasp_execution.md)
- [手机盒装配](docs/phone_assembly.md)

界面启动后，通过 **开发 → 空闲时自动应用代码更改** 开启/关闭代码监视，或按 **Ctrl+Shift+R** 检查并应用。兼容的方法体修改可直接作用于现有按钮回调；结构性修改会明确提示重启。接入这套新目录和更新机制需要先重启一次旧界面。

离线回归：`bash scripts/run_offline_tests.sh`。脚本分别使用已有的 UI/ROS Python 和规划/仿真环境，不控制真机。
