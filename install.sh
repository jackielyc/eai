#!/usr/bin/env bash
# 为系统 Python 3.10 安装 pip 依赖（ROS2 Humble 专用）
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> 使用 Python 3.10 安装依赖..."
# opencv-python 自带 Qt 插件，会与 PyQt5 冲突，先卸载
/usr/bin/python3.10 -m pip uninstall -y opencv-python 2>/dev/null || true
/usr/bin/python3.10 -m pip install --user -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo ">>> 验证依赖..."
/usr/bin/python3.10 - <<'PY'
import numpy
import cv2
import pyqtgraph
from PyQt5.QtWidgets import QApplication

assert numpy.__version__.startswith("1."), f"numpy 须为 1.x，当前 {numpy.__version__}"
print(f"  numpy      {numpy.__version__}  OK")
print(f"  opencv     {cv2.__version__}  OK")
print(f"  PyQt5      OK")
print(f"  pyqtgraph  {pyqtgraph.__version__}  OK")
PY

if [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    set -e
    /usr/bin/python3.10 - <<'PY'
import rclpy
from cv_bridge import CvBridge
print(f"  rclpy      OK")
print(f"  cv_bridge  OK")
PY
else
    echo "  警告: 未找到 ROS2 Humble，跳过 rclpy/cv_bridge 检查"
fi

echo ""
echo "安装完成。运行: bash run.sh"
