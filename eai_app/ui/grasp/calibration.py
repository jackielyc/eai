"""Explicit operator-owned calibration for the executable grasp profile."""
from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eai_app.domain.grasp.execution import load_profile


RUIYAN_FACTORY_HAND_MODEL = "ruiyan_6dof_v1"
RUIYAN_FACTORY_OPEN_POSITIONS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# Ruiyan commands are normalized to [0, 1].  The shipped driver limits the
# two thumb motors to 0.9 while the four finger motors accept the full range.
RUIYAN_FACTORY_CLOSE_POSITIONS = (0.9, 0.9, 1.0, 1.0, 1.0, 1.0)


def _spin(low: float, high: float, value: float, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(2)
    box.setSingleStep(0.5)
    box.setValue(value)
    box.setSuffix(suffix)
    return box


class GraspProfileCalibrationPanel(QWidget):
    """Writes an executable profile only after the operator verifies measurements."""

    def __init__(self, profile_path: str, refresh_profile, parent=None):
        super().__init__(parent)
        self.default_path = profile_path
        self.refresh_profile = refresh_profile
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("抓取 TCP 与手型标定")
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        layout.addWidget(title)
        hint = QLabel(
            "在机器人静止、手臂与手已使能前完成测量。本页不发送运动命令。"
            "T_grasp_tcp 是从抓取接触坐标系到 Link7 的实测刚体变换；"
            "开合量按拇指旋转、拇指弯曲、食指、中指、无名指、小指排列。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("配置文件"))
        self.path = QLineEdit(profile_path)
        path_row.addWidget(self.path, 1)
        self.load_button = QPushButton("读取")
        self.load_button.clicked.connect(self.load)
        path_row.addWidget(self.load_button)
        layout.addLayout(path_row)

        form = QFormLayout()
        self.arm = QComboBox()
        self.arm.addItem("右臂 / Link7_r", "right")
        self.arm.addItem("左臂 / Link7_l", "left")
        form.addRow("执行侧", self.arm)
        self.hand_model = QLineEdit(RUIYAN_FACTORY_HAND_MODEL)
        form.addRow("实际手型标识", self.hand_model)
        self.ruiyan_factory_button = QPushButton("加载睿研出厂手型默认值")
        self.ruiyan_factory_button.setToolTip(
            "填入睿研六通道的出厂开合范围；不会改动机器人安装相关的 TCP 和工作空间。"
        )
        self.ruiyan_factory_button.clicked.connect(self.load_ruiyan_factory_defaults)
        form.addRow("出厂预设", self.ruiyan_factory_button)
        self.workspace = []
        grid = QGridLayout()
        for col, label in enumerate(("X", "Y", "Z")):
            grid.addWidget(QLabel(label), 0, col + 1)
        for row, label, initial in ((1, "最小 m", (0.0, -0.5, 0.4)), (2, "最大 m", (0.8, 0.5, 1.3))):
            grid.addWidget(QLabel(label), row, 0)
            values = []
            for col, value in enumerate(initial):
                box = _spin(-2.0, 2.0, value, " m")
                grid.addWidget(box, row, col + 1)
                values.append(box)
            self.workspace.append(values)
        form.addRow("实测 TCP 工作空间", grid)
        layout.addLayout(form)

        layout.addWidget(QLabel("接触坐标系 → Link7：平移（cm）和 RPY（度）"))
        self.tcp_translation = [_spin(-30, 30, 0, " cm") for _ in range(3)]
        self.tcp_rpy = [_spin(-180, 180, 0, "°") for _ in range(3)]
        tcp_grid = QGridLayout()
        for col, label in enumerate(("X", "Y", "Z")):
            tcp_grid.addWidget(QLabel(label), 0, col + 1)
            tcp_grid.addWidget(self.tcp_translation[col], 1, col + 1)
            tcp_grid.addWidget(self.tcp_rpy[col], 2, col + 1)
        tcp_grid.addWidget(QLabel("平移"), 1, 0)
        tcp_grid.addWidget(QLabel("RPY"), 2, 0)
        layout.addLayout(tcp_grid)

        hand_grid = QGridLayout()
        labels = ("拇指旋转", "拇指弯曲", "食指", "中指", "无名指", "小指")
        self.open_positions = []
        self.close_positions = []
        for col, label in enumerate(labels):
            hand_grid.addWidget(QLabel(label), 0, col + 1)
            opened = _spin(0, 1, 0, "")
            closed = _spin(0, 1, 0.5, "")
            opened.setDecimals(3)
            closed.setDecimals(3)
            hand_grid.addWidget(opened, 1, col + 1)
            hand_grid.addWidget(closed, 2, col + 1)
            self.open_positions.append(opened)
            self.close_positions.append(closed)
        hand_grid.addWidget(QLabel("张开"), 1, 0)
        hand_grid.addWidget(QLabel("闭合"), 2, 0)
        layout.addWidget(QLabel("实测六通道开合量"))
        layout.addLayout(hand_grid)

        self.tcp_checked = QCheckBox("我已用实物测量并复核接触点到 Link7 的平移和方向")
        self.hand_checked = QCheckBox("我已在空载、低风险状态下验证张开与闭合的六通道数值")
        self.workspace_checked = QCheckBox("我已测量工作空间，且确认所填范围不包含人体、桌面或障碍物")
        for box in (self.tcp_checked, self.hand_checked, self.workspace_checked):
            box.toggled.connect(self._update_save)
            layout.addWidget(box)
        self.save_button = QPushButton("确认实测并启用抓取配置")
        self.save_button.clicked.connect(self.save)
        layout.addWidget(self.save_button)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        layout.addStretch(1)
        self.load()

    def _update_save(self):
        self.save_button.setEnabled(
            self.tcp_checked.isChecked()
            and self.hand_checked.isChecked()
            and self.workspace_checked.isChecked()
        )

    def load_ruiyan_factory_defaults(self):
        """Fill the hand fields backed by the shipped Ruiyan driver limits."""
        self.hand_model.setText(RUIYAN_FACTORY_HAND_MODEL)
        for box, value in zip(self.open_positions, RUIYAN_FACTORY_OPEN_POSITIONS):
            box.setValue(value)
        for box, value in zip(self.close_positions, RUIYAN_FACTORY_CLOSE_POSITIONS):
            box.setValue(value)
        self.hand_checked.setChecked(False)
        self.status.setText(
            "已加载睿研出厂手型默认值：0 为全张，拇指两通道最大 0.9，"
            "其余四指最大 1.0。TCP 和工作空间属于整机安装参数，未被改动；"
            "请在低风险空载状态复核后再确认并保存。"
        )
        self._update_save()

    def load(self):
        try:
            path = Path(self.path.text()).expanduser()
            profile = load_profile(path, require_calibration=False)
            side = str(profile.get("arm_side") or "right")
            self.arm.setCurrentIndex(0 if side == "right" else 1)
            self.hand_model.setText(str(profile.get("hand_model") or RUIYAN_FACTORY_HAND_MODEL))
            for row, values in zip(self.workspace, profile.get("workspace_m") or []):
                for box, value in zip(row, values):
                    box.setValue(float(value))
            transform = profile.get("T_grasp_tcp") or []
            if len(transform) == 4:
                for box, value in zip(self.tcp_translation, [transform[0][3], transform[1][3], transform[2][3]]):
                    box.setValue(float(value) * 100)
            for boxes, values in ((self.open_positions, profile.get("open_positions") or []),
                                  (self.close_positions, profile.get("close_positions") or [])):
                for box, value in zip(boxes, values):
                    box.setValue(float(value))
            self.status.setText(
                "已读取配置。当前标定状态：" + ("已标定" if profile.get("calibrated") else "未标定")
            )
        except Exception as exc:
            self.status.setText("无法读取配置：" + str(exc))
        self._update_save()

    def save(self):
        try:
            import math
            import numpy as np
            path = Path(self.path.text()).expanduser()
            base = load_profile(path, require_calibration=False)
            r, p, y = [math.radians(box.value()) for box in self.tcp_rpy]
            cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
            rotation = np.array([
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ])
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = [box.value() / 100 for box in self.tcp_translation]
            side = str(self.arm.currentData())
            workspace = [[box.value() for box in row] for row in self.workspace]
            if any(workspace[0][i] >= workspace[1][i] for i in range(3)):
                raise ValueError("每个工作空间最小值必须小于最大值")
            base.update(
                calibrated=True,
                arm_side=side,
                hand_model=self.hand_model.text().strip(),
                tcp_link="Link7_r" if side == "right" else "Link7_l",
                T_grasp_tcp=transform.tolist(),
                workspace_m=workspace,
                open_positions=[box.value() for box in self.open_positions],
                close_positions=[box.value() for box in self.close_positions],
                calibration_pending=[],
            )
            if not base["hand_model"]:
                raise ValueError("请填写实际手型标识")
            path.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.status.setText("已写入实测抓取配置。回到“分阶段抓取”点击“检查准备状态”。")
            if self.refresh_profile:
                self.refresh_profile(str(path))
        except Exception as exc:
            self.status.setText("未保存：" + str(exc))
