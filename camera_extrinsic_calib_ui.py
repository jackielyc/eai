"""手臂页里的头部/腕部手工外参标定。"""
from __future__ import annotations

import time

import numpy as np
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import click_pose
from camera_extrinsic_calib import (
    CAMERAS,
    express_in_tcp,
    point_from_tcp,
    save_camera,
    solve_rigid,
    unproject,
)


def _spin(value, low, high, step, suffix):
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(2)
    box.setSingleStep(step)
    box.setValue(value)
    box.setSuffix(suffix)
    return box


class CameraExtrinsicCalibPanel(QWidget):
    def __init__(self, node, get_click, get_tcp, parent=None):
        super().__init__(parent)
        self.node = node
        self.get_click = get_click
        self.get_tcp = get_tcp
        self.samples = []
        self.marker_base = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        title = QLabel("手工手眼标定")
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        layout.addWidget(title)
        hint = QLabel(
            "在相机画面上点击目标点，再记录。头部：点击被标定臂的 TCP 所在位置，求 T_base_camera。"
            "腕部：先用一只臂点住桌上的固定标记，再移动戴着相机的那只臂，从多个视角点击同一标记，求 T_tcp_camera。"
            "三路图像的 frame_id 都是 camera_frame，结果按话题分别保存。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("相机"))
        self.camera = QComboBox()
        for item in CAMERAS:
            self.camera.addItem(item["label"] + "  " + item["topic"], item)
        self.camera.currentIndexChanged.connect(self._camera_changed)
        row.addWidget(self.camera, 1)
        row.addWidget(QLabel("探测臂"))
        self.probe = QComboBox()
        self.probe.addItem("左臂 TCP", "left")
        self.probe.addItem("右臂 TCP", "right")
        row.addWidget(self.probe)
        layout.addLayout(row)

        krow = QHBoxLayout()
        self.fx = _spin(0, 0, 5000, 10, "")
        self.fy = _spin(0, 0, 5000, 10, "")
        self.cx = _spin(0, 0, 4000, 10, "")
        self.cy = _spin(0, 0, 4000, 10, "")
        for label, box in (("fx", self.fx), ("fy", self.fy), ("cx", self.cx), ("cy", self.cy)):
            krow.addWidget(QLabel(label))
            box.setDecimals(4)
            box.setToolTip("原图内参，单位像素。切换相机会填入机器人上读到的彩色内参。")
            krow.addWidget(box)
        self.center_button = QPushButton("填图像中心")
        self.center_button.clicked.connect(self.fill_center)
        krow.addWidget(self.center_button)
        layout.addLayout(krow)

        orow = QHBoxLayout()
        orow.addWidget(QLabel("点击点相对 TCP（cm，TCP 坐标系）"))
        self.ox = _spin(0, -30, 30, 0.5, "")
        self.oy = _spin(0, -30, 30, 0.5, "")
        self.oz = _spin(0, -30, 30, 0.5, "")
        for label, box in (("X", self.ox), ("Y", self.oy), ("Z", self.oz)):
            orow.addWidget(QLabel(label))
            orow.addWidget(box)
        layout.addLayout(orow)

        brow = QHBoxLayout()
        self.marker_button = QPushButton("记下固定标记（当前探测臂 TCP）")
        self.marker_button.clicked.connect(self.store_marker)
        brow.addWidget(self.marker_button)
        self.record_button = QPushButton("记录当前点击")
        self.record_button.clicked.connect(self.record)
        brow.addWidget(self.record_button)
        self.undo_button = QPushButton("删除上一点")
        self.undo_button.clicked.connect(self.undo)
        brow.addWidget(self.undo_button)
        self.clear_button = QPushButton("清空")
        self.clear_button.clicked.connect(self.clear)
        brow.addWidget(self.clear_button)
        self.solve_button = QPushButton("求解并保存")
        self.solve_button.clicked.connect(self.solve)
        brow.addWidget(self.solve_button)
        layout.addLayout(brow)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        layout.addWidget(self.log)
        self._camera_changed()

    def _spec(self):
        return self.camera.currentData()

    def _camera_changed(self):
        self.samples = []
        self.marker_base = None
        spec = self._spec()
        wrist = spec["model"] == "eye_in_hand"
        self.marker_button.setEnabled(wrist)
        self.fx.setValue(float(spec["fx"]))
        self.fy.setValue(float(spec["fy"]))
        self.cx.setValue(float(spec["cx"]))
        self.cy.setValue(float(spec["cy"]))
        self.log.setPlainText(self._instructions())

    def _instructions(self) -> str:
        spec = self._spec()
        if spec["model"] == "eye_to_hand":
            return (
                "头部标定步骤：\n"
                "1. 勾选头部彩色和深度。fx/fy/cx/cy 已填入机器人上的头部彩色内参。\n"
                "2. 把探测臂 TCP 移到头部相机能看见的位置。若点的是指尖而不是 Link7，填写相对 TCP 的偏移。\n"
                "3. 在头部画面上点击该点，再点「记录当前点击」。换位置重复，至少 4 个、彼此拉开。\n"
                "4. 求解得到 T_base_camera。腰或头再动之后要重标。\n"
            )
        return (
            f"{spec['label']}标定步骤：\n"
            "1. 在桌上放一个固定标记。用探测臂 TCP 碰住它，点「记下固定标记」。\n"
            f"2. 移动{spec['label']}所在的手臂，让该相机从不同角度看这个标记。\n"
            "3. 每次在该腕部画面点击标记，再点「记录当前点击」。至少 4 个视角。\n"
            "4. 求解得到 T_tcp_camera。使用时会乘上当时的 TCP，得到 T_base_camera。\n"
        )

    def _k(self):
        return (self.fx.value(), self.fy.value(), self.cx.value(), self.cy.value())

    def _offset_m(self):
        return (self.ox.value() / 100.0, self.oy.value() / 100.0, self.oz.value() / 100.0)

    def fill_center(self):
        click = self.get_click() or {}
        hw = click.get("preview_hw")
        full = self._full_size(click.get("topic") or "")
        h, w = full or hw or (0, 0)
        if w <= 0:
            self._say("还没有该相机画面，无法填中心。")
            return
        self.cx.setValue(w / 2.0)
        self.cy.setValue(h / 2.0)
        self._say(f"已填 cx={w/2:.1f}, cy={h/2:.1f}。fx/fy 仍需填写真实内参。")

    def _full_size(self, topic):
        frames = getattr(self.node, "click_frames", None)
        if frames is None or not topic:
            return None
        with frames.lock:
            bucket = list(frames.frames.get(topic, ()))
        if not bucket:
            return None
        image = bucket[-1].image
        return image.shape[:2]

    def _say(self, text):
        self.log.append(text)

    def store_marker(self):
        tcp = self.get_tcp(self.probe.currentData())
        if tcp is None:
            self._say("没有探测臂 TCP。请先连接机器人并确认 host_fk 在更新。")
            return
        self.marker_base = point_from_tcp(tcp["xyz"], tcp["xyzw"], self._offset_m())
        p = self.marker_base
        self._say(f"固定标记 base XYZ=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m")

    def note_click(self, topic: str, u: int, v: int) -> None:
        spec = self._spec()
        if topic == spec["topic"]:
            self._say(f"收到 {spec['label']} 点击 ({int(u)}, {int(v)})，确认后点「记录当前点击」。")

    def _pixel_and_depth(self, topic):
        click = self.get_click() or {}
        if click.get("topic") != topic:
            raise ValueError(f"最近一次点击在 {click.get('topic') or '无'}，请先点击 {topic} 的画面")
        frames = self.node.click_frames
        with frames.lock:
            colors = list(frames.frames.get(topic, ()))
        if not colors:
            raise ValueError("没有该彩色图的原始帧，请先勾选并等待画面")
        color = colors[-1]
        ph, pw = (click.get("preview_hw") or color.image.shape[:2])[:2]
        h, w = color.image.shape[:2]
        u = int((float(click["u"]) + 0.5) * w / max(1, pw))
        v = int((float(click["v"]) + 0.5) * h / max(1, ph))
        u = min(w - 1, max(0, u))
        v = min(h - 1, max(0, v))
        depth_topic = topic.replace("_color", "_depth")
        with frames.lock:
            depths = list(frames.frames.get(depth_topic, ()))
            if not depths:
                depths = list(frames.frames.get(depth_topic + "_z", ()))
                if depths:
                    depth_topic = depth_topic + "_z"
        if not depths:
            if "/camera/hand_" in depth_topic:
                raise ValueError(
                    f"{depth_topic} 没有深度帧。话题已经在列表里，界面也会跟着彩色图订阅，"
                    "但腕部 D405 的 depthEnable 当前是关的，勾选也不会出图。"
                    "需要在机器人相机配置里打开 hand_left / hand_right 的 depthEnable，并重启相机后再记录。"
                )
            raise ValueError(f"没有 {depth_topic} 的深度帧。请确认该深度话题正在出图后再记录")
        depth = depths[-1]
        try:
            metric = click_pose.metric_depth(depth, (h, w))
        except ValueError as exc:
            image = np.asarray(depth.image)
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[:, :, 0]
            if image.shape[:2] != (h, w):
                raise ValueError("深度与彩色分辨率不同，不能把点击像素直接对应到深度") from exc
            if image.dtype == np.uint16:
                metric = image.astype(np.float32) / 1000.0
            elif np.issubdtype(image.dtype, np.floating):
                metric = image.astype(np.float32)
            else:
                raise ValueError(f"不支持的深度编码/类型：{depth.encoding}/{image.dtype}") from exc
        patch = metric[max(0, v - 2) : v + 3, max(0, u - 2) : u + 3]
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 5.0)]
        if valid.size == 0:
            raise ValueError(f"像素 ({u},{v}) 没有有效深度")
        return u, v, float(np.median(valid)), str(color.frame_id or "")

    def record(self):
        spec = self._spec()
        try:
            if min(self.fx.value(), self.fy.value()) <= 1:
                raise ValueError("请先填写 fx 和 fy")
            u, v, z, frame_id = self._pixel_and_depth(spec["topic"])
            camera_point = unproject(u, v, z, self._k())
            if spec["model"] == "eye_to_hand":
                tcp = self.get_tcp(self.probe.currentData())
                if tcp is None:
                    raise ValueError("没有探测臂 TCP")
                target = point_from_tcp(tcp["xyz"], tcp["xyzw"], self._offset_m())
                sample = {"camera": camera_point, "target": target, "uv": (u, v), "z": z}
            else:
                if self.marker_base is None:
                    raise ValueError("请先记下固定标记")
                tcp = self.get_tcp(spec["arm_side"])
                if tcp is None:
                    raise ValueError(f"没有{spec['label']} TCP")
                target = express_in_tcp(self.marker_base, tcp["xyz"], tcp["xyzw"])
                sample = {
                    "camera": camera_point,
                    "target": target,
                    "uv": (u, v),
                    "z": z,
                    "tcp_xyz": list(tcp["xyz"]),
                    "tcp_xyzw": list(tcp["xyzw"]),
                }
            sample["frame_id"] = frame_id
            self.samples.append(sample)
            self._say(
                f"#{len(self.samples)} uv=({u},{v}) depth={z:.3f} m  "
                f"相机点=({camera_point[0]:.3f},{camera_point[1]:.3f},{camera_point[2]:.3f})"
            )
            waist_age = None if tcp is None else tcp.get("waist_age_s")
            if waist_age is None or waist_age > 2.0:
                self._say("腰部反馈超过 2 秒未更新，TCP 的 base 坐标可能不准。")
        except Exception as exc:
            self._say("未记录：" + str(exc))

    def undo(self):
        if self.samples:
            self.samples.pop()
            self._say(f"剩余 {len(self.samples)} 点")

    def clear(self):
        self.samples = []
        self._say("样本已清空")

    def solve(self):
        spec = self._spec()
        try:
            if len(self.samples) < 4:
                raise ValueError("至少记录 4 个点")
            src = np.stack([item["camera"] for item in self.samples])
            dst = np.stack([item["target"] for item in self.samples])
            T, err = solve_rigid(src, dst)
            if float(np.max(err)) > 0.03:
                self._say(f"最大点误差 {np.max(err)*100:.1f} cm，结果已保存但请核对点击和内参。")
            entry = {
                "label": spec["label"],
                "model": spec["model"],
                "frame_id": self.samples[-1].get("frame_id") or "camera_frame",
                "base_frame": "base_link",
                "intrinsics": {"fx": self.fx.value(), "fy": self.fy.value(), "cx": self.cx.value(), "cy": self.cy.value()},
                "rms_m": float(np.sqrt(np.mean(err ** 2))),
                "max_m": float(np.max(err)),
                "samples": len(self.samples),
                "calibrated_at": time.time(),
            }
            if spec["model"] == "eye_to_hand":
                entry["T_base_camera"] = T.tolist()
                entry["note"] = "眼在手外。腰或头移动后此 T_base_camera 失效。"
            else:
                entry["arm_side"] = spec["arm_side"]
                entry["tcp_link"] = "Link7_l" if spec["arm_side"] == "left" else "Link7_r"
                entry["T_tcp_camera"] = T.tolist()
                entry["note"] = "眼在手上。T_base_camera = T_base_tcp @ T_tcp_camera。"
            path = save_camera(spec["topic"], entry)
            self._say(
                f"已保存 {spec['topic']}  RMS={entry['rms_m']*100:.1f} cm  最大={entry['max_m']*100:.1f} cm\n{path}"
            )
        except Exception as exc:
            self._say("求解失败：" + str(exc))
