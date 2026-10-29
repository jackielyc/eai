#!/usr/bin/python3.10
"""把头、左腕、右腕深度画成 RViz 点云。

头部用已标定的 T_base_camera。腕部标定误差是几十厘米，点云改挂在 Link7 上，不用那组外参。
"""
from __future__ import annotations

from eai_app.paths import PROJECT_ROOT

import json
import math
import os
import time

import numpy as np
from eai_app.perception.geometry import calibrated_intrinsics, DEPTH_MIN_M, DEPTH_MAX_M, valid_depth_mask, validated_camera_transform
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster

_EXTRINSICS = os.path.join(
    str(PROJECT_ROOT),
    "config",
    "camera_extrinsics.json",
)

_CAMERAS = (
    {
        "depth": "/camera/head_depth",
        "color": "/camera/head_color",
        "cloud": "/eai/depth_cloud/head",
        "frame": "eai_head_depth",
        "fallback_parent": "base_link",
        "rgb": (80, 220, 220),
    },
    {
        "depth": "/camera/hand_left_depth",
        "color": "/camera/hand_left_color",
        "cloud": "/eai/depth_cloud/hand_left",
        "frame": "eai_hand_left_depth",
        "fallback_parent": "Link7_l",
        "rgb": (70, 140, 230),
    },
    {
        "depth": "/camera/hand_right_depth",
        "color": "/camera/hand_right_color",
        "cloud": "/eai/depth_cloud/hand_right",
        "frame": "eai_hand_right_depth",
        "fallback_parent": "Link7_r",
        "rgb": (230, 140, 60),
    },
)

_STRIDE = 6
_MIN_M = DEPTH_MIN_M
_MAX_M = DEPTH_MAX_M
_MIN_PERIOD = 0.12
_MOSAIC_TOPIC = "/eai/color_mosaic"
_MOSAIC_W = 640


def _quat_from_R(R: np.ndarray):
    t = float(R[0, 0] + R[1, 1] + R[2, 2])
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return qx / n, qy / n, qz / n, qw / n


def _load_extrinsics() -> dict:
    try:
        with open(_EXTRINSICS, "r", encoding="utf-8") as handle:
            return {topic: dict(entry, color_topic=topic) for topic, entry in (json.load(handle).get("cameras") or {}).items()}
    except Exception:
        return {}


def _placement(cam: dict, spec: dict):
    """返回 parent, xyz, quat。外参不可用时挂到 fallback，平移和旋转都是单位。"""
    parent = spec["fallback_parent"]
    xyz = (0.0, 0.0, 0.0)
    quat = (0.0, 0.0, 0.0, 1.0)
    T, reason = validated_camera_transform(cam)
    if T is None:
        link = cam.get("tcp_link") or parent
        return link, xyz, quat, f"未标定示意：{reason}，挂在 {link}"
    model = cam.get("model")
    trans = T[:3, 3]
    parent = cam.get("base_frame") or "base_link"
    if model == "eye_in_hand":
        parent = cam.get("tcp_link") or parent
    return parent, tuple(float(v) for v in trans), _quat_from_R(T[:3, :3]), "使用外参 %s → %s" % (parent, spec["frame"])


def _intrinsics(cam: dict, width: int, height: int):
    k = calibrated_intrinsics(cam, width, height)
    if k is not None:
        return k
    return 0.9 * max(width, height), 0.9 * max(width, height), width / 2., height / 2.


def _depth_meters(msg: Image) -> np.ndarray:
    if msg.encoding in ("16UC1", "mono16"):
        depth = np.frombuffer(msg.data, dtype=np.uint16)
        if depth.size != msg.height * msg.width:
            raise ValueError("depth size %s != %sx%s" % (depth.size, msg.width, msg.height))
        return depth.reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    if msg.encoding == "32FC1":
        depth = np.frombuffer(msg.data, dtype=np.float32)
        if depth.size != msg.height * msg.width:
            raise ValueError("depth size mismatch")
        return depth.reshape(msg.height, msg.width)
    raise ValueError("unsupported depth encoding %s" % msg.encoding)


def _bgr(msg: Image):
    if msg.encoding not in ("bgr8", "rgb8") or msg.height <= 0 or msg.width <= 0:
        return None
    row = msg.step if msg.step else msg.width * 3
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if arr.size < msg.height * row:
        return None
    img = arr.reshape(msg.height, row)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def _fit_width(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] == width:
        return img
    scale = width / float(img.shape[1])
    height = max(1, int(round(img.shape[0] * scale)))
    ys = (np.linspace(0, img.shape[0] - 1, height)).astype(np.int32)
    xs = (np.linspace(0, img.shape[1] - 1, width)).astype(np.int32)
    return np.ascontiguousarray(img[ys][:, xs])


def _mosaic(head, left, right) -> np.ndarray:
    blank = np.zeros((_MOSAIC_W // 4, _MOSAIC_W, 3), dtype=np.uint8)
    top = _fit_width(head, _MOSAIC_W) if head is not None else blank
    half = _MOSAIC_W // 2
    left_i = _fit_width(left, half) if left is not None else np.zeros((half // 2, half, 3), np.uint8)
    right_i = _fit_width(right, half) if right is not None else np.zeros_like(left_i)
    h = min(left_i.shape[0], right_i.shape[0])
    bottom = np.concatenate([left_i[:h], right_i[:h]], axis=1)
    if bottom.shape[1] != top.shape[1]:
        bottom = _fit_width(bottom, top.shape[1])
    return np.ascontiguousarray(np.concatenate([top, bottom], axis=0))


def _image_msg(bgr: np.ndarray, stamp) -> Image:
    msg = Image()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = "map"
    msg.height, msg.width = int(bgr.shape[0]), int(bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = bgr.tobytes()
    return msg


def _cloud(points: np.ndarray, rgb, frame: str, stamp) -> PointCloud2:
    n = int(points.shape[0])
    packed = np.empty((n, 4), dtype=np.float32)
    packed[:, :3] = points
    rgb_u32 = np.uint32((int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2]))
    packed[:, 3] = np.array([rgb_u32], dtype=np.uint32).view(np.float32)[0]
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.data = packed.tobytes()
    msg.is_dense = True
    return msg


class DepthCloudNode(Node):
    def __init__(self) -> None:
        super().__init__("eai_depth_cloud")
        self._cams = _load_extrinsics()
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pubs = {}
        self._last = {}
        self._seen = set()
        self._color = {spec["color"]: None for spec in _CAMERAS}
        self._mosaic_pub = self.create_publisher(Image, _MOSAIC_TOPIC, qos_pub)
        self._mosaic_last = 0.0
        self._tf = StaticTransformBroadcaster(self)
        transforms = []
        for spec in _CAMERAS:
            cam = dict(self._cams.get(spec["color"]) or {}, color_topic=spec["color"])
            parent, xyz, quat, note = _placement(cam, spec)
            self.get_logger().info("%s: %s" % (spec["depth"], note))
            transforms.append(self._tf_msg(parent, spec["frame"], xyz, quat))
            self._pubs[spec["depth"]] = self.create_publisher(PointCloud2, spec["cloud"], qos_pub)
            self.create_subscription(Image, spec["depth"], self._make_cb(spec, cam), qos_sub)
            self.create_subscription(Image, spec["color"], self._make_color_cb(spec["color"]), qos_sub)
            self._last[spec["depth"]] = 0.0
        self._tf.sendTransform(transforms)

    def _make_color_cb(self, topic: str):
        def _cb(msg: Image) -> None:
            img = _bgr(msg)
            if img is None:
                return
            self._color[topic] = img
            now = time.monotonic()
            if now - self._mosaic_last < 0.2:
                return
            if not any(v is not None for v in self._color.values()):
                return
            self._mosaic_last = now
            mosaic = _mosaic(
                self._color.get("/camera/head_color"),
                self._color.get("/camera/hand_left_color"),
                self._color.get("/camera/hand_right_color"),
            )
            self._mosaic_pub.publish(_image_msg(mosaic, self.get_clock().now().to_msg()))
            if "mosaic" not in self._seen:
                self._seen.add("mosaic")
                self.get_logger().info(
                    "相机图像 %sx%s -> %s" % (mosaic.shape[1], mosaic.shape[0], _MOSAIC_TOPIC)
                )

        return _cb

    def _tf_msg(self, parent, child, xyz, quat) -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = float(xyz[0])
        msg.transform.translation.y = float(xyz[1])
        msg.transform.translation.z = float(xyz[2])
        msg.transform.rotation.x = float(quat[0])
        msg.transform.rotation.y = float(quat[1])
        msg.transform.rotation.z = float(quat[2])
        msg.transform.rotation.w = float(quat[3])
        return msg

    def _make_cb(self, spec, cam):
        def _cb(msg: Image) -> None:
            now = time.monotonic()
            if now - self._last[spec["depth"]] < _MIN_PERIOD:
                return
            self._last[spec["depth"]] = now
            try:
                depth = _depth_meters(msg)
            except Exception as exc:
                if spec["depth"] not in self._seen:
                    self.get_logger().warn("%s: %s" % (spec["depth"], exc))
                    self._seen.add(spec["depth"])
                return
            h, w = depth.shape
            fx, fy, cx, cy = _intrinsics(cam, w, h)
            sub = depth[::_STRIDE, ::_STRIDE]
            vs, us = np.mgrid[0:h:_STRIDE, 0:w:_STRIDE]
            valid = valid_depth_mask(sub)
            if not np.any(valid):
                return
            z = sub[valid]
            u = us[valid].astype(np.float32)
            v = vs[valid].astype(np.float32)
            points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1).astype(np.float32)
            # 相机时间戳和宿主机 TF 不是同一时钟，沿用原戳会被 RViz 丢掉。
            self._pubs[spec["depth"]].publish(
                _cloud(points, spec["rgb"], spec["frame"], self.get_clock().now().to_msg())
            )
            if spec["depth"] not in self._seen:
                self._seen.add(spec["depth"])
                self.get_logger().info(
                    "%s %sx%s -> %s  %d points" % (spec["depth"], w, h, spec["cloud"], points.shape[0])
                )

        return _cb


def main() -> int:
    rclpy.init()
    node = DepthCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
