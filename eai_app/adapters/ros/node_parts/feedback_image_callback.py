"""adapters.ros.node parts.feedback image callback — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class FeedbackImageCallbackMixin:
    def _image_callback(self, msg: _eai_platform_bootstrap.Image, topic: str) -> None:
        try:
            now = _eai_platform_bootstrap.time.time()
            if msg.header.frame_id:
                self._camera_frame_ids[topic] = msg.header.frame_id
            # 心跳：未到显示间隔则不解码，避免 CPU/内存打满
            count = self._frame_counts.get(topic, 0)
            self._last_frame_time[topic] = now
            if count > 0:
                interval = (
                    _eai_configuration_policies.UI_DEPTH_MIN_INTERVAL_S
                    if _eai_perception_image_geometry.is_depth_topic(topic)
                    else _eai_configuration_policies.UI_IMAGE_MIN_INTERVAL_S
                )
                if now - self._last_ui_emit_time.get(topic, 0.0) < interval:
                    self._frame_counts[topic] = count + 1
                    return
            cv_image = self._convert_image(msg)
            if count == 0:
                self.get_logger().info(
                    f"首帧 {topic}: {msg.width}x{msg.height} encoding={msg.encoding}"
                )
            self._publish_frame_to_ui(topic, cv_image, now, (
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                msg.header.frame_id, msg.encoding))
        except Exception as exc:
            try:
                from cv_bridge import CvBridgeError

                if isinstance(exc, CvBridgeError):
                    self.get_logger().warn(f"{topic}: {exc}")
                    return
            except Exception:
                pass
            self.get_logger().error(f"{topic}: 处理失败 {exc}")

    def _compressed_callback(self, msg: _eai_platform_bootstrap.CompressedImage, topic: str) -> None:
        try:
            now = _eai_platform_bootstrap.time.time()
            if msg.header.frame_id:
                self._camera_frame_ids[topic] = msg.header.frame_id
            count = self._frame_counts.get(topic, 0)
            self._last_frame_time[topic] = now
            if count > 0:
                interval = (
                    _eai_configuration_policies.UI_DEPTH_MIN_INTERVAL_S
                    if _eai_perception_image_geometry.is_depth_topic(topic)
                    else _eai_configuration_policies.UI_IMAGE_MIN_INTERVAL_S
                )
                if now - self._last_ui_emit_time.get(topic, 0.0) < interval:
                    self._frame_counts[topic] = count + 1
                    return
            cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
            if count == 0:
                self.get_logger().info(f"首帧 {topic}: compressed {len(msg.data)} bytes")
            self._publish_frame_to_ui(topic, cv_image, now, (
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                msg.header.frame_id, "bgr8"))
        except Exception as exc:
            try:
                from cv_bridge import CvBridgeError

                if isinstance(exc, CvBridgeError):
                    self.get_logger().warn(f"{topic}: {exc}")
                    return
            except Exception:
                pass
            self.get_logger().error(f"{topic}: 处理失败 {exc}")

