#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在「有摄像头的本机」上采集并提供 MJPEG HTTP 流，供远端手骨架遥控使用。

典型 VNC 场景（摄像头在笔记本，UI 跑在远端）::

  # 本机终端 1：推流
  python3 examples/local_webcam_mjpeg_server.py --device 0 --port 8090

  # 本机终端 2：把本机 8090 映射到远端 localhost:8090
  ssh -N -R 8090:127.0.0.1:8090 user@remote-host

  # 远端 UI：手骨架遥控 → 相机选「网络流」
  # URL: http://127.0.0.1:8090/cam.mjpg → 开始识别

依赖: opencv-python / opencv-contrib-python（与 eai conda 一致即可）。
"""

from __future__ import annotations

import argparse
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


def _open_capture(device: int | str):
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    if isinstance(device, int):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(device)
    else:
        cap = cv2.VideoCapture(device)
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description="Local webcam → MJPEG HTTP server")
    parser.add_argument("--device", default="0", help="摄像头 index 或路径，默认 0")
    parser.add_argument("--port", type=int, default=8090, help="监听端口，默认 8090")
    parser.add_argument("--bind", default="127.0.0.1", help="监听地址，默认仅本机")
    parser.add_argument("--width", type=int, default=640, help="可选缩放宽度（0=不缩放）")
    parser.add_argument("--fps", type=float, default=15.0, help="目标帧率上限")
    parser.add_argument("--quality", type=int, default=70, help="JPEG 质量 1-100")
    args = parser.parse_args()

    device: int | str
    try:
        device = int(args.device)
    except ValueError:
        device = str(args.device)

    cap = _open_capture(device)
    if not cap.isOpened():
        print(f"无法打开摄像头: {device}", file=sys.stderr)
        return 1
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    boundary = b"frame"
    path_ok = {"/", "/cam.mjpg", "/cam.mjpeg"}
    min_interval = 1.0 / max(1.0, float(args.fps))
    quality = max(1, min(100, int(args.quality)))
    state = {"cap": cap, "last_t": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *fmt_args) -> None:  # noqa: A003
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % fmt_args))

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] not in path_ok:
                self.send_error(404, "try /cam.mjpg")
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={boundary.decode()}",
            )
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            local_cap = state["cap"]
            try:
                while True:
                    now = time.time()
                    wait = min_interval - (now - state["last_t"])
                    if wait > 0:
                        time.sleep(wait)
                    ok, frame = local_cap.read()
                    state["last_t"] = time.time()
                    if not ok or frame is None:
                        time.sleep(0.05)
                        continue
                    if args.width and frame.shape[1] > args.width:
                        h, w = frame.shape[:2]
                        nh = max(1, int(h * (args.width / float(w))))
                        frame = cv2.resize(frame, (args.width, nh), interpolation=cv2.INTER_AREA)
                    ok_enc, buf = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                    )
                    if not ok_enc:
                        continue
                    jpg = buf.tobytes()
                    self.wfile.write(b"--" + boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    server = ThreadingHTTPServer((args.bind, int(args.port)), Handler)
    print(
        f"MJPEG 已启动: http://{args.bind}:{args.port}/cam.mjpg  (device={device})",
        flush=True,
    )
    print(
        "远端 VNC 场景请在本机另开: "
        f"ssh -N -R {args.port}:127.0.0.1:{args.port} user@remote",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止", flush=True)
    finally:
        server.server_close()
        cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
