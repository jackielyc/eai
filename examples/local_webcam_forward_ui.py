#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本机摄像头推流 + SSH 反向隧道（图形按钮版）。

在有摄像头的本机（推荐 macOS）运行本界面，对应远端 eai「本地摄像头」tab 的两项能力：

  1) 启动 MJPEG 推流  →  examples/local_webcam_mjpeg_server.py
  2) 启动 SSH 反向隧道 →  ssh -N -R 8090:127.0.0.1:8090 -p 10693 root@42.192.34.154

用法::

  cd /path/to/eai
  python3 examples/local_webcam_forward_ui.py

依赖: Python 标准库 + 推流脚本所需的 opencv（与 mjpeg server 相同）。
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

_HERE = os.path.dirname(os.path.abspath(__file__))
_EAI_DIR = os.path.dirname(_HERE)
_MJPEG_SCRIPT = os.path.join(_HERE, "local_webcam_mjpeg_server.py")
_SSH_TARGET_DEFAULT = "root@42.192.34.154"
_SSH_EXTRA_DEFAULT = "-p 10693 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes"


class LocalWebcamForwardApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("eai · 本机摄像头转发")
        self.root.geometry("720x520")
        self.root.minsize(640, 440)

        self._mjpeg_proc: subprocess.Popen | None = None
        self._tunnel_proc: subprocess.Popen | None = None

        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frm,
            text=(
                "在有摄像头的本机点击下方按钮。"
                "远端 eai「本地摄像头 / 真机评测」用 http://127.0.0.1:端口/cam.mjpg 预览。"
            ),
            wraplength=680,
        ).pack(fill=tk.X, **pad)

        # --- 推流 ---
        g1 = ttk.LabelFrame(frm, text="1. 本机推流 (MJPEG)", padding=8)
        g1.pack(fill=tk.X, **pad)
        r1 = ttk.Frame(g1)
        r1.pack(fill=tk.X)
        ttk.Label(r1, text="设备").pack(side=tk.LEFT)
        self.device_var = tk.StringVar(value="0")
        ttk.Entry(r1, textvariable=self.device_var, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(r1, text="端口").pack(side=tk.LEFT, padx=(12, 0))
        self.port_var = tk.StringVar(value="8090")
        ttk.Entry(r1, textvariable=self.port_var, width=8).pack(side=tk.LEFT, padx=4)
        self.mjpeg_btn = ttk.Button(r1, text="启动推流", command=self._toggle_mjpeg)
        self.mjpeg_btn.pack(side=tk.LEFT, padx=8)
        self.mjpeg_status = ttk.Label(r1, text="未启动")
        self.mjpeg_status.pack(side=tk.LEFT, padx=4)

        # --- 隧道 ---
        g2 = ttk.LabelFrame(frm, text="2. SSH 反向隧道", padding=8)
        g2.pack(fill=tk.X, **pad)
        r2 = ttk.Frame(g2)
        r2.pack(fill=tk.X)
        ttk.Label(r2, text="远端").pack(side=tk.LEFT)
        self.ssh_target_var = tk.StringVar(
            value=os.environ.get("EAI_SSH_REVERSE_TARGET", _SSH_TARGET_DEFAULT)
        )
        ttk.Entry(r2, textvariable=self.ssh_target_var, width=28).pack(side=tk.LEFT, padx=4)
        ttk.Label(r2, text="远端端口").pack(side=tk.LEFT, padx=(8, 0))
        self.remote_port_var = tk.StringVar(value="8090")
        ttk.Entry(r2, textvariable=self.remote_port_var, width=8).pack(side=tk.LEFT, padx=4)
        self.tunnel_btn = ttk.Button(r2, text="启动隧道", command=self._toggle_tunnel)
        self.tunnel_btn.pack(side=tk.LEFT, padx=8)
        self.tunnel_status = ttk.Label(r2, text="未启动")
        self.tunnel_status.pack(side=tk.LEFT, padx=4)

        r2b = ttk.Frame(g2)
        r2b.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(r2b, text="额外 ssh 参数").pack(side=tk.LEFT)
        self.ssh_extra_var = tk.StringVar(value=_SSH_EXTRA_DEFAULT)
        ttk.Entry(r2b, textvariable=self.ssh_extra_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # --- 日志 ---
        ttk.Label(frm, text="日志").pack(anchor=tk.W, padx=8)
        self.log = scrolledtext.ScrolledText(frm, height=16, font=("Menlo", 11))
        self.log.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._log("就绪。先「启动推流」，再「启动隧道」。")
        if not os.path.isfile(_MJPEG_SCRIPT):
            self._log(f"警告: 找不到推流脚本 {_MJPEG_SCRIPT}")

    def _log(self, line: str) -> None:
        self.log.insert(tk.END, line.rstrip() + "\n")
        self.log.see(tk.END)

    def _append_proc_output(self, proc: subprocess.Popen, prefix: str) -> None:
        assert proc.stdout is not None

        def _reader() -> None:
            for raw in proc.stdout:
                line = raw.rstrip()
                if line:
                    self.root.after(0, lambda l=line: self._log(f"[{prefix}] {l}"))

        threading.Thread(target=_reader, daemon=True).start()

    def _toggle_mjpeg(self) -> None:
        if self._mjpeg_proc is not None and self._mjpeg_proc.poll() is None:
            self._stop_mjpeg()
            return
        if not os.path.isfile(_MJPEG_SCRIPT):
            messagebox.showerror("推流", f"找不到脚本:\n{_MJPEG_SCRIPT}")
            return
        device = self.device_var.get().strip() or "0"
        try:
            port = int(self.port_var.get().strip() or "8090")
        except ValueError:
            messagebox.showerror("推流", "端口必须是整数")
            return
        cmd = [
            sys.executable,
            _MJPEG_SCRIPT,
            "--device",
            device,
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
        ]
        self._log("$ " + " ".join(shlex.quote(c) for c in cmd))
        try:
            self._mjpeg_proc = subprocess.Popen(
                cmd,
                cwd=_EAI_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            messagebox.showerror("推流", str(exc))
            return
        self._append_proc_output(self._mjpeg_proc, "推流")
        self.mjpeg_btn.configure(text="停止推流")
        self.mjpeg_status.configure(text=f"运行中 :{port}")
        self.root.after(800, self._poll_mjpeg)

    def _poll_mjpeg(self) -> None:
        proc = self._mjpeg_proc
        if proc is None:
            return
        code = proc.poll()
        if code is None:
            self.root.after(800, self._poll_mjpeg)
            return
        self._mjpeg_proc = None
        self.mjpeg_btn.configure(text="启动推流")
        self.mjpeg_status.configure(text=f"已退出 code={code}")
        self._log(f"[推流] 进程退出 code={code}")

    def _stop_mjpeg(self) -> None:
        proc = self._mjpeg_proc
        self._mjpeg_proc = None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
        self.mjpeg_btn.configure(text="启动推流")
        self.mjpeg_status.configure(text="已停止")
        self._log("[推流] 已停止")

    def _toggle_tunnel(self) -> None:
        if self._tunnel_proc is not None and self._tunnel_proc.poll() is None:
            self._stop_tunnel()
            return
        target = self.ssh_target_var.get().strip()
        if not target or "@" not in target or "remote-host" in target:
            messagebox.showwarning(
                "隧道",
                "请填写真实远端，例如:\n  root@42.192.34.154\n或设置环境变量 EAI_SSH_REVERSE_TARGET",
            )
            return
        try:
            local_port = int(self.port_var.get().strip() or "8090")
            remote_port = int(self.remote_port_var.get().strip() or str(local_port))
        except ValueError:
            messagebox.showerror("隧道", "端口必须是整数")
            return
        extra = shlex.split(self.ssh_extra_var.get().strip())
        cmd = [
            "ssh",
            "-N",
            "-R",
            f"{remote_port}:127.0.0.1:{local_port}",
            *extra,
            target,
        ]
        self._log("$ " + " ".join(shlex.quote(c) for c in cmd))
        try:
            self._tunnel_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            messagebox.showerror("隧道", str(exc))
            return
        self._append_proc_output(self._tunnel_proc, "隧道")
        self.tunnel_btn.configure(text="停止隧道")
        self.tunnel_status.configure(text=f"运行中 → {target}:{remote_port}")
        self.root.after(800, self._poll_tunnel)

    def _poll_tunnel(self) -> None:
        proc = self._tunnel_proc
        if proc is None:
            return
        code = proc.poll()
        if code is None:
            self.root.after(800, self._poll_tunnel)
            return
        self._tunnel_proc = None
        self.tunnel_btn.configure(text="启动隧道")
        self.tunnel_status.configure(text=f"已退出 code={code}")
        self._log(f"[隧道] 进程退出 code={code}")

    def _stop_tunnel(self) -> None:
        proc = self._tunnel_proc
        self._tunnel_proc = None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
        self.tunnel_btn.configure(text="启动隧道")
        self.tunnel_status.configure(text="已停止")
        self._log("[隧道] 已停止")

    def _on_close(self) -> None:
        self._stop_mjpeg()
        self._stop_tunnel()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    LocalWebcamForwardApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
