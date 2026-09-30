"""GLFW-free MuJoCo passive viewer (EGL render + Tk display).

Compatible with the subset of ``mujoco.viewer.Handle`` used by MolmoSpaces
(``cam``, ``opt``, ``sync``, ``close``, ``is_running``).

On TurboVNC / no-GLX hosts, ``mujoco.viewer.launch_passive`` aborts the
process, so callers must patch *before* importing pipeline code that opens
a viewer.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import weakref
from typing import Any, Callable, Optional


def glx_available() -> bool:
    """Return True if this display has usable *hardware* GLX (via glxinfo).

    Do **not** probe with ``glfw.create_window``: on TurboVNC the native
    MuJoCo/GLFW path can abort the whole process when GLX is missing.

    Software GLX (llvmpipe / swrast) is treated as unavailable: GLFW windows
    often never appear on TurboVNC; callers should fall back to EGL+Tk.
    """
    import shutil
    import subprocess

    glxinfo = shutil.which("glxinfo")
    if not glxinfo:
        return False
    try:
        proc = subprocess.run(
            [glxinfo, "-B"],
            capture_output=True,
            text=True,
            timeout=8,
            env={**os.environ},
        )
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    out = ((proc.stdout or "") + (proc.stderr or "")).lower()
    if "error:" in out or "glx extension" in out:
        return False
    # Mesa software renderers: not useful for interactive MuJoCo GLFW.
    if any(x in out for x in ("llvmpipe", "softpipe", "swrast", "microsoft basic render")):
        return False
    if "opengl renderer string:" not in out and "opengl version string:" not in out:
        return False
    return True

class EglPassiveHandle:
    """Minimal Handle stand-in backed by MUJOCO_GL=egl + Tkinter."""

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        width: int = 960,
        height: int = 720,
        key_callback: Optional[Callable] = None,
        title: str = "MuJoCo EGL passive",
    ) -> None:
        import mujoco

        os.environ.setdefault("MUJOCO_GL", "egl")

        self._model = model
        self._data = data
        self._key_callback = key_callback
        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._cam)
        self._opt = mujoco.MjvOption()
        self._pert = mujoco.MjvPerturb()
        self._user_scn = None
        self._running = True
        self._cmd_q: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._status_msg: Optional[str] = None

        off_w = int(getattr(model.vis.global_, "offwidth", 640) or 640)
        off_h = int(getattr(model.vis.global_, "offheight", 480) or 480)
        self._width = max(64, min(int(width), off_w))
        self._height = max(64, min(int(height), off_h))
        self._title = title

        self._thread = threading.Thread(target=self._ui_main, name="egl-passive", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            self._running = False
            raise RuntimeError("EGL passive viewer failed to start (Tk/EGL)")

    @property
    def cam(self):
        return self._cam

    @property
    def opt(self):
        return self._opt

    @property
    def perturb(self):
        return self._pert

    @property
    def user_scn(self):
        return self._user_scn

    @property
    def m(self):
        return self._model

    @property
    def d(self):
        return self._data

    @property
    def viewport(self):
        return None

    def is_running(self) -> bool:
        return self._running and self._thread.is_alive()

    def sync(self) -> None:
        if not self._running:
            return
        # Ask UI thread to redraw; do not block physics long.
        try:
            self._cmd_q.put_nowait("sync")
        except queue.Full:
            pass

    def set_status(self, message: str) -> None:
        """Update the status bar text on the Tk viewer (thread-safe)."""
        self._status_msg = str(message or "")
        if not self._running:
            return
        try:
            self._cmd_q.put_nowait(("status", self._status_msg))
        except queue.Full:
            pass

    def close(self) -> None:
        if not self._running and self._closed.is_set():
            return
        self._running = False
        try:
            self._cmd_q.put_nowait("quit")
        except Exception:
            pass
        self._closed.wait(timeout=5.0)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def lock(self):
        return self._lock

    def set_figures(self, *args, **kwargs):
        return None

    def clear_figures(self):
        return None

    def set_texts(self, *args, **kwargs):
        return None

    def clear_texts(self):
        return None

    def set_images(self, *args, **kwargs):
        return None

    def clear_images(self):
        return None

    def update_hfield(self, *args, **kwargs):
        return None

    def update_mesh(self, *args, **kwargs):
        return None

    def update_texture(self, *args, **kwargs):
        return None

    def _ui_main(self) -> None:
        import mujoco
        import numpy as np
        import tkinter as tk
        from PIL import Image, ImageTk

        try:
            renderer = mujoco.Renderer(self._model, self._height, self._width)
        except Exception as exc:
            print(f"[egl-passive] Renderer init failed: {exc}", flush=True)
            self._running = False
            self._ready.set()
            self._closed.set()
            return

        root = tk.Tk()
        root.title(self._title)
        root.geometry(f"{self._width}x{self._height + 28}")
        status = tk.StringVar(value="EGL viewer (no GLFW/GLX)")
        tk.Label(root, textvariable=status, anchor="w").pack(fill="x", padx=4)
        panel = tk.Label(root, bd=0)
        panel.pack(fill="both", expand=True)
        photo_box = {"img": None}

        def _render_once() -> None:
            if not self._running:
                return
            with self._lock:
                # Do NOT call mj_forward here: the sim thread owns stepping /
                # forward, and concurrent mj_forward races can raise
                # FatalError (e.g. "equality ... between two static bodies").
                try:
                    renderer.update_scene(self._data, camera=self._cam)
                    rgb = renderer.render()
                except Exception as exc:
                    status.set(f"EGL viewer render skip: {exc}")
                    return
            img = Image.fromarray(np.asarray(rgb))
            photo = ImageTk.PhotoImage(img)
            photo_box["img"] = photo
            panel.configure(image=photo)

        def _poll() -> None:
            try:
                while True:
                    cmd = self._cmd_q.get_nowait()
                    if cmd == "quit":
                        self._running = False
                        _teardown_widgets()
                        root.quit()
                        return
                    if cmd == "sync":
                        _render_once()
                    elif isinstance(cmd, tuple) and cmd and cmd[0] == "status":
                        try:
                            status.set(str(cmd[1]))
                        except Exception:
                            pass
            except queue.Empty:
                pass
            if self._running:
                root.after(33, _poll)
            else:
                _teardown_widgets()
                root.quit()

        def _teardown_widgets() -> None:
            # Drop Tk image/var refs on the UI thread before the interpreter dies.
            try:
                photo_box["img"] = None
                panel.configure(image="")
            except Exception:
                pass
            try:
                status.set("")
            except Exception:
                pass

        def _on_key(event) -> None:
            if self._key_callback is None:
                return
            # Best-effort: pass tk keysym string; policies may ignore unknown keys.
            try:
                self._key_callback(event.keysym)
            except TypeError:
                try:
                    self._key_callback(event)
                except Exception:
                    pass
            except Exception:
                pass

        def _on_close() -> None:
            self._running = False
            _teardown_widgets()
            root.quit()

        root.bind("<Key>", _on_key)
        root.protocol("WM_DELETE_WINDOW", _on_close)

        _render_once()
        self._ready.set()
        root.after(33, _poll)
        try:
            root.mainloop()
        finally:
            self._running = False
            _teardown_widgets()
            try:
                renderer.close()
            except Exception:
                pass
            # Do NOT call root.destroy() here. Destroying (or GC of StringVar /
            # PhotoImage from another thread after worker teardown) triggers
            # Tcl_AsyncDelete / "main thread is not in main loop".
            self._closed.set()


def launch_passive_egl(
    model,
    data,
    *,
    key_callback=None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
):
    _ = show_left_ui, show_right_ui
    return EglPassiveHandle(model, data, key_callback=key_callback)


def patch_launch_passive(force_egl: bool = False) -> str:
    """Monkeypatch ``mujoco.viewer.launch_passive`` when GLX is unavailable.

    Returns a short status string describing the chosen backend.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")

    use_egl = force_egl or not glx_available()
    if not use_egl:
        return "glfw"

    import mujoco.viewer as mjv

    mjv.launch_passive = launch_passive_egl  # type: ignore[assignment]
    print(
        "[egl-passive] GLX unavailable — using EGL+Tk passive viewer",
        flush=True,
    )
    return "egl_tk"
