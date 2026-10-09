#!/usr/bin/env python3
"""Async RynnValue Live HUD for sim rollouts.

Env-agnostic scoring / overlay layer: push RGB frames from any sim loop,
score via reward_server on a wall-clock interval (not control-rate), and
draw an OpenCV overlay of remaining time.

Per-sim observation adapters live in ``rynnvalue_sim_bridge`` (LIBERO first;
RoboTwin / MolmoSpaces / MuJoCo profiles stubbed for the same CLI).
"""
from __future__ import annotations

import io
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://127.0.0.1:8001"
DEFAULT_ROBOT_DESCRIPTION = "a Franka single-arm robot"
DEFAULT_CAMERA_DESCRIPTION = "the third-person agentview camera"
# OpenCV window defaults (WINDOW_NORMAL + free resize).
DEFAULT_WINDOW_WIDTH = 960
DEFAULT_WINDOW_HEIGHT = 720
# Upscale tiny sim frames before drawing HUD so text/plot stay readable.
HUD_DRAW_MIN_SIDE = 720


def _primary_monitor_rect() -> Tuple[int, int, int, int]:
    """Return (origin_x, origin_y, width, height) of the primary monitor."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        sw = int(root.winfo_screenwidth())
        sh = int(root.winfo_screenheight())
        root.destroy()
        if sw > 0 and sh > 0:
            return 0, 0, sw, sh
    except Exception:  # noqa: BLE001
        pass
    return 0, 0, 1920, 1080


# OpenCV QT5 HighGUI persists geometry by window title; bumping a zero-width
# suffix makes each open a fresh window so restore cannot yank it off-center.
_WINDOW_NAME_SEQ = 0


def _fresh_window_name(base: str) -> str:
    global _WINDOW_NAME_SEQ
    _WINDOW_NAME_SEQ += 1
    # Title bar still reads as ``base``; ZWSP count is invisible.
    # Keep the escape outside the f-string expr (SyntaxError on Python < 3.12).
    zwsp = "\u200b"
    return f"{base}{zwsp * _WINDOW_NAME_SEQ}"


def _center_opencv_window(
    cv2_mod: Any,
    window_name: str,
    width: int = DEFAULT_WINDOW_WIDTH,
    height: int = DEFAULT_WINDOW_HEIGHT,
) -> bool:
    """Place an OpenCV window near the center of the primary screen."""
    w, h = max(1, int(width)), max(1, int(height))
    try:
        rect = cv2_mod.getWindowImageRect(window_name)
        if rect is not None and int(rect[2]) > 0 and int(rect[3]) > 0:
            w, h = int(rect[2]), int(rect[3])
    except Exception:  # noqa: BLE001
        pass
    ox, oy, sw, sh = _primary_monitor_rect()
    x = int(ox + max(0, (sw - w) // 2))
    y = int(oy + max(0, (sh - h) // 2))
    try:
        cv2_mod.moveWindow(window_name, x, y)
        cv2_mod.waitKey(1)
        return True
    except Exception:  # noqa: BLE001
        return False


def _set_opencv_window_topmost(cv2_mod: Any, window_name: str) -> None:
    """Keep the Live HUD above other windows (sim viewer, main UI)."""
    prop = getattr(cv2_mod, "WND_PROP_TOPMOST", None)
    if prop is not None:
        try:
            cv2_mod.setWindowProperty(window_name, prop, 1.0)
        except Exception:  # noqa: BLE001
            pass
    # QT HighGUI often ignores / cannot report TOPMOST; pin via EWMH as well.
    _x11_set_window_above(window_name)


def _x11_set_window_above(window_name: str) -> None:
    """Best-effort ``_NET_WM_STATE_ABOVE`` for the OpenCV window title."""
    want = window_name or ""
    want_plain = want.replace("\u200b", "").strip()
    if not want_plain:
        return
    try:
        from Xlib import X, display
        from Xlib.protocol import event
    except Exception:  # noqa: BLE001
        return
    try:
        dpy = display.Display()
        root = dpy.screen().root
        net_client_list = dpy.intern_atom("_NET_CLIENT_LIST")
        net_wm_name = dpy.intern_atom("_NET_WM_NAME")
        wm_name = dpy.intern_atom("WM_NAME")
        net_wm_state = dpy.intern_atom("_NET_WM_STATE")
        net_wm_state_above = dpy.intern_atom("_NET_WM_STATE_ABOVE")
        prop = root.get_full_property(net_client_list, X.AnyPropertyType)
        if prop is None:
            dpy.close()
            return

        def _title(win: Any) -> str:
            for atom in (net_wm_name, wm_name):
                try:
                    p = win.get_full_property(atom, 0)
                except Exception:  # noqa: BLE001
                    continue
                if not p or p.value is None:
                    continue
                raw = p.value
                text = (
                    raw.decode("utf-8", "ignore")
                    if isinstance(raw, (bytes, bytearray))
                    else str(raw)
                )
                return text.replace("\x1b%G", "").replace("\x1b%@", "")
            return ""

        exact = None
        fuzzy = None
        # Prefer the last matching client (most recently mapped).
        for wid in prop.value:
            win = dpy.create_resource_object("window", wid)
            text = _title(win)
            if not text:
                continue
            if text == want or text.replace("\u200b", "") == want_plain:
                exact = win
            elif want_plain in text.replace("\u200b", ""):
                fuzzy = win
        target = exact if exact is not None else fuzzy
        if target is not None:
            # data: action=1 (add), first property=ABOVE, source=1 (application)
            ev = event.ClientMessage(
                window=target,
                client_type=net_wm_state,
                data=(32, [1, int(net_wm_state_above), 0, 1, 0]),
            )
            mask = X.SubstructureRedirectMask | X.SubstructureNotifyMask
            root.send_event(ev, event_mask=mask)
            dpy.flush()
        dpy.close()
    except Exception:  # noqa: BLE001
        pass


@dataclass
class HudSnapshot:
    remaining_s: Optional[float] = None
    reward_phi: Optional[float] = None
    progress_curve: List[float] = field(default_factory=list)
    step: int = 0
    scored_at: float = 0.0
    latency_s: float = 0.0
    error: str = ""
    instruction: str = ""


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"unexpected frame shape: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _uniform_sample(frames: Sequence[np.ndarray], num_frames: int) -> np.ndarray:
    if not frames:
        raise ValueError("empty frame buffer")
    n = len(frames)
    k = max(1, int(num_frames))
    if n == 1 or k == 1:
        idx = [0]
    elif n <= k:
        idx = list(range(n))
    else:
        idx = np.linspace(0, n - 1, k, dtype=int).tolist()
    return np.stack([_as_uint8_rgb(frames[i]) for i in idx], axis=0)


def _numpy_to_npy_file_tuple(arr: np.ndarray, filename: str) -> Tuple[str, io.BytesIO, str]:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return (filename, buf, "application/octet-stream")


def _build_multipart_payload(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    files: Dict[str, Any] = {}
    data: Dict[str, str] = {}
    for i, sample in enumerate(samples):
        sample_copy = json.loads(json.dumps(sample, default=str))
        traj = sample.get("trajectory", {})
        traj_copy = sample_copy.get("trajectory", {})
        frames = traj.get("frames")
        if isinstance(frames, np.ndarray):
            file_key = f"sample_{i}_trajectory_frames"
            files[file_key] = _numpy_to_npy_file_tuple(frames, f"{file_key}.npy")
            traj_copy["frames"] = {"__numpy_file__": file_key}
        if "frames_shape" in traj_copy and isinstance(traj_copy["frames_shape"], (tuple, list)):
            traj_copy["frames_shape"] = [int(x) for x in traj_copy["frames_shape"]]
        sample_copy["trajectory"] = traj_copy
        data[f"sample_{i}"] = json.dumps(sample_copy)
    return files, data


def _post_multipart(
    url: str,
    fields: Dict[str, str],
    files: Dict[str, Tuple[str, io.BytesIO, str]],
    timeout_s: float,
) -> Dict[str, Any]:
    """stdlib multipart POST (avoids requests dependency in libero env)."""
    import uuid
    import urllib.error
    import urllib.request

    boundary = f"----RynnValueHud{uuid.uuid4().hex}"
    body = io.BytesIO()

    def _write(data: bytes) -> None:
        body.write(data)

    for key, value in fields.items():
        _write(f"--{boundary}\r\n".encode("utf-8"))
        _write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        _write(str(value).encode("utf-8"))
        _write(b"\r\n")

    for key, (filename, buf, content_type) in files.items():
        raw = buf.getvalue() if hasattr(buf, "getvalue") else buf.read()
        _write(f"--{boundary}\r\n".encode("utf-8"))
        _write(
            (
                f'Content-Disposition: form-data; name="{key}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        _write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        _write(raw)
        _write(b"\r\n")

    _write(f"--{boundary}--\r\n".encode("utf-8"))
    payload = body.getvalue()
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(payload)),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    return json.loads(raw.decode("utf-8"))


def score_frames(
    server_url: str,
    frames: np.ndarray,
    instruction: str,
    *,
    robot_description: str = DEFAULT_ROBOT_DESCRIPTION,
    camera_description: str = DEFAULT_CAMERA_DESCRIPTION,
    timeout_s: float = 60.0,
    use_frame_steps: bool = False,
) -> Tuple[np.ndarray, float]:
    """POST frames to reward_server; return (progress_pred, latency_s)."""
    t0 = time.time()
    sample = {
        "sample_type": "progress",
        "trajectory": {
            "frames": frames,
            "frames_shape": tuple(int(x) for x in frames.shape),
            "task": instruction,
            "id": "live_hud",
            "metadata": {
                "subsequence_length": int(frames.shape[0]),
                "robot_description": robot_description,
                "camera_description": camera_description,
            },
            "video_embeddings": None,
        },
    }
    files, data = _build_multipart_payload([sample])
    data["use_frame_steps"] = "true" if use_frame_steps else "false"
    url = server_url.rstrip("/") + "/evaluate_batch_npy"
    payload = _post_multipart(url, data, files, timeout_s=timeout_s)
    latency = time.time() - t0
    outputs_progress = payload.get("outputs_progress") or {}
    progress_pred = outputs_progress.get("progress_pred") or []
    if progress_pred and len(progress_pred) > 0:
        arr = np.asarray(progress_pred[0], dtype=np.float32).reshape(-1)
    else:
        arr = np.asarray([], dtype=np.float32)
    return arr, latency


def _fit_put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int,
    max_width: int,
) -> None:
    """Draw text clipped to ``max_width`` (ellipsis when truncated)."""
    import cv2

    if max_width <= 8 or not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    shown = text
    while shown:
        (tw, _th), _ = cv2.getTextSize(shown, font, font_scale, thickness)
        if tw <= max_width:
            break
        if len(shown) <= 1:
            shown = ""
            break
        cut = max(1, len(shown) - 2)
        shown = shown[:cut] + ("…" if cut < len(text) else "")
    if shown:
        cv2.putText(img, shown, org, font, font_scale, color, thickness, cv2.LINE_AA)


def draw_hud_overlay(
    frame: np.ndarray,
    snap: HudSnapshot,
    *,
    history: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return BGR image with remaining-time HUD (OpenCV)."""
    import cv2

    img = _as_uint8_rgb(frame)
    h0, w0 = img.shape[:2]
    # Upscale tiny sim frames so panel text / curve stay readable in the window.
    side = max(h0, w0)
    if side < HUD_DRAW_MIN_SIDE:
        scale = HUD_DRAW_MIN_SIDE / float(side)
        img = cv2.resize(
            img,
            (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))),
            interpolation=cv2.INTER_LINEAR,
        )
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # Top panel: left text column + right curve plot (no overlap).
    panel_h = min(160, max(108, h // 5))
    plot_w = max(160, min(280, w // 3))
    plot_pad = 10
    plot_x0 = w - plot_w - plot_pad
    text_max_w = max(40, plot_x0 - 20)

    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, bgr, 0.35, 0, bgr)

    age = (time.time() - snap.scored_at) if snap.scored_at else float("nan")
    if snap.remaining_s is not None:
        title = f"RynnValue  rem={snap.remaining_s:.2f}s"
        phi_line = f"Phi={-(snap.remaining_s):.2f}"
        color = (80, 220, 120)
    elif snap.error:
        title = f"RynnValue  ERR: {snap.error[:64]}"
        phi_line = ""
        color = (60, 60, 220)
    else:
        title = "RynnValue  waiting…"
        phi_line = ""
        color = (180, 180, 180)

    _fit_put_text(bgr, title, (12, 32), 0.7, color, 2, text_max_w)
    if phi_line:
        _fit_put_text(bgr, phi_line, (12, 58), 0.6, color, 2, text_max_w)
        meta_y = 82
    else:
        meta_y = 58
    meta = f"step={snap.step}  latency={snap.latency_s:.2f}s  age={age:.1f}s"
    _fit_put_text(bgr, meta, (12, meta_y), 0.5, (220, 220, 220), 1, text_max_w)
    if snap.instruction:
        _fit_put_text(
            bgr,
            snap.instruction,
            (12, min(panel_h - 12, meta_y + 24)),
            0.45,
            (200, 200, 200),
            1,
            text_max_w,
        )

    hist = list(history or [])
    if len(hist) >= 2:
        plot_y0 = 10
        plot_h = panel_h - 20
        cv2.rectangle(
            bgr,
            (plot_x0, plot_y0),
            (plot_x0 + plot_w, plot_y0 + plot_h),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            bgr,
            (plot_x0, plot_y0),
            (plot_x0 + plot_w, plot_y0 + plot_h),
            (90, 90, 90),
            1,
        )
        vals = np.asarray(hist[-64:], dtype=np.float32)
        vmin, vmax = float(vals.min()), float(vals.max())
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
        # Leave a small right margin inside the plot for the latest value label.
        label_w = 56
        xs = np.linspace(plot_x0 + 4, plot_x0 + plot_w - label_w - 4, num=len(vals))
        pts = []
        for x, v in zip(xs, vals):
            y = plot_y0 + plot_h - 4 - (float(v) - vmin) / (vmax - vmin) * (plot_h - 8)
            pts.append([int(x), int(y)])
        cv2.polylines(
            bgr,
            [np.asarray(pts, dtype=np.int32)],
            False,
            (80, 200, 255),
            2,
            cv2.LINE_AA,
        )
        last_v = float(vals[-1])
        last_y = int(
            plot_y0 + plot_h - 4 - (last_v - vmin) / (vmax - vmin) * (plot_h - 8)
        )
        last_y = max(plot_y0 + 14, min(plot_y0 + plot_h - 6, last_y))
        cv2.putText(
            bgr,
            f"{last_v:.2f}",
            (plot_x0 + plot_w - label_w + 2, last_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 200, 255),
            1,
            cv2.LINE_AA,
        )

    return bgr


class RynnValueLiveHud:
    """Non-blocking Live HUD: push frames from the sim loop; scores by wall clock."""

    def __init__(
        self,
        *,
        server_url: str = DEFAULT_SERVER_URL,
        instruction: str = "",
        refresh_sec: float = 1.0,
        num_frames: int = 8,
        max_buffer: int = 256,
        timeout_s: float = 60.0,
        robot_description: str = DEFAULT_ROBOT_DESCRIPTION,
        camera_description: str = DEFAULT_CAMERA_DESCRIPTION,
        show_window: bool = True,
        window_name: str = "RynnValue Live HUD",
        status_path: Optional[str] = None,
        use_frame_steps: bool = False,
        flip_ud: bool = True,
    ) -> None:
        self.server_url = (server_url or DEFAULT_SERVER_URL).rstrip("/")
        self.instruction = instruction or ""
        self.refresh_sec = max(0.2, float(refresh_sec))
        self.num_frames = max(1, int(num_frames))
        self.max_buffer = max(self.num_frames, int(max_buffer))
        self.timeout_s = float(timeout_s)
        self.robot_description = robot_description
        self.camera_description = camera_description
        self.show_window = bool(show_window)
        self._window_name_base = window_name or "RynnValue Live HUD"
        self.window_name = self._window_name_base
        self.status_path = Path(status_path) if status_path else None
        self.use_frame_steps = bool(use_frame_steps)
        # LIBERO/robosuite offscreen RGB is OpenGL origin (first row = bottom).
        # mujoco.Renderer already returns top-left origin — do not flip that path.
        self.flip_ud = bool(flip_ud)

        self._frames: Deque[np.ndarray] = deque(maxlen=self.max_buffer)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snap = HudSnapshot(instruction=self.instruction)
        self._history: Deque[float] = deque(maxlen=256)
        self._last_overlay_bgr: Optional[np.ndarray] = None
        self._last_display_rgb: Optional[np.ndarray] = None
        self._inflight = False
        self._dirty = False
        self._step = 0
        self._cv2 = None
        self._window_dismissed = False  # user closed window / pressed Esc
        self._last_err_log_at = 0.0
        self._last_err_text = ""
        self._last_overlay_write_at = 0.0
        self._overlay_write_min_interval_s = 0.1  # ~10 Hz for eai workspace preview
        # Keep recentering for a few frames after (re)create.
        self._center_attempts_left = 0
        self._window_seen_open = False
        self._window_missing_streak = 0

    def _arm_window_center(self, attempts: int = 20) -> None:
        """Schedule recenter on subsequent pump_window calls."""
        self._center_attempts_left = max(0, int(attempts))

    def _maybe_center_window(self, overlay: Optional[np.ndarray] = None) -> None:
        """Retry moveWindow for a few frames after create / recreate."""
        if self._center_attempts_left <= 0 or self._cv2 is None:
            return
        self._center_attempts_left -= 1
        if overlay is not None:
            h, w = int(overlay.shape[0]), int(overlay.shape[1])
        else:
            w, h = DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
        try:
            _center_opencv_window(self._cv2, self.window_name, w, h)
        except Exception:  # noqa: BLE001
            pass

    def _ensure_window(self) -> bool:
        """Create the OpenCV window if needed. Deferred until first frame.

        Deferring avoids robosuite ``destroyAllWindows()`` during env reset
        wiping a blank HUD that was opened too early in ``start()``.
        """
        if self._window_dismissed or not self.show_window:
            return False
        try:
            import cv2
        except Exception as exc:  # noqa: BLE001
            logger.warning("Live HUD window unavailable: %s", exc)
            self.show_window = False
            self._cv2 = None
            return False
        self._cv2 = cv2
        if not self.window_name or self.window_name == self._window_name_base:
            self.window_name = _fresh_window_name(self._window_name_base)
        try:
            # If already open, leave it.
            if self.window_is_open():
                return True
        except Exception:  # noqa: BLE001
            pass
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.window_name, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
            )
            prop = getattr(cv2, "WND_PROP_ASPECT_RATIO", None)
            free_ratio = getattr(cv2, "WINDOW_FREERATIO", None)
            if prop is not None and free_ratio is not None:
                try:
                    cv2.setWindowProperty(
                        self.window_name, prop, float(free_ratio)
                    )
                except Exception:  # noqa: BLE001
                    pass
            _set_opencv_window_topmost(cv2, self.window_name)
            self._arm_window_center()
            self._window_seen_open = False
            self._window_missing_streak = 0
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Live HUD window unavailable: %s", exc)
            self.show_window = False
            self._cv2 = None
            return False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Import cv2 early; actual namedWindow waits until first pump_window so
        # sim backends that call destroyAllWindows on reset cannot wipe it.
        if self.show_window and not self._window_dismissed:
            try:
                import cv2

                self._cv2 = cv2
                self.window_name = _fresh_window_name(self._window_name_base)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Live HUD window unavailable: %s", exc)
                self.show_window = False
                self._cv2 = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="rynnvalue-live-hud", daemon=True)
        self._thread.start()
        logger.info(
            "RynnValue Live HUD started url=%s refresh=%.2fs num_frames=%s",
            self.server_url,
            self.refresh_sec,
            self.num_frames,
        )

    def dismiss_window(self, reason: str = "") -> None:
        """Close the OpenCV window and stop scoring.

        After dismiss, ``imshow`` is skipped so the window does not pop back open
        on the next frame (OpenCV recreates windows on imshow otherwise).
        """
        if self._window_dismissed:
            return
        self._window_dismissed = True
        self.show_window = False
        # Stop the scoring worker immediately; drop buffered frames so an
        # in-flight HTTP reply is the last possible score (and is discarded).
        with self._lock:
            self._frames.clear()
            self._dirty = False
        self._stop.set()
        self._wake.set()
        if self._cv2 is not None:
            try:
                # Never destroyAllWindows — that would kill sibling sim viewers.
                self._cv2.destroyWindow(self.window_name)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._cv2.waitKey(1)
            except Exception:  # noqa: BLE001
                pass
        extra = f" ({reason})" if reason else ""
        logger.info("Live HUD window closed%s; scoring stopped", extra)
        print(f"[rynnvalue] Live HUD window closed{extra}; scoring stopped", flush=True)

    def window_is_open(self) -> bool:
        if self._window_dismissed or not self.show_window or self._cv2 is None:
            return False
        try:
            visible = self._cv2.getWindowProperty(
                self.window_name, self._cv2.WND_PROP_VISIBLE
            )
            # Closed window reports < 1 (Qt often -1).
            return float(visible) >= 1.0
        except Exception:  # noqa: BLE001
            return False

    def user_closed_window(self) -> bool:
        return bool(self._window_dismissed)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._cv2.waitKey(1)
            except Exception:  # noqa: BLE001
                pass
        self.show_window = False

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction or ""
        with self._lock:
            self._snap.instruction = self.instruction

    def reset_episode(self, instruction: str = "") -> None:
        with self._lock:
            self._frames.clear()
            self._history.clear()
            self._step = 0
            if instruction:
                self.instruction = instruction
            self._snap = HudSnapshot(instruction=self.instruction)
            self._last_overlay_bgr = None
            self._dirty = False

    def push_frame(
        self,
        frame: np.ndarray,
        *,
        step: Optional[int] = None,
        display_frame: Optional[np.ndarray] = None,
    ) -> HudSnapshot:
        """Push a frame for scoring; optionally show a different image in the window.

        ``frame`` goes to the score buffer (e.g. openpi-flipped LIBERO agentview).
        ``display_frame`` (if set) is used only for the OpenCV overlay / saved HUD
        video so the window can match the sim viewer orientation.
        """
        if self._window_dismissed or self._stop.is_set():
            return self.snapshot()
        rgb = _as_uint8_rgb(frame)
        display_rgb = (
            _as_uint8_rgb(display_frame) if display_frame is not None else rgb
        )
        if self.flip_ud:
            rgb = np.ascontiguousarray(rgb[::-1])
            display_rgb = (
                rgb
                if display_frame is None
                else np.ascontiguousarray(display_rgb[::-1])
            )
        with self._lock:
            self._frames.append(rgb)
            if step is not None:
                self._step = int(step)
            else:
                self._step += 1
            self._dirty = True
            snap = HudSnapshot(
                remaining_s=self._snap.remaining_s,
                reward_phi=self._snap.reward_phi,
                progress_curve=list(self._snap.progress_curve),
                step=self._step,
                scored_at=self._snap.scored_at,
                latency_s=self._snap.latency_s,
                error=self._snap.error,
                instruction=self.instruction,
            )
            history = list(self._history)

        overlay = draw_hud_overlay(display_rgb, snap, history=history)
        with self._lock:
            self._last_overlay_bgr = overlay
            self._last_display_rgb = display_rgb
        self._write_overlay_image(overlay)

        self.pump_window(force_overlay=overlay)
        return snap

    def pump_window(self, force_overlay: Optional[np.ndarray] = None) -> None:
        """Refresh OpenCV window (call from the process/thread that owns the GUI).

        Qt/HighGUI is not thread-safe: RoboTwin should run the frame-bus sidecar
        in a dedicated process so ``imshow`` / ``waitKey`` stay on that main thread.

        Closing the window (X) or pressing Esc/q stops the GUI and scoring.
        Subsequent ``imshow`` calls are skipped so the window does not pop back open.
        """
        if self._window_dismissed or not self.show_window:
            return
        if self._cv2 is None or not self._ensure_window():
            return
        overlay = force_overlay
        if overlay is None:
            # Rebuild overlay when score snapshot advanced since last push.
            with self._lock:
                display_rgb = self._last_display_rgb
                snap = HudSnapshot(
                    remaining_s=self._snap.remaining_s,
                    reward_phi=self._snap.reward_phi,
                    progress_curve=list(self._snap.progress_curve),
                    step=self._snap.step,
                    scored_at=self._snap.scored_at,
                    latency_s=self._snap.latency_s,
                    error=self._snap.error,
                    instruction=self._snap.instruction,
                )
                history = list(self._history)
                cached = self._last_overlay_bgr
            if display_rgb is not None:
                overlay = draw_hud_overlay(display_rgb, snap, history=history)
                with self._lock:
                    self._last_overlay_bgr = overlay
                self._write_overlay_image(overlay)
            else:
                overlay = cached
        try:
            if overlay is not None:
                # Recreate if a sim backend wiped HighGUI windows.
                if not self.window_is_open():
                    self._ensure_window()
                self._cv2.imshow(self.window_name, overlay)
                # Re-assert topmost while recentering (not every frame — X11 is heavy).
                if self._center_attempts_left > 0:
                    _set_opencv_window_topmost(self._cv2, self.window_name)
                self._maybe_center_window(overlay)
            key = int(self._cv2.waitKey(1) & 0xFF)
        except Exception as exc:  # noqa: BLE001
            logger.warning("imshow failed: %s", exc)
            self.dismiss_window("imshow error")
            return
        if key in (27, ord("q"), ord("Q")):  # Esc / q
            self.dismiss_window("Esc/q")
            return
        if self.window_is_open():
            self._window_seen_open = True
            self._window_missing_streak = 0
        else:
            # QT reports -1 during create/move; only treat as user-X after the
            # window was stably open, and only after several consecutive misses.
            # If never seen open (e.g. destroyAllWindows mid-reset), recreate.
            self._window_missing_streak += 1
            if not self._window_seen_open:
                self._ensure_window()
            elif self._window_missing_streak >= 8:
                self.dismiss_window("title-bar X")

    def snapshot(self) -> HudSnapshot:
        with self._lock:
            return HudSnapshot(
                remaining_s=self._snap.remaining_s,
                reward_phi=self._snap.reward_phi,
                progress_curve=list(self._snap.progress_curve),
                step=self._snap.step,
                scored_at=self._snap.scored_at,
                latency_s=self._snap.latency_s,
                error=self._snap.error,
                instruction=self._snap.instruction,
            )

    def last_overlay_rgb(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._last_overlay_bgr is None:
                return None
            import cv2

            return cv2.cvtColor(self._last_overlay_bgr, cv2.COLOR_BGR2RGB)

    def overlay_image_path(self) -> Optional[Path]:
        """Sibling JPEG path for eai workspace preview (``*_overlay.jpg``)."""
        if self.status_path is None:
            return None
        return self.status_path.with_name(self.status_path.stem + "_overlay.jpg")

    def _write_overlay_image(self, overlay_bgr: Optional[np.ndarray]) -> None:
        """Atomically write BGR overlay JPEG next to status_path for GUI embedding."""
        path = self.overlay_image_path()
        if path is None or overlay_bgr is None:
            return
        now = time.time()
        if now - self._last_overlay_write_at < self._overlay_write_min_interval_s:
            return
        self._last_overlay_write_at = now
        try:
            import cv2

            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            ok = cv2.imwrite(str(tmp), overlay_bgr)
            if ok:
                tmp.replace(path)
            elif tmp.is_file():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("overlay write failed: %s", exc)

    def _write_status(self, snap: HudSnapshot) -> None:
        if self.status_path is None:
            return
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path = self.overlay_image_path()
            payload = {
                "remaining_s": snap.remaining_s,
                "reward_phi": snap.reward_phi,
                "step": snap.step,
                "scored_at": snap.scored_at,
                "latency_s": snap.latency_s,
                "error": snap.error,
                "instruction": snap.instruction,
                "progress_curve": snap.progress_curve,
                "server_url": self.server_url,
                "refresh_sec": self.refresh_sec,
                "overlay_path": str(overlay_path) if overlay_path else "",
            }
            tmp = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.status_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("status write failed: %s", exc)

    def _worker(self) -> None:
        next_due = time.time()
        while not self._stop.is_set():
            now = time.time()
            wait_s = max(0.05, next_due - now)
            self._wake.wait(timeout=wait_s)
            self._wake.clear()
            if self._stop.is_set() or self._window_dismissed:
                break
            now = time.time()
            if now < next_due:
                continue
            next_due = now + self.refresh_sec

            with self._lock:
                if (
                    not self._frames
                    or self._inflight
                    or self._window_dismissed
                    or self._stop.is_set()
                ):
                    continue
                frames_copy = list(self._frames)
                instruction = self.instruction
                step = self._step
                self._inflight = True
                self._dirty = False

            try:
                batch = _uniform_sample(frames_copy, self.num_frames)
                progress, latency = score_frames(
                    self.server_url,
                    batch,
                    instruction,
                    robot_description=self.robot_description,
                    camera_description=self.camera_description,
                    timeout_s=self.timeout_s,
                    use_frame_steps=self.use_frame_steps,
                )
                if self._stop.is_set() or self._window_dismissed:
                    break
                rem = float(progress[-1]) if progress.size else None
                snap = HudSnapshot(
                    remaining_s=rem,
                    reward_phi=(-rem if rem is not None else None),
                    progress_curve=[float(x) for x in progress.tolist()],
                    step=step,
                    scored_at=time.time(),
                    latency_s=float(latency),
                    error="",
                    instruction=instruction,
                )
                with self._lock:
                    self._snap = snap
                    if rem is not None:
                        self._history.append(rem)
                    display_rgb = self._last_display_rgb
                    history = list(self._history)
                self._write_status(snap)
                if display_rgb is not None:
                    overlay = draw_hud_overlay(display_rgb, snap, history=history)
                    with self._lock:
                        self._last_overlay_bgr = overlay
                    self._write_overlay_image(overlay)
                # Status is on the HUD window / status JSON. Do not print every score.
                logger.debug(
                    "Live HUD score step=%s rem=%s latency=%.2fs frames=%s",
                    step,
                    f"{rem:.3f}" if rem is not None else "n/a",
                    latency,
                    batch.shape[0],
                )
            except Exception as exc:  # noqa: BLE001
                if self._stop.is_set() or self._window_dismissed:
                    break
                err = f"{type(exc).__name__}: {exc}"
                snap = HudSnapshot(
                    remaining_s=None,
                    reward_phi=None,
                    progress_curve=[],
                    step=step,
                    scored_at=time.time(),
                    latency_s=0.0,
                    error=err[:200],
                    instruction=instruction,
                )
                with self._lock:
                    self._snap = snap
                self._write_status(snap)
                now = time.monotonic()
                if err != self._last_err_text or now - self._last_err_log_at >= 15.0:
                    self._last_err_text = err
                    self._last_err_log_at = now
                    logger.warning("Live HUD score failed: %s", err)
                else:
                    logger.debug("Live HUD score failed: %s", err)
            finally:
                with self._lock:
                    self._inflight = False


__all__ = [
    "DEFAULT_CAMERA_DESCRIPTION",
    "DEFAULT_ROBOT_DESCRIPTION",
    "DEFAULT_SERVER_URL",
    "HudSnapshot",
    "RynnValueLiveHud",
    "draw_hud_overlay",
    "score_frames",
]
