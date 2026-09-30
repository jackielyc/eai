#!/usr/bin/env python3
"""Cross-process RGB frame bus for RynnValue Live HUD.

Used when the sim step loop runs in another process (e.g. RLinf Ray
EnvWorker for RoboTwin) while the HUD / OpenCV window must stay on the
driver (or a sidecar) that has DISPLAY + HTTP to reward_server.

Protocol (directory ``bus_dir``)::

  meta.json   {"seq": int, "step": int, "instruction": str, "ts": float}
  latest.npy  HxWx3 uint8 RGB (atomic replace via *.tmp + rename)

Writer (env worker) calls ``FrameBusWriter.push``.
Reader (driver HUD) polls ``FrameBusReader.poll`` and feeds ``push_frame``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"unexpected frame shape: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        # torch images may be float 0-1 or 0-255
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        if mx <= 1.5:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class FrameBusWriter:
    def __init__(self, bus_dir: str | Path) -> None:
        self.bus_dir = Path(bus_dir)
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._lock = threading.Lock()

    def push(
        self,
        frame: np.ndarray,
        *,
        step: Optional[int] = None,
        instruction: str = "",
    ) -> int:
        rgb = _as_uint8_rgb(frame)
        with self._lock:
            self._seq += 1
            seq = self._seq
            meta = {
                "seq": seq,
                "step": int(step if step is not None else seq),
                "instruction": str(instruction or ""),
                "ts": time.time(),
            }
            npy_tmp = self.bus_dir / "latest.npy.tmp"
            npy_path = self.bus_dir / "latest.npy"
            meta_tmp = self.bus_dir / "meta.json.tmp"
            meta_path = self.bus_dir / "meta.json"
            # Use a file handle so np.save does not append another ".npy".
            with open(npy_tmp, "wb") as f:
                np.save(f, rgb)
            os.replace(npy_tmp, npy_path)
            meta_tmp.write_text(json.dumps(meta), encoding="utf-8")
            os.replace(meta_tmp, meta_path)
            return seq


class FrameBusReader:
    def __init__(self, bus_dir: str | Path) -> None:
        self.bus_dir = Path(bus_dir)
        self._last_seq = -1

    def poll(self) -> Optional[Tuple[np.ndarray, dict]]:
        meta_path = self.bus_dir / "meta.json"
        npy_path = self.bus_dir / "latest.npy"
        if not meta_path.is_file() or not npy_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            seq = int(meta.get("seq", -1))
            if seq <= self._last_seq:
                return None
            frame = np.load(npy_path)
            self._last_seq = seq
            return _as_uint8_rgb(frame), meta
        except Exception as exc:  # noqa: BLE001
            logger.debug("frame bus poll failed: %s", exc)
            return None


class FrameBusHudSidecar:
    """Poll frame bus → RynnValueLiveHud (runs on driver / DISPLAY host)."""

    def __init__(
        self,
        bus_dir: str | Path,
        *,
        server_url: str = "http://127.0.0.1:8001",
        refresh_sec: float = 1.0,
        num_frames: int = 8,
        robot_description: str = "",
        camera_description: str = "",
        show_window: bool = True,
        status_path: Optional[str | Path] = None,
        poll_sec: float = 0.1,
        timeout_s: float = 60.0,
    ) -> None:
        from rynnvalue_live_hud import RynnValueLiveHud

        self.reader = FrameBusReader(bus_dir)
        self.poll_sec = max(0.05, float(poll_sec))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.hud = RynnValueLiveHud(
            server_url=server_url,
            refresh_sec=refresh_sec,
            num_frames=num_frames,
            robot_description=robot_description,
            camera_description=camera_description,
            show_window=show_window,
            status_path=str(status_path) if status_path else None,
            timeout_s=timeout_s,
        )
        self._last_instruction = ""

    def start(self) -> None:
        self.hud.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="rynnvalue-frame-bus-hud", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self.hud.stop()

    def _loop(self) -> None:
        while not self._stop.is_set():
            got = self.reader.poll()
            if got is not None:
                frame, meta = got
                instr = str(meta.get("instruction") or "")
                if instr and instr != self._last_instruction:
                    self.hud.reset_episode(instr)
                    self._last_instruction = instr
                step = meta.get("step")
                self.hud.push_frame(frame, step=int(step) if step is not None else None)
            self._stop.wait(self.poll_sec)


def install_robotwin_env_frame_hook(
    writer: FrameBusWriter,
    *,
    bridge: Any = None,
) -> bool:
    """Monkeypatch ``RobotwinEnv._extract_obs_image`` to publish main cam frames.

    Must run inside the EnvWorker process (before / during env setup).
    Returns True if patch applied.
    """
    try:
        from rlinf.envs.robotwin.robotwin_env import RobotwinEnv
    except Exception as exc:  # noqa: BLE001
        logger.warning("RobotwinEnv import failed for HUD hook: %s", exc)
        return False

    if bridge is None:
        from rynnvalue_sim_bridge import RoboTwinHudBridge

        bridge = RoboTwinHudBridge()

    if getattr(RobotwinEnv, "_rynnvalue_frame_hook_installed", False):
        return True

    orig = RobotwinEnv._extract_obs_image

    def _wrapped(self, raw_obs):  # type: ignore[no-untyped-def]
        out = orig(self, raw_obs)
        try:
            images = out.get("main_images")
            instrs = out.get("task_descriptions") or []
            if images is None:
                return out
            if hasattr(images, "detach"):
                img0 = images[0].detach().cpu().numpy()
            else:
                img0 = np.asarray(images[0])
            instr = ""
            if instrs:
                item = instrs[0]
                instr = item if isinstance(item, str) else str(item)
            rgb = bridge.extract_rgb(img0)
            writer.push(rgb, instruction=bridge.instruction(instr))
        except Exception as exc:  # noqa: BLE001
            logger.debug("frame bus push skipped: %s", exc)
        return out

    RobotwinEnv._extract_obs_image = _wrapped  # type: ignore[method-assign]
    RobotwinEnv._rynnvalue_frame_hook_installed = True  # type: ignore[attr-defined]
    logger.info("RoboTwin frame-bus HUD hook installed → %s", writer.bus_dir)
    return True


def maybe_install_robotwin_hook_from_env() -> bool:
    """Install writer hook when ``RYNNVALUE_FRAME_BUS`` is set."""
    bus = (os.environ.get("RYNNVALUE_FRAME_BUS") or "").strip()
    if not bus:
        return False
    return install_robotwin_env_frame_hook(FrameBusWriter(bus))


__all__ = [
    "FrameBusHudSidecar",
    "FrameBusReader",
    "FrameBusWriter",
    "install_robotwin_env_frame_hook",
    "maybe_install_robotwin_hook_from_env",
]
