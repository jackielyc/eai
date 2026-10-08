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
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_SIDECAR_STOP_NAME = "sidecar.stop"


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


def run_frame_bus_hud_loop(
    bus_dir: str | Path,
    *,
    server_url: str = "http://127.0.0.1:8001",
    refresh_sec: float = 1.0,
    num_frames: int = 8,
    robot_description: str = "",
    camera_description: str = "",
    show_window: bool = True,
    status_path: Optional[str | Path] = None,
    poll_sec: float = 0.05,
    timeout_s: float = 60.0,
    flip_ud: bool = False,
) -> None:
    """Blocking Live HUD on *this* thread (OpenCV-safe).

    Stops when ``bus_dir/sidecar.stop`` appears or the process receives SIGTERM.
    """
    from rynnvalue_live_hud import RynnValueLiveHud

    bus = Path(bus_dir)
    bus.mkdir(parents=True, exist_ok=True)
    stop_path = bus / _SIDECAR_STOP_NAME
    if stop_path.is_file():
        try:
            stop_path.unlink()
        except OSError:
            pass

    reader = FrameBusReader(bus)
    hud = RynnValueLiveHud(
        server_url=server_url,
        refresh_sec=refresh_sec,
        num_frames=num_frames,
        robot_description=robot_description,
        camera_description=camera_description,
        show_window=show_window,
        status_path=str(status_path) if status_path else None,
        timeout_s=timeout_s,
        flip_ud=flip_ud,
    )
    hud.start()
    last_instruction = ""
    print(
        f"[robotwin] Live HUD main-thread loop bus={bus} "
        f"show_window={show_window} url={hud.server_url}",
        flush=True,
    )
    try:
        while not stop_path.is_file():
            if hud.user_closed_window():
                print(
                    "[robotwin] Live HUD window closed by user; exiting sidecar",
                    flush=True,
                )
                break
            got = reader.poll()
            if got is not None:
                frame, meta = got
                instr = str(meta.get("instruction") or "")
                if instr and instr != last_instruction:
                    hud.reset_episode(instr)
                    last_instruction = instr
                step = meta.get("step")
                hud.push_frame(frame, step=int(step) if step is not None else None)
            else:
                hud.pump_window()
            if hud.user_closed_window():
                print(
                    "[robotwin] Live HUD window closed by user; exiting sidecar",
                    flush=True,
                )
                break
            time.sleep(max(0.02, float(poll_sec)))
    finally:
        hud.stop()


class FrameBusHudSidecar:
    """Launch Live HUD in a *child process* so OpenCV owns that main thread.

    In-process background threads often fail to show Qt HighGUI windows under
    Ray / VGL; a dedicated process keeps ``imshow`` / ``waitKey`` reliable.
    """

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
        poll_sec: float = 0.05,
        timeout_s: float = 60.0,
        flip_ud: bool = False,
    ) -> None:
        self.bus_dir = Path(bus_dir)
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self.server_url = server_url
        self.refresh_sec = refresh_sec
        self.num_frames = num_frames
        self.robot_description = robot_description
        self.camera_description = camera_description
        self.show_window = show_window
        self.status_path = Path(status_path) if status_path else None
        self.poll_sec = poll_sec
        self.timeout_s = timeout_s
        self.flip_ud = flip_ud
        self._proc: Optional[subprocess.Popen] = None
        # Compatibility for callers that read ``sidecar.hud.server_url``.
        self.hud = type("HudUrl", (), {"server_url": server_url})()

    def _stop_file(self) -> Path:
        return self.bus_dir / _SIDECAR_STOP_NAME

    def start(self) -> None:
        self.stop()
        stop = self._stop_file()
        if stop.is_file():
            try:
                stop.unlink()
            except OSError:
                pass
        env = dict(os.environ)
        # Prefer the same interpreter (RLinf) so OpenCV/Qt plugins match.
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "sidecar",
            f"--bus={self.bus_dir}",
            f"--server_url={self.server_url}",
            f"--refresh_sec={self.refresh_sec}",
            f"--num_frames={self.num_frames}",
            f"--poll_sec={self.poll_sec}",
            f"--timeout_s={self.timeout_s}",
            f"--robot_description={self.robot_description}",
            f"--camera_description={self.camera_description}",
            "--show_window" if self.show_window else "--no_show_window",
            "--flip_ud" if self.flip_ud else "--no_flip_ud",
        ]
        if self.status_path is not None:
            cmd.append(f"--status_path={self.status_path}")
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )
        logger.info(
            "FrameBusHudSidecar process pid=%s bus=%s",
            self._proc.pid,
            self.bus_dir,
        )

    def stop(self) -> None:
        stop = self._stop_file()
        try:
            stop.write_text("1", encoding="utf-8")
        except OSError:
            pass
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.wait(timeout=3.0)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        try:
            if stop.is_file():
                stop.unlink()
        except OSError:
            pass


def _resolve_robotwin_env_cls() -> Any:
    """Return RoboTwin env class (RLinf renamed RobotwinEnv → RoboTwinEnv)."""
    try:
        from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

        return RoboTwinEnv
    except Exception:
        pass
    try:
        from rlinf.envs.robotwin.robotwin_env import RobotwinEnv

        return RobotwinEnv
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            f"cannot import RoboTwinEnv/RobotwinEnv from robotwin_env: {exc}"
        ) from exc


def push_obs_to_frame_bus(
    writer: FrameBusWriter,
    obs: Any,
    *,
    bridge: Any = None,
    step: Optional[int] = None,
) -> bool:
    """Publish one RGB frame from an extracted RoboTwin obs dict. Returns True on write."""
    if obs is None or not isinstance(obs, dict):
        return False
    if bridge is None:
        from rynnvalue_sim_bridge import RoboTwinHudBridge

        bridge = RoboTwinHudBridge()
    try:
        images = obs.get("main_images")
        if images is None:
            # Some wrappers nest under "obs"
            nested = obs.get("obs")
            if isinstance(nested, dict):
                images = nested.get("main_images")
                instrs = nested.get("task_descriptions") or []
            else:
                return False
        else:
            instrs = obs.get("task_descriptions") or []
        if images is None:
            return False
        if hasattr(images, "detach"):
            img0 = images[0].detach().cpu().numpy()
        else:
            arr = np.asarray(images)
            img0 = arr[0] if arr.ndim >= 4 else arr
        instr = ""
        if instrs:
            item = instrs[0]
            instr = item if isinstance(item, str) else str(item)
        rgb = bridge.extract_rgb(img0)
        writer.push(rgb, step=step, instruction=bridge.instruction(instr))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("frame bus push skipped: %s", exc)
        return False


def install_robotwin_env_frame_hook(
    writer: FrameBusWriter,
    *,
    bridge: Any = None,
) -> bool:
    """Monkeypatch ``RoboTwinEnv._extract_obs_image`` to publish main cam frames.

    Must run inside the EnvWorker process (before / during env setup).
    Returns True if patch applied.
    """
    try:
        env_cls = _resolve_robotwin_env_cls()
    except Exception as exc:  # noqa: BLE001
        msg = f"RoboTwinEnv import failed for HUD hook: {exc}"
        logger.warning(msg)
        print(f"[robotwin] {msg}", flush=True)
        return False

    if bridge is None:
        from rynnvalue_sim_bridge import RoboTwinHudBridge

        bridge = RoboTwinHudBridge()

    if getattr(env_cls, "_rynnvalue_frame_hook_installed", False):
        return True

    orig = env_cls._extract_obs_image

    def _wrapped(self, raw_obs):  # type: ignore[no-untyped-def]
        out = orig(self, raw_obs)
        push_obs_to_frame_bus(writer, out, bridge=bridge)
        return out

    env_cls._extract_obs_image = _wrapped  # type: ignore[method-assign]
    env_cls._rynnvalue_frame_hook_installed = True  # type: ignore[attr-defined]
    msg = f"RoboTwin frame-bus HUD hook installed → {writer.bus_dir}"
    logger.info(msg)
    print(f"[robotwin] {msg}", flush=True)
    return True


def maybe_install_robotwin_hook_from_env() -> bool:
    """Install writer hook when ``RYNNVALUE_FRAME_BUS`` is set."""
    bus = (os.environ.get("RYNNVALUE_FRAME_BUS") or "").strip()
    if not bus:
        return False
    return install_robotwin_env_frame_hook(FrameBusWriter(bus))

def _cli_main(argv: Optional[list] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="RynnValue frame-bus utilities")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sidecar", help="OpenCV Live HUD (main-thread process)")
    sp.add_argument("--bus", required=True)
    sp.add_argument("--server_url", default="http://127.0.0.1:8001")
    sp.add_argument("--refresh_sec", type=float, default=1.0)
    sp.add_argument("--num_frames", type=int, default=8)
    sp.add_argument("--poll_sec", type=float, default=0.05)
    sp.add_argument("--timeout_s", type=float, default=60.0)
    sp.add_argument("--robot_description", default="")
    sp.add_argument("--camera_description", default="")
    sp.add_argument("--status_path", default="")
    sp.add_argument("--show_window", action="store_true", default=True)
    sp.add_argument("--no_show_window", action="store_false", dest="show_window")
    sp.add_argument("--flip_ud", action="store_true", default=False)
    sp.add_argument("--no_flip_ud", action="store_false", dest="flip_ud")
    args = p.parse_args(argv)
    if args.cmd == "sidecar":
        run_frame_bus_hud_loop(
            args.bus,
            server_url=args.server_url,
            refresh_sec=args.refresh_sec,
            num_frames=args.num_frames,
            robot_description=args.robot_description,
            camera_description=args.camera_description,
            show_window=bool(args.show_window),
            status_path=args.status_path or None,
            poll_sec=args.poll_sec,
            timeout_s=args.timeout_s,
            flip_ud=bool(args.flip_ud),
        )
        return 0
    return 2


__all__ = [
    "FrameBusHudSidecar",
    "FrameBusReader",
    "FrameBusWriter",
    "install_robotwin_env_frame_hook",
    "maybe_install_robotwin_hook_from_env",
    "push_obs_to_frame_bus",
    "run_frame_bus_hud_loop",
]


if __name__ == "__main__":
    raise SystemExit(_cli_main())
