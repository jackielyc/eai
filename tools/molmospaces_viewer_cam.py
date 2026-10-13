"""Apply MolmoSpaces viewer camera (first-person via camera_manager).

MolmoSpaces robot-mounted cameras (e.g. ``head_camera_rgb``) live in
``env.camera_manager`` — they are **not** MuJoCo XML cameras.

MuJoCo free-cam (verified empirically):
  look direction f = [cos(el)*cos(az), cos(el)*sin(az), sin(el)]
  camera sits at lookat - distance * f  (looking toward lookat)
so:
  az = atan2(f_y, f_x),  el = asin(f_z)
"""
from __future__ import annotations

import math
import os
import weakref
from typing import Any, Mapping, Optional, Sequence, Tuple


def _find_robot_body_id(model: Any) -> int:
    import mujoco

    prefer = (
        "robot_0/head_link2",
        "robot_0/body_link4",
        "robot_0/body_link1",
        "robot_0/base",
        "head_link2",
        "body_link4",
        "body_link1",
        "base",
    )
    for name in prefer:
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            bid = -1
        if bid >= 0:
            return int(bid)
    return -1


def _look_dir_to_az_el(forward: Sequence[float]) -> Tuple[float, float]:
    """Convert unit look direction → (azimuth_deg, elevation_deg) for MuJoCo free-cam."""
    import numpy as np

    f = np.asarray(forward, dtype=float).reshape(3)
    n = float(np.linalg.norm(f))
    if n < 1e-8:
        return 90.0, -20.0
    f = f / n
    elev = math.degrees(math.asin(float(np.clip(f[2], -1.0, 1.0))))
    azim = math.degrees(math.atan2(float(f[1]), float(f[0])))
    return float(azim), float(elev)


def _apply_free_cam(
    viewer: Any,
    *,
    eye: Sequence[float],
    lookat: Sequence[float],
) -> str:
    """Place free-cam so the optical center is at ``eye``, looking at ``lookat``."""
    import mujoco
    import numpy as np

    e = np.asarray(eye, dtype=float).reshape(3)
    la = np.asarray(lookat, dtype=float).reshape(3)
    delta = la - e
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:
        dist = 1.0
        delta = np.array([1.0, 0.0, 0.0])
    fwd = delta / dist
    azim, elev = _look_dir_to_az_el(fwd)

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1
    viewer.cam.trackbodyid = -1
    viewer.cam.distance = dist
    viewer.cam.azimuth = azim
    viewer.cam.elevation = elev
    try:
        viewer.cam.lookat[:] = la.tolist()
    except Exception:
        for i in range(3):
            viewer.cam.lookat[i] = float(la[i])
    return (
        f"free eye={[round(float(x), 3) for x in e]} "
        f"lookat={[round(float(x), 3) for x in la]} "
        f"dist={dist:.2f} az={azim:.1f} el={elev:.1f}"
    )


def _resolve_managed_camera(env: Any, candidates: Sequence[str]) -> Tuple[Optional[Any], str]:
    if env is None:
        return None, ""
    try:
        cams = env.camera_manager.registry.cameras
    except Exception:
        return None, ""
    for name in candidates:
        if name and name in cams:
            cam = cams[name]
            try:
                cam.update_pose(env)
            except Exception:
                pass
            return cam, name
    return None, ""


_HEAD_BODY_SUBSTR = (
    "head_link",
    "head_rgb",
    "head_rgbd",
    "head_fisheye",
    "fisheye_camera",
)


def _hide_head_geoms_keep_hands(mj_model: Any) -> int:
    """Make head / onboard-cam meshes transparent; keep arms & hands visible.

    Returns number of geoms made transparent (0 if already done / unavailable).
    """
    import mujoco

    if mj_model is None:
        return 0
    if getattr(mj_model, "_eai_fp_head_hidden", False):
        return 0
    n = 0
    for gid in range(int(mj_model.ngeom)):
        try:
            bid = int(mj_model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            gname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        except Exception:
            continue
        low = f"{bname} {gname}".lower()
        if any(s in low for s in _HEAD_BODY_SUBSTR):
            mj_model.geom_rgba[gid][3] = 0.0
            n += 1
    try:
        mj_model._eai_fp_head_hidden = True  # type: ignore[attr-defined]
    except Exception:
        pass
    return n


def _reduce_fp_self_occlusion(viewer: Any, mj_model: Any = None) -> str:
    """Keep hands/arms visible; only clear head shell + collision hulls."""
    bits = []
    if viewer is not None and hasattr(viewer, "opt"):
        try:
            # Robot visuals are group=1 — must stay ON so hands are visible.
            viewer.opt.geomgroup[1] = 1
            viewer.opt.geomgroup[3] = 0  # collision meshes
            viewer.opt.sitegroup[0] = 0
            bits.append("geom1=on,geom3=off")
        except Exception:
            pass
    hidden = _hide_head_geoms_keep_hands(mj_model)
    if hidden:
        bits.append(f"head_geoms_alpha0={hidden}")
    return ",".join(bits) or "none"


def apply_managed_camera_to_viewer(
    viewer: Any,
    env: Any,
    cam_name: str,
    *,
    fallbacks: Optional[Sequence[str]] = None,
    # Pulled-back "over-shoulder" egocentric view: far enough to see both
    # hands and the tabletop target (tight head FP was too close).
    look_ahead: float = 1.8,
    back_out: float = 1.1,
    up_nudge: float = 0.35,
    look_down: float = 0.55,
) -> str:
    """Egocentric free-cam from camera_manager, framed farther back.

    Eye sits ~1m behind / above the head mount so arms & hands stay in view.
    """
    import numpy as np

    if viewer is None:
        return "none"
    candidates = []
    for n in (cam_name, *(fallbacks or ())):
        s = str(n or "").strip()
        if s and s not in candidates:
            candidates.append(s)
    cam, used = _resolve_managed_camera(env, candidates)
    if cam is None:
        keys = []
        try:
            keys = list(env.camera_manager.registry.cameras.keys())
        except Exception:
            pass
        return f"managed_miss candidates={candidates} have={keys}"

    pos = np.asarray(cam.pos, dtype=float).reshape(3)
    fwd = np.asarray(cam.forward, dtype=float).reshape(3)
    up = np.asarray(getattr(cam, "up", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
    fn = float(np.linalg.norm(fwd))
    un = float(np.linalg.norm(up))
    if fn < 1e-8:
        fwd = np.array([1.0, 0.0, 0.0])
    else:
        fwd = fwd / fn
    if un < 1e-8:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = up / un

    # Behind + above the head, looking forward/down at hands & workspace.
    eye = pos - fwd * back_out + up * up_nudge
    lookat = pos + fwd * look_ahead - np.array([0.0, 0.0, look_down])
    mj_model = None
    try:
        mj_model = env.current_model
    except Exception:
        try:
            mj_model = env.mj_datas[0].model
        except Exception:
            mj_model = None
    occ = _reduce_fp_self_occlusion(viewer, mj_model)
    detail = _apply_free_cam(viewer, eye=eye, lookat=lookat)
    return f"fp:{used} {detail} occ=[{occ}]"


def _hud_enabled() -> bool:
    return os.environ.get("RYNNVALUE_LIVE_HUD", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
        "YES",
    )


def _push_env_rgb_to_hud(env: Any) -> None:
    """Render one MolmoSpaces camera frame into RynnValue Live HUD.

    Official GLFW ``launch_passive`` does not go through EGL+Tk, so nothing
    calls ``push_rgb`` unless we hook it here.
    """
    if env is None or not _hud_enabled():
        return
    try:
        from rynnvalue_hud_attach import push_rgb
    except Exception:
        return
    # Match the Psi GLFW viewer: first-person head cam (same as MOLMOSPACES_VIEWER_CAM).
    prefer = (
        os.environ.get("RYNNVALUE_HUD_CAMERA")
        or os.environ.get("MOLMOSPACES_VIEWER_CAM")
        or ""
    ).strip()
    candidates = []
    for name in (prefer, "head_camera_rgb", "head", "exo_camera_1"):
        if name and name not in candidates:
            candidates.append(name)
    rgb = None
    used = ""
    for name in candidates:
        try:
            if hasattr(env, "render_rgb_frame"):
                rgb = env.render_rgb_frame(name)
                used = name
                break
        except Exception:
            continue
    if rgb is None:
        return
    try:
        push_rgb(rgb, min_interval_s=float(os.environ.get("RYNNVALUE_REFRESH_SEC") or 0.5))
        if not getattr(_push_env_rgb_to_hud, "_logged", False):
            print(f"[viewer-cam] RynnValue Live HUD feeding camera={used}", flush=True)
            _push_env_rgb_to_hud._logged = True  # type: ignore[attr-defined]
    except Exception as exc:
        print(f"[viewer-cam] HUD push skipped: {exc}", flush=True)


def _ensure_hud_window() -> None:
    if not _hud_enabled():
        return
    try:
        from rynnvalue_hud_attach import ensure_started

        ensure_started()
        print("[viewer-cam] RynnValue Live HUD window ensure_started()", flush=True)
    except Exception as exc:
        print(f"[viewer-cam] Live HUD start failed: {exc}", flush=True)


def _bind_sync_refresh(
    viewer: Any,
    env: Any,
    cam_name: str,
    fallbacks: Sequence[str],
) -> None:
    if viewer is None or not hasattr(viewer, "sync"):
        return
    if getattr(viewer, "_eai_fp_cam_bound", False):
        return
    orig_sync = viewer.sync
    env_ref = weakref.ref(env) if env is not None else None
    names = tuple(fallbacks)
    _ensure_hud_window()

    def sync(*args, **kwargs):
        # Update cam *before* sync so this frame uses the new pose.
        e = env_ref() if env_ref is not None else None
        try:
            if e is not None and viewer.is_running():
                apply_managed_camera_to_viewer(
                    viewer, e, cam_name, fallbacks=names
                )
        except Exception:
            pass
        result = orig_sync(*args, **kwargs)
        try:
            if e is not None:
                _push_env_rgb_to_hud(e)
        except Exception:
            pass
        return result

    viewer.sync = sync  # type: ignore[method-assign]
    viewer._eai_fp_cam_bound = True  # type: ignore[attr-defined]


def _bind_hud_only(viewer: Any, env: Any) -> None:
    """HUD frame push without first-person cam refresh (GLFW path)."""
    if viewer is None or not hasattr(viewer, "sync"):
        return
    if getattr(viewer, "_eai_hud_bound", False):
        return
    if not _hud_enabled():
        return
    orig_sync = viewer.sync
    env_ref = weakref.ref(env) if env is not None else None
    _ensure_hud_window()

    def sync(*args, **kwargs):
        result = orig_sync(*args, **kwargs)
        try:
            e = env_ref() if env_ref is not None else None
            if e is not None:
                _push_env_rgb_to_hud(e)
        except Exception:
            pass
        return result

    viewer.sync = sync  # type: ignore[method-assign]
    viewer._eai_hud_bound = True  # type: ignore[attr-defined]


def apply_viewer_cam_dict(
    viewer: Any,
    cam_dict: Optional[Mapping[str, Any]],
    *,
    mj_data: Any = None,
    env: Any = None,
) -> str:
    import mujoco

    if viewer is None or not cam_dict:
        return "none"

    if "camera" in cam_dict:
        primary = str(cam_dict.get("camera") or "").strip()
        fallbacks = [
            str(x).strip()
            for x in (cam_dict.get("camera_fallbacks") or ())
            if str(x).strip()
        ]
        # Prefer head, then exo (elevated robot-mounted — always sees the scene).
        if "exo_camera_1" not in fallbacks:
            fallbacks.append("exo_camera_1")
        status = apply_managed_camera_to_viewer(
            viewer, env, primary, fallbacks=fallbacks
        )
        if status.startswith("fp:"):
            _bind_sync_refresh(viewer, env, primary, fallbacks)
            return status
        print(f"[viewer-cam] first-person failed: {status}", flush=True)

    # Fallback: tracking third-person on robot body (never inside a wall).
    body_id = _find_robot_body_id(mj_data.model) if mj_data is not None else -1
    distance = float(cam_dict.get("distance", 4.0))
    elevation = float(cam_dict.get("elevation", -25.0))
    azimuth = float(cam_dict.get("azimuth", 90.0))
    if body_id >= 0 and mj_data is not None:
        import numpy as np

        xmat = np.array(mj_data.xmat[body_id], dtype=float).reshape(3, 3)
        # Body +X facing → MuJoCo look dir along +X needs az=0; front view from
        # +X side of robot uses look dir = -body_x → az = yaw+180.
        yaw = math.degrees(math.atan2(float(xmat[1, 0]), float(xmat[0, 0])))
        azimuth = yaw + 180.0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = int(body_id)
        viewer.cam.fixedcamid = -1
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    while azimuth > 180.0:
        azimuth -= 360.0
    while azimuth < -180.0:
        azimuth += 360.0
    viewer.cam.distance = distance
    viewer.cam.azimuth = azimuth
    viewer.cam.elevation = elevation
    # GLFW path still needs HUD frame push (EGL+Tk used to do this).
    _bind_hud_only(viewer, env)
    return f"track_fallback body={body_id} dist={distance} az={azimuth:.1f} el={elevation}"


def _strip_camera_key(exp_config: Any) -> Optional[dict]:
    """Remove ``camera`` so upstream setup_viewer won't KeyError on soft cams."""
    cam_dict = getattr(exp_config, "viewer_cam_dict", None)
    if not isinstance(cam_dict, dict):
        return None
    saved = dict(cam_dict)
    if "camera" in cam_dict:
        exp_config.viewer_cam_dict = {
            "mode": "front",
            "distance": 4.0,
            "azimuth": 90.0,
            "elevation": -25.0,
            "lookat": [0.0, 0.0, 0.7],
        }
    return saved


def patch_pipeline_viewer_cam() -> None:
    import molmo_spaces.data_generation.pipeline as pipeline

    orig_setup = pipeline.setup_viewer
    orig_scene = pipeline.setup_scene_viewer

    def setup_viewer(exp_config, task, policy, current_viewer):
        saved = _strip_camera_key(exp_config)
        viewer = orig_setup(exp_config, task, policy, current_viewer)
        if saved is not None:
            exp_config.viewer_cam_dict = saved
        cam_dict = saved or getattr(exp_config, "viewer_cam_dict", None)
        env = getattr(task, "env", None)
        mj_data = None
        try:
            mj_data = task.env.mj_datas[task.env.current_batch_index]
        except Exception:
            try:
                mj_data = task.env.current_data
            except Exception:
                mj_data = None
        status = apply_viewer_cam_dict(
            viewer, cam_dict, mj_data=mj_data, env=env
        )
        # If FP path did not bind sync (no camera key), still hook HUD for GLFW.
        if not getattr(viewer, "_eai_fp_cam_bound", False):
            _bind_hud_only(viewer, env)
        try:
            from rynnvalue_hud_attach import set_instruction

            instr = ""
            for attr in ("language_instruction", "language", "instruction"):
                val = getattr(task, attr, None)
                if isinstance(val, str) and val.strip():
                    instr = val.strip()
                    break
            if not instr:
                lang = getattr(task, "task_info", None) or {}
                if isinstance(lang, dict):
                    instr = str(
                        lang.get("task_description")
                        or lang.get("instruction")
                        or ""
                    ).strip()
            if instr:
                set_instruction(instr)
        except Exception:
            pass
        print(f"[viewer-cam] setup_viewer → {status}", flush=True)
        return viewer

    def setup_scene_viewer(exp_config, env, current_viewer):
        saved = _strip_camera_key(exp_config)
        viewer = orig_scene(exp_config, env, current_viewer)
        if saved is not None:
            exp_config.viewer_cam_dict = saved
        cam_dict = saved or getattr(exp_config, "viewer_cam_dict", None)
        mj_data = None
        try:
            mj_data = env.current_data
        except Exception:
            try:
                mj_data = env.mj_datas[0]
            except Exception:
                mj_data = None
        status = apply_viewer_cam_dict(
            viewer, cam_dict, mj_data=mj_data, env=env
        )
        if not getattr(viewer, "_eai_fp_cam_bound", False):
            _bind_hud_only(viewer, env)
        print(f"[viewer-cam] setup_scene_viewer → {status}", flush=True)
        return viewer

    pipeline.setup_viewer = setup_viewer  # type: ignore[assignment]
    pipeline.setup_scene_viewer = setup_scene_viewer  # type: ignore[assignment]
    print(
        "[viewer-cam] patched setup_viewer (FP + RynnValue HUD on GLFW)",
        flush=True,
    )
