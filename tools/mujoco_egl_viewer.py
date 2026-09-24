#!/usr/bin/env python3
"""MuJoCo interactive viewer without GLFW/GLX.

Renders via MUJOCO_GL=egl and shows frames with Tkinter.
Supports body picking, mouse perturb, and actuator (ctrl) sliders.

Controls:
  Space           pause / resume
  R               reset
  I / Tab         toggle Camera ↔ Interact
  [ / ]           slower / faster
  Camera mode:    LMB orbit, RMB/wheel zoom, A/D W/S Z/X
  Interact mode:  LMB pick+drag translate, RMB rotate body (perturb)
  Ctrl+arrows     nudge selected actuator
  Q / Esc         quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tkinter as tk
from tkinter import font as tkfont
from typing import Dict, List, Optional, Tuple


def _body_name(model, body_id: int) -> str:
    import mujoco

    if body_id < 0:
        return "(none)"
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name or f"body_{body_id}"


def _actuators_for_body(model, body_id: int) -> List[int]:
    """Actuator indices that drive a joint belonging to body_id (or descendants)."""
    import mujoco

    if body_id <= 0:
        return []
    # Include descendant bodies so clicking a link can still find its joint actuator.
    bodies = {body_id}
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if b in bodies:
                continue
            parent = int(model.body_parentid[b])
            if parent in bodies:
                bodies.add(b)
                changed = True

    joint_ids = {
        j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in bodies
    }
    acts: List[int] = []
    for a in range(model.nu):
        # trntype joint → trnid[0] is joint id
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            jid = int(model.actuator_trnid[a, 0])
            if jid in joint_ids:
                acts.append(a)
    return acts


def _actuator_name(model, act_id: int) -> str:
    import mujoco

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
    return name or f"act_{act_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", required=True, help="path to MJCF/XML model")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()

    mjcf = os.path.abspath(args.mjcf)
    if not os.path.isfile(mjcf):
        print(f"[mujoco-egl] model not found: {mjcf}", file=sys.stderr)
        return 1

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")

    import mujoco
    import numpy as np
    from PIL import Image, ImageTk

    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    off_w = int(getattr(model.vis.global_, "offwidth", 640) or 640)
    off_h = int(getattr(model.vis.global_, "offheight", 480) or 480)
    width = max(64, min(int(args.width), off_w))
    height = max(64, min(int(args.height), off_h))
    if width != args.width or height != args.height:
        print(
            f"[mujoco-egl] clamp render size {args.width}x{args.height} "
            f"→ {width}x{height} (model offscreen {off_w}x{off_h})",
            flush=True,
        )

    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    mujoco.mj_forward(model, data)

    # ctrl limits
    ctrl_lo = np.array(model.actuator_ctrlrange[:, 0], dtype=np.float64, copy=True)
    ctrl_hi = np.array(model.actuator_ctrlrange[:, 1], dtype=np.float64, copy=True)
    for a in range(model.nu):
        if not (model.actuator_ctrllimited[a] or abs(ctrl_hi[a] - ctrl_lo[a]) > 1e-9):
            # fallback: use joint range if available
            if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
                jid = int(model.actuator_trnid[a, 0])
                if model.jnt_limited[jid]:
                    ctrl_lo[a] = float(model.jnt_range[jid, 0])
                    ctrl_hi[a] = float(model.jnt_range[jid, 1])
                else:
                    ctrl_lo[a], ctrl_hi[a] = -1.0, 1.0
            else:
                ctrl_lo[a], ctrl_hi[a] = -1.0, 1.0
        # avoid zero-span
        if abs(ctrl_hi[a] - ctrl_lo[a]) < 1e-9:
            ctrl_lo[a], ctrl_hi[a] = -1.0, 1.0

    root = tk.Tk()
    root.title(f"MuJoCo EGL — {os.path.basename(mjcf)}")
    root.geometry(f"{width + 280}x{height + 36}")

    status = tk.StringVar(value="starting…")
    tk.Label(root, textvariable=status, anchor="w", font=tkfont.Font(size=10)).pack(
        fill="x", padx=4, pady=2
    )

    body = tk.Frame(root)
    body.pack(fill="both", expand=True)
    panel = tk.Label(body, bd=0)
    panel.pack(side="left", fill="both", expand=True)

    side = tk.Frame(body, width=270)
    side.pack(side="right", fill="y")
    side.pack_propagate(False)
    tk.Label(side, text="交互 / 关节", font=tkfont.Font(size=11, weight="bold")).pack(
        anchor="w", padx=6, pady=(4, 2)
    )
    sel_var = tk.StringVar(value="选中: (none)  |  模式: Camera")
    tk.Label(side, textvariable=sel_var, wraplength=250, justify="left").pack(
        anchor="w", padx=6
    )
    tip = (
        "I/Tab 切换 Interact\n"
        "Interact: 左键点选并拖动物体\n"
        "右键旋转扰动\n"
        "下方滑条直接写 data.ctrl"
    )
    tk.Label(side, text=tip, justify="left", fg="#666").pack(anchor="w", padx=6, pady=4)

    slider_frame = tk.Frame(side)
    slider_frame.pack(fill="both", expand=True, padx=4, pady=4)
    slider_canvas = tk.Canvas(slider_frame, highlightthickness=0)
    slider_scroll = tk.Scrollbar(slider_frame, orient="vertical", command=slider_canvas.yview)
    slider_inner = tk.Frame(slider_canvas)
    slider_inner.bind(
        "<Configure>",
        lambda e: slider_canvas.configure(scrollregion=slider_canvas.bbox("all")),
    )
    slider_canvas.create_window((0, 0), window=slider_inner, anchor="nw")
    slider_canvas.configure(yscrollcommand=slider_scroll.set)
    slider_canvas.pack(side="left", fill="both", expand=True)
    slider_scroll.pack(side="right", fill="y")

    state = {
        "paused": False,
        "speed": 1.0,
        "alive": True,
        "photo": None,
        "mode": "camera",  # camera | interact
        "drag": None,  # dict
        "sel_body": -1,
        "sel_acts": [],  # actuator ids shown
        "act_vars": {},  # act_id -> DoubleVar
        "act_scales": {},
        "nudge_act": None,  # focused actuator for keyboard
        "dt_target": 1.0 / max(args.fps, 1.0),
        "aspect": float(width) / float(height),
        "view_w": width,
        "view_h": height,
    }

    def clear_sliders() -> None:
        for child in slider_inner.winfo_children():
            child.destroy()
        state["act_vars"].clear()
        state["act_scales"].clear()

    def rebuild_sliders(act_ids: List[int]) -> None:
        clear_sliders()
        state["sel_acts"] = list(act_ids)
        state["nudge_act"] = act_ids[0] if act_ids else None
        if not act_ids:
            tk.Label(slider_inner, text="(该部位无执行器)", fg="#888").pack(anchor="w")
            return
        for a in act_ids[:24]:
            lo, hi = float(ctrl_lo[a]), float(ctrl_hi[a])
            name = _actuator_name(model, a)
            row = tk.Frame(slider_inner)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=name[-28:], anchor="w", width=22).pack(side="left")
            var = tk.DoubleVar(value=float(data.ctrl[a]))
            state["act_vars"][a] = var

            def _on_slide(val: str, act: int = a, vlo: float = lo, vhi: float = hi) -> None:
                try:
                    v = float(val)
                except ValueError:
                    return
                data.ctrl[act] = float(np.clip(v, vlo, vhi))
                state["nudge_act"] = act

            scale = tk.Scale(
                row,
                from_=lo,
                to=hi,
                resolution=(hi - lo) / 200.0,
                orient="horizontal",
                variable=var,
                length=140,
                showvalue=True,
                command=_on_slide,
            )
            scale.pack(side="left", fill="x", expand=True)
            state["act_scales"][a] = scale

    def sync_sliders_from_ctrl() -> None:
        for a, var in state["act_vars"].items():
            try:
                cur = float(data.ctrl[a])
                if abs(var.get() - cur) > 1e-4:
                    var.set(cur)
            except tk.TclError:
                pass

    def set_mode(mode: str) -> None:
        state["mode"] = mode
        if mode != "interact":
            pert.active = 0
        update_sel_label()

    def update_sel_label() -> None:
        bname = _body_name(model, state["sel_body"])
        sel_var.set(f"选中: {bname}  |  模式: {state['mode'].title()}")

    def pick_body(x: int, y: int) -> int:
        """Return body id under pixel (x,y) in panel coords, or -1."""
        vw, vh = state["view_w"], state["view_h"]
        if vw <= 0 or vh <= 0:
            return -1
        # Map from label widget size to render buffer.
        pw = max(panel.winfo_width(), 1)
        ph = max(panel.winfo_height(), 1)
        relx = float(np.clip(x / pw, 0.0, 1.0))
        # MuJoCo rely is from bottom
        rely = float(np.clip(1.0 - (y / ph), 0.0, 1.0))
        renderer.update_scene(data, camera=cam)
        scn = renderer.scene
        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        flexid = np.zeros(1, dtype=np.int32)
        skinid = np.zeros(1, dtype=np.int32)
        body_id = int(
            mujoco.mjv_select(
                model,
                data,
                vopt,
                state["aspect"],
                relx,
                rely,
                scn,
                selpnt,
                geomid,
                flexid,
                skinid,
            )
        )
        return body_id

    def begin_perturb(body_id: int, translate: bool) -> None:
        if body_id <= 0:
            return
        pert.select = int(body_id)
        pert.skinselect = -1
        renderer.update_scene(data, camera=cam)
        mujoco.mjv_initPerturb(model, data, renderer.scene, pert)
        pert.active = int(
            mujoco.mjtPertBit.mjPERT_TRANSLATE
            if translate
            else mujoco.mjtPertBit.mjPERT_ROTATE
        )

    def render_frame() -> None:
        # Apply pose perturb when paused so dragging still moves the body.
        if state["paused"] and pert.active:
            mujoco.mjv_applyPerturbPose(model, data, pert, flg_paused=1)
            mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        img = Image.fromarray(rgb)
        # Resize to panel if needed for display only
        photo = ImageTk.PhotoImage(image=img)
        state["photo"] = photo
        panel.configure(image=photo)
        mode = "PAUSE" if state["paused"] else "RUN"
        bname = _body_name(model, state["sel_body"])
        status.set(
            f"{mode} x{state['speed']:.1f} t={data.time:.2f}s  "
            f"[{state['mode']}] sel={bname}  |  "
            "I interact  LMB pick/drag  RMB rotate  Space pause  Q quit"
        )
        update_sel_label()
        sync_sliders_from_ctrl()

    def on_key(event: tk.Event) -> None:
        key = (event.keysym or "").lower()
        char = event.char or ""
        if key in ("q", "escape"):
            state["alive"] = False
            root.destroy()
            return
        if key in ("i", "tab"):
            set_mode("interact" if state["mode"] == "camera" else "camera")
            return
        if key == "space" or char == " ":
            state["paused"] = not state["paused"]
        elif key == "r":
            mujoco.mj_resetData(model, data)
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)
            pert.active = 0
            rebuild_sliders(state["sel_acts"])
        elif char == "[":
            state["speed"] = max(0.25, state["speed"] * 0.5)
        elif char == "]":
            state["speed"] = min(16.0, state["speed"] * 2.0)
        elif state["mode"] == "camera":
            if key == "a":
                cam.azimuth += 3.0
            elif key == "d":
                cam.azimuth -= 3.0
            elif key == "w":
                cam.elevation = float(np.clip(cam.elevation + 2.0, -89, 89))
            elif key == "s":
                cam.elevation = float(np.clip(cam.elevation - 2.0, -89, 89))
            elif key == "z":
                cam.distance = float(np.clip(cam.distance * 0.9, 0.05, 100.0))
            elif key == "x":
                cam.distance = float(np.clip(cam.distance * 1.1, 0.05, 100.0))
        # Ctrl+Left/Right nudge selected actuator
        if event.state & 0x4:  # Control
            act = state.get("nudge_act")
            if act is not None and key in ("left", "right", "up", "down"):
                lo, hi = float(ctrl_lo[act]), float(ctrl_hi[act])
                step = (hi - lo) * 0.02
                delta = step if key in ("right", "up") else -step
                data.ctrl[act] = float(np.clip(data.ctrl[act] + delta, lo, hi))
                var = state["act_vars"].get(act)
                if var is not None:
                    var.set(float(data.ctrl[act]))

    def on_press(event: tk.Event) -> None:
        btn = int(getattr(event, "num", 1))
        x, y = int(event.x), int(event.y)
        if state["mode"] == "interact" and btn in (1, 3):
            body_id = pick_body(x, y)
            if body_id > 0:
                state["sel_body"] = body_id
                acts = _actuators_for_body(model, body_id)
                rebuild_sliders(acts)
                begin_perturb(body_id, translate=(btn == 1))
                state["drag"] = {
                    "kind": "perturb",
                    "btn": btn,
                    "x": x,
                    "y": y,
                    "action": (
                        mujoco.mjtMouse.mjMOUSE_MOVE_V
                        if btn == 1
                        else mujoco.mjtMouse.mjMOUSE_ROTATE_V
                    ),
                }
                update_sel_label()
                return
            pert.active = 0
        # camera drag
        state["drag"] = {"kind": "camera", "btn": btn, "x": x, "y": y}

    def on_release(_event: tk.Event) -> None:
        state["drag"] = None
        # keep perturb select but stop active drag force after release
        pert.active = 0

    def on_motion(event: tk.Event) -> None:
        drag = state["drag"]
        if not drag:
            return
        x, y = int(event.x), int(event.y)
        dx, dy = x - drag["x"], y - drag["y"]
        drag["x"], drag["y"] = x, y
        pw = max(panel.winfo_width(), 1)
        ph = max(panel.winfo_height(), 1)
        if drag["kind"] == "perturb" and pert.select > 0:
            # relative mouse motion in [-1,1] viewport fraction (MuJoCo convention)
            reldx = dx / float(pw)
            reldy = dy / float(ph)
            renderer.update_scene(data, camera=cam)
            mujoco.mjv_movePerturb(
                model,
                data,
                drag["action"],
                reldx,
                reldy,
                renderer.scene,
                pert,
            )
            return
        # camera
        btn = drag["btn"]
        if btn == 1:
            cam.azimuth -= dx * 0.3
            cam.elevation = float(np.clip(cam.elevation - dy * 0.3, -89, 89))
        elif btn in (2, 3):
            cam.distance = float(
                np.clip(cam.distance * (1.0 + dy * 0.005), 0.05, 100.0)
            )

    def on_wheel(event: tk.Event) -> None:
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            cam.distance = float(np.clip(cam.distance * 0.9, 0.05, 100.0))
        else:
            cam.distance = float(np.clip(cam.distance * 1.1, 0.05, 100.0))

    root.bind("<Key>", on_key)
    panel.bind("<ButtonPress-1>", on_press)
    panel.bind("<ButtonPress-2>", on_press)
    panel.bind("<ButtonPress-3>", on_press)
    panel.bind("<ButtonRelease-1>", on_release)
    panel.bind("<ButtonRelease-2>", on_release)
    panel.bind("<ButtonRelease-3>", on_release)
    panel.bind("<B1-Motion>", on_motion)
    panel.bind("<B2-Motion>", on_motion)
    panel.bind("<B3-Motion>", on_motion)
    panel.bind("<Button-4>", on_wheel)
    panel.bind("<Button-5>", on_wheel)
    root.bind("<MouseWheel>", on_wheel)
    panel.focus_set()
    root.after(0, lambda: panel.focus_force())

    # Default: show a few arm actuators so user can interact immediately.
    default_acts = list(range(min(model.nu, 12)))
    rebuild_sliders(default_acts)

    print(
        f"[mujoco-egl] {mjcf}  MUJOCO_GL={os.environ.get('MUJOCO_GL')}  "
        f"DISPLAY={os.environ.get('DISPLAY', '')}  backend=tkinter+perturb",
        flush=True,
    )
    print(
        "[mujoco-egl] I=Interact | LMB pick/drag | RMB rotate | sliders=ctrl | Q quit",
        flush=True,
    )

    def tick() -> None:
        if not state["alive"]:
            return
        t0 = time.perf_counter()
        try:
            if not state["paused"]:
                nsub = max(1, int(round(state["speed"])))
                for _ in range(nsub):
                    if pert.active:
                        mujoco.mjv_applyPerturbForce(model, data, pert)
                    mujoco.mj_step(model, data)
            render_frame()
        except tk.TclError:
            return
        except Exception as exc:
            print(f"[mujoco-egl] render error: {exc}", file=sys.stderr)
            state["alive"] = False
            try:
                root.destroy()
            except tk.TclError:
                pass
            return
        elapsed = time.perf_counter() - t0
        delay_ms = max(1, int((state["dt_target"] - elapsed) * 1000))
        root.after(delay_ms, tick)

    def on_close() -> None:
        state["alive"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(1, tick)
    try:
        root.mainloop()
    finally:
        try:
            renderer.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
