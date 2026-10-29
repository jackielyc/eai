"""A2D joint ordering and FK; no GUI or camera dependencies."""
from __future__ import annotations
import os
from typing import Dict, Sequence
import numpy as np
from eai_app.paths import PROJECT_ROOT

JOINT_NAMES = [
    "joint_lift_body",
    "joint_body_pitch",
    "joint1_l",
    "joint2_l",
    "joint3_l",
    "joint4_l",
    "joint5_l",
    "joint6_l",
    "joint7_l",
    "joint1_r",
    "joint2_r",
    "joint3_r",
    "joint4_r",
    "joint5_r",
    "joint6_r",
    "joint7_r",
]

def _resolve_urdf() -> str:
    env = os.environ.get("EAI_HOST_FK_URDF", "").strip()
    here = str(PROJECT_ROOT)
    candidates = [
        env,
        os.path.normpath(
            os.path.join(
                here,
                "..",
                "a2d-tele",
                "modules",
                "control",
                "wbc",
                "assets",
                "A2D_NoHand",
                "A2D_NoHand_Flattened.urdf",
            )
        ),
        "/share_data/projects/mahjong/share/personal/liyichao/a2d-tele/modules/control/wbc/assets/A2D_NoHand/A2D_NoHand_Flattened.urdf",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("找不到 A2D_NoHand_Flattened.urdf")

class _Kinematics:
    def __init__(self, urdf_path: str) -> None:
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._q_idx = {}
        for name in JOINT_NAMES:
            jid = self.model.getJointId(name)
            self._q_idx[name] = self.model.joints[jid].idx_q
        self._frame = {}
        for fid in range(self.model.nframes):
            frame = self.model.frames[fid]
            self._frame[frame.name] = fid
        self.q = pin.neutral(self.model)

    def link_poses(self, q16: Sequence[float]) -> Dict[str, np.ndarray]:
        pin = self._pin
        for i, name in enumerate(JOINT_NAMES):
            if i < len(q16):
                self.q[self._q_idx[name]] = float(q16[i])
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        out: Dict[str, np.ndarray] = {}
        for name, fid in self._frame.items():
            out[name] = np.array(self.data.oMf[fid].homogeneous, dtype=np.float64)
        return out

