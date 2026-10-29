"""Rigid-transform validation shared by perception and execution."""
import numpy as np

def validate_transform(value):
    T = np.asarray(value, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    if (not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-3)):
        raise ValueError("位姿必须是米制平移和有效右手系刚体旋转")
    return T
