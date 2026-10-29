"""Small geometry helpers for the synthetic dataset (camera convention: x right, y down, z forward; world z up)."""
import numpy as np

# phone rear camera in portrait (web/slamclient.js phoneImuSettings, record.js): camera -> body (device axes)
R_BC = np.diag([1.0, -1.0, -1.0])
G_W = np.array([0.0, 0.0, -9.81])  # gravity in the world (z up)


def look_rotation(fwd, roll=0.0, up=(0.0, 0.0, 1.0)):
    """camera -> world rotation for a forward direction (world), rolled about the optical axis by `roll` rad.
    Columns: camera x (right), y (down), z (forward)."""
    f = np.asarray(fwd, float); f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        r, d = c * r + s * d, -s * r + c * d
    return np.stack([r, d, f], 1)


def ypr_rotation(yaw, pitch, roll):
    """camera -> world from yaw (about world z, 0 = looking along +x), pitch (+ = up) and roll (about the optical
    axis); arrays broadcast -> (..., 3, 3)"""
    yaw, pitch, roll = np.broadcast_arrays(np.asarray(yaw, float), np.asarray(pitch, float), np.asarray(roll, float))
    cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    f = np.stack([cp * cy, cp * sy, sp], -1)
    r = np.stack([sy, -cy, np.zeros_like(sy)], -1)          # f x up, normalised (cos(pitch) > 0)
    d = np.cross(f, r)
    c, s = np.cos(roll)[..., None], np.sin(roll)[..., None]
    r2, d2 = c * r + s * d, -s * r + c * d
    return np.stack([r2, d2, f], -1)


def so3_log(R):
    """rotation matrices (..., 3, 3) -> rotation vectors (..., 3)"""
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(tr)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    k = np.where(s > 1e-9, th / (2 * np.maximum(s, 1e-12)), 0.5 + th ** 2 / 12)
    return w * k[..., None]


def quat_from_R(R):
    """(..., 3, 3) -> quaternions (..., 4) as x, y, z, w (scipy / TUM order), w >= 0"""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat()
    q[q[:, 3] < 0] *= -1
    return q.reshape(R.shape[:-2] + (4,))
