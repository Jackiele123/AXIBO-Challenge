"""MotionLib: loads a reference motion NPZ and provides per-env frame queries with
linear / SLERP interpolation.  Also computes BeyondMimic anchor frames and
body-relative states for observation and reward computation.

NPZ format (G1 12-DOF, 607 frames, fps=50):
  fps             : (1,)          int
  joint_names     : (12,)         str
  body_names      : (13,)         str
  joint_pos       : (607, 12)     float32
  joint_vel       : (607, 12)     float32
  body_pos_w      : (607, 13, 3)  float32  – world positions
  body_quat_w     : (607, 13, 4)  float32  – world quats [w,x,y,z]
  body_lin_vel_w  : (607, 13, 3)  float32
  body_ang_vel_w  : (607, 13, 3)  float32

Anchor frame (BeyondMimic):
  - anchor_pos  : pelvis XY projected to ground  (Z=0)
  - anchor_quat : yaw-only quaternion of pelvis  [w, 0, 0, z]
Both robot anchor and motion anchor are defined this way so that body poses
can be compared in a heading-normalised frame independent of global drift.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Standalone geometry helpers  (no Genesis dependency)
# ---------------------------------------------------------------------------

def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """SLERP between unit quaternions.  q shape [..., 4], t shape [...].
    Convention: [w, x, y, z].
    """
    dot = (q0 * q1).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    # Shortest-path: flip q1 if dot < 0
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()
    theta = torch.acos(dot.clamp(0.0, 1.0))        # [..., 1]
    sin_theta = torch.sin(theta)
    safe = sin_theta.abs() > 1e-6
    t_ = t.unsqueeze(-1)
    w0 = torch.where(safe, torch.sin((1 - t_) * theta) / sin_theta, 1.0 - t_)
    w1 = torch.where(safe, torch.sin(t_ * theta) / sin_theta, t_)
    return w0 * q0 + w1 * q1


def yaw_quat_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only quaternion [w, 0, 0, z] (normalised).
    q: [..., 4] in [w, x, y, z] convention.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    out = torch.zeros_like(q)
    out[..., 0] = torch.cos(half)   # w
    out[..., 3] = torch.sin(half)   # z
    return out


def rot_mat_2col(q: torch.Tensor) -> torch.Tensor:
    """First 2 columns of the rotation matrix from quaternion [w,x,y,z].
    Returns [..., 6]  (col-0 || col-1, each 3 elements).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    col0 = torch.stack([
        1 - 2*(y*y + z*z),
        2*(x*y + w*z),
        2*(x*z - w*y),
    ], dim=-1)
    col1 = torch.stack([
        2*(x*y - w*z),
        1 - 2*(x*x + z*z),
        2*(y*z + w*x),
    ], dim=-1)
    return torch.cat([col0, col1], dim=-1)


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angular error (radians) between two unit quaternions [..., 4]."""
    dot = (q1 * q2).sum(-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _rotate_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v [..., 3] by unit quaternions q [..., 4] [w,x,y,z]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    px = (1 - 2*(y*y + z*z)) * vx + 2*(x*y - w*z) * vy + 2*(x*z + w*y) * vz
    py = 2*(x*y + w*z) * vx + (1 - 2*(x*x + z*z)) * vy + 2*(y*z - w*x) * vz
    pz = 2*(x*z - w*y) * vx + 2*(y*z + w*x) * vy + (1 - 2*(x*x + y*y)) * vz
    return torch.stack([px, py, pz], dim=-1)


def _inv_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate = inverse for unit quaternions."""
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2, both [w,x,y,z]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def subtract_frame_transforms(
    parent_pos: torch.Tensor,         # [..., 3]
    parent_quat: torch.Tensor,        # [..., 4]
    child_pos: torch.Tensor,          # [..., 3]
    child_quat: torch.Tensor | None = None,  # [..., 4]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Express child position (and optionally orientation) in parent frame.
    Returns (pos_b, quat_b) where quat_b is None if child_quat is None.
    """
    inv_p = _inv_quat(parent_quat)
    pos_b = _rotate_by_quat(child_pos - parent_pos, inv_p)
    if child_quat is None:
        return pos_b, None
    quat_b = _quat_mul(inv_p, child_quat)
    return pos_b, quat_b


# ---------------------------------------------------------------------------
# MotionLib
# ---------------------------------------------------------------------------

class MotionLib:
    """Reference motion data for BeyondMimic-style tracking.

    Pre-computes anchor frames and body-relative tensors for the entire
    sequence so that get_frame() only needs to interpolate.
    """

    def __init__(self, npz_path: str, device: str | torch.device):
        self.device = device
        data = np.load(npz_path, allow_pickle=True)

        raw_fps = data["fps"]
        self.fps = float(raw_fps.item() if raw_fps.ndim == 0 else raw_fps[0])
        self.ref_dt = 1.0 / self.fps

        self.joint_names: list[str] = [str(n) for n in data["joint_names"]]
        self.body_names: list[str] = [str(n) for n in data["body_names"]]
        self.num_bodies = len(self.body_names)

        def t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr, dtype=torch.float32, device=device)

        self.joint_pos     = t(data["joint_pos"])       # (F, 12)
        self.joint_vel     = t(data["joint_vel"])       # (F, 12)
        self.body_pos_w    = t(data["body_pos_w"])      # (F, 13, 3)
        self.body_quat_w   = t(data["body_quat_w"])     # (F, 13, 4)
        self.body_lin_vel_w = t(data["body_lin_vel_w"]) # (F, 13, 3)
        self.body_ang_vel_w = t(data["body_ang_vel_w"]) # (F, 13, 3)

        self.num_frames = self.joint_pos.shape[0]
        self.duration = (self.num_frames - 1) * self.ref_dt

        # --- Pre-compute anchor frames for every reference frame ---
        pelvis_pos  = self.body_pos_w[:, 0, :]   # (F, 3)
        pelvis_quat = self.body_quat_w[:, 0, :]  # (F, 4)

        self._anchor_pos = pelvis_pos.clone()
        self._anchor_pos[:, 2] = 0.0                        # ground projection
        self._anchor_quat = yaw_quat_from_quat(pelvis_quat)  # (F, 4)

        # Pelvis velocity (anchor linear/angular vel = pelvis vel)
        self._anchor_lin_vel = self.body_lin_vel_w[:, 0, :]  # (F, 3)
        self._anchor_ang_vel = self.body_ang_vel_w[:, 0, :]  # (F, 3)

        # --- Body positions relative to anchor in anchor-yaw frame (F, 13, 3) ---
        self._body_pos_relative = self._batch_body_in_anchor(
            self.body_pos_w, self._anchor_pos, self._anchor_quat
        )

        # --- Body quats relative to anchor quat (F, 13, 4) ---
        self._body_quat_relative = self._batch_body_quat_in_anchor(
            self.body_quat_w, self._anchor_quat
        )

    # ------------------------------------------------------------------
    # Pre-computation helpers
    # ------------------------------------------------------------------

    def _batch_body_in_anchor(
        self,
        body_pos_w: torch.Tensor,   # (F, B, 3)
        anchor_pos: torch.Tensor,   # (F, 3)
        anchor_quat: torch.Tensor,  # (F, 4)
    ) -> torch.Tensor:
        """R_anchor^T @ (body_pos_w - anchor_pos)  for each frame and body."""
        F, B, _ = body_pos_w.shape
        delta = body_pos_w - anchor_pos[:, None, :]          # (F, B, 3)
        inv_aq = _inv_quat(anchor_quat)                       # (F, 4)
        inv_aq_exp = inv_aq[:, None, :].expand(F, B, 4).reshape(F * B, 4)
        return _rotate_by_quat(delta.reshape(F * B, 3), inv_aq_exp).reshape(F, B, 3)

    def _batch_body_quat_in_anchor(
        self,
        body_quat_w: torch.Tensor,  # (F, B, 4)
        anchor_quat: torch.Tensor,  # (F, 4)
    ) -> torch.Tensor:
        """inv_anchor * body_quat  for each frame and body."""
        F, B, _ = body_quat_w.shape
        inv_aq = _inv_quat(anchor_quat)                       # (F, 4)
        inv_aq_exp = inv_aq[:, None, :].expand(F, B, 4).reshape(F * B, 4)
        return _quat_mul(inv_aq_exp, body_quat_w.reshape(F * B, 4)).reshape(F, B, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self, times: torch.Tensor) -> dict[str, torch.Tensor]:
        """Interpolated reference data at arbitrary times in [0, duration].

        Args:
            times: (N,) tensor of query times (seconds).

        Returns dict with keys:
            joint_pos              (N, 12)
            joint_vel              (N, 12)
            body_pos_w             (N, 13, 3)   – world positions
            body_quat_w            (N, 13, 4)   – world quats
            body_lin_vel_w         (N, 13, 3)
            body_ang_vel_w         (N, 13, 3)
            anchor_pos_w           (N, 3)       – motion anchor position
            anchor_quat_w          (N, 4)       – motion anchor orientation
            anchor_lin_vel_w       (N, 3)
            anchor_ang_vel_w       (N, 3)
            body_pos_relative_w    (N, 13, 3)   – body pos in anchor-yaw frame
            body_quat_relative_w   (N, 13, 4)   – body quat rel. to anchor
        """
        times = times.clamp(0.0, self.duration)
        float_idx = times * self.fps
        idx_lo = float_idx.long().clamp(0, self.num_frames - 2)
        idx_hi = (idx_lo + 1).clamp(0, self.num_frames - 1)
        alpha = float_idx - idx_lo.float()                    # (N,)

        def lerp(tensor: torch.Tensor) -> torch.Tensor:
            lo = tensor[idx_lo]
            hi = tensor[idx_hi]
            return lo + alpha.reshape(-1, *([1] * (tensor.ndim - 1))) * (hi - lo)

        def slerp_nd(tensor: torch.Tensor) -> torch.Tensor:
            """SLERP for (F, ..., 4) tensors."""
            lo = tensor[idx_lo]   # (N, ..., 4)
            hi = tensor[idx_hi]
            shape = lo.shape
            N = shape[0]
            extra = shape[1:-1]
            n_q = 1
            for d in extra:
                n_q *= d
            t_exp = alpha[:, None].expand(N, n_q).reshape(N * n_q)
            result = quat_slerp(
                lo.reshape(N * n_q, 4),
                hi.reshape(N * n_q, 4),
                t_exp,
            )
            return result.reshape(shape)

        return {
            "joint_pos":             lerp(self.joint_pos),
            "joint_vel":             lerp(self.joint_vel),
            "body_pos_w":            lerp(self.body_pos_w),
            "body_quat_w":           slerp_nd(self.body_quat_w),
            "body_lin_vel_w":        lerp(self.body_lin_vel_w),
            "body_ang_vel_w":        lerp(self.body_ang_vel_w),
            "anchor_pos_w":          lerp(self._anchor_pos),
            "anchor_quat_w":         slerp_nd(self._anchor_quat),
            "anchor_lin_vel_w":      lerp(self._anchor_lin_vel),
            "anchor_ang_vel_w":      lerp(self._anchor_ang_vel),
            "body_pos_relative_w":   lerp(self._body_pos_relative),
            "body_quat_relative_w":  slerp_nd(self._body_quat_relative),
        }

    def get_init_state(self, frame_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exact reference state at integer frame indices (no interpolation).

        Returns dict with keys:
            base_pos     (N, 3)
            base_quat    (N, 4)
            base_lin_vel (N, 3)
            base_ang_vel (N, 3)
            joint_pos    (N, 12)
            joint_vel    (N, 12)
        """
        idx = frame_idx.long().clamp(0, self.num_frames - 1)
        return {
            "base_pos":     self.body_pos_w[idx, 0, :],
            "base_quat":    self.body_quat_w[idx, 0, :],
            "base_lin_vel": self.body_lin_vel_w[idx, 0, :],
            "base_ang_vel": self.body_ang_vel_w[idx, 0, :],
            "joint_pos":    self.joint_pos[idx],
            "joint_vel":    self.joint_vel[idx],
        }

    def build_joint_idx_map(self, robot_joint_names: list[str]) -> torch.LongTensor:
        """Permutation tensor: maps NPZ joint order → robot DOF order.

        Usage: joint_tensor[:, idx_map] reorders NPZ joints to match
        robot_joint_names (the order used by motors_dof_idx in the env).
        """
        name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        idx = []
        for name in robot_joint_names:
            if name not in name_to_idx:
                raise ValueError(
                    f"Joint '{name}' not found in motion lib. "
                    f"Available: {self.joint_names}"
                )
            idx.append(name_to_idx[name])
        return torch.tensor(idx, dtype=torch.long, device=self.device)
