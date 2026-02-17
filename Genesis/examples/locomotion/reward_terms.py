"""Modular reward terms for motion tracking and control."""

from abc import ABC, abstractmethod

import torch

import genesis as gs
from genesis.utils.geom import transform_quat_by_quat, inv_quat, transform_by_quat


class RewardTerm(ABC):
    """
    Abstract base class for reward terms.
    
    A reward term declares the names of tensors it needs via `required_keys`.
    The caller must supply these keys in the state dictionary.
    
    Returns:
        A tensor of shape (B,) where B is the batch size.
    """

    #: Ordered list of keys the term expects to find in the `state` dict.
    required_keys: tuple[str, ...] = ()

    def __init__(self, scale: float = 1.0, name: str | None = None):
        self.name = name or self.__class__.__name__
        self.scale = float(scale)

    def __call__(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the reward given state tensors.
        
        Args:
            state: Dictionary containing all required tensors
        
        Returns:
            Reward tensor of shape (B,)
        """
        tensors = [state[k] for k in self.required_keys]
        return self.scale * self._compute(*tensors)

    @abstractmethod
    def _compute(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Actual reward computation. Signature is fixed by `required_keys`."""


### ---- Penalty Terms ---- ###

class ActionRatePenalty(RewardTerm):
    """
    Penalize changes in actions between timesteps.
    
    Encourages smooth action sequences.
    
    Args:
        actions: Current actions (B, D)
        last_actions: Previous actions (B, D)
    """

    required_keys = ("actions", "last_actions")

    def _compute(self, actions: torch.Tensor, last_actions: torch.Tensor) -> torch.Tensor:
        return -torch.sum(torch.square(actions - last_actions), dim=-1)


class DofVelPenalty(RewardTerm):
    """
    Penalize high joint velocities.
    
    Args:
        dof_vel: Joint velocities (B, D)
    """

    required_keys = ("dof_vel",)

    def _compute(self, dof_vel: torch.Tensor) -> torch.Tensor:
        return -torch.sum(torch.square(dof_vel), dim=-1)


class TorquePenalty(RewardTerm):
    """
    Penalize high joint torques (if available).
    
    Args:
        torques: Joint torques (B, D)
    """

    required_keys = ("torques",)

    def _compute(self, torques: torch.Tensor) -> torch.Tensor:
        return -torch.sum(torch.square(torques), dim=-1)


### ---- Motion Tracking Rewards ---- ###

class DofPosReward(RewardTerm):
    """
    Reward DoF position tracking with deviation weighting.
    
    Args:
        dof_pos_error_weighted: Weighted DoF position error (B,)
        deviation_buf: Deviation buffer for adaptive weighting (B,)
    """

    required_keys = ("dof_pos_error_weighted", "deviation_buf")

    def _compute(
        self, dof_pos_error_weighted: torch.Tensor, deviation_buf: torch.Tensor
    ) -> torch.Tensor:
        return -dof_pos_error_weighted * deviation_buf


class DofVelReward(RewardTerm):
    """
    Reward DoF velocity tracking with deviation weighting.
    
    Args:
        dof_vel_error_weighted: Weighted DoF velocity error (B,)
        deviation_buf: Deviation buffer for adaptive weighting (B,)
    """

    required_keys = ("dof_vel_error_weighted", "deviation_buf")

    def _compute(
        self, dof_vel_error_weighted: torch.Tensor, deviation_buf: torch.Tensor
    ) -> torch.Tensor:
        return -dof_vel_error_weighted * deviation_buf


class BaseHeightReward(RewardTerm):
    """
    Reward base height tracking.
    
    Args:
        base_pos: Current base position (B, 3)
        ref_base_pos: Reference base position (B, 3)
    """

    required_keys = ("base_pos", "ref_base_pos")

    def _compute(self, base_pos: torch.Tensor, ref_base_pos: torch.Tensor) -> torch.Tensor:
        base_height_error = torch.square(base_pos[:, 2] - ref_base_pos[:, 2])
        return -base_height_error


class BaseQuatReward(RewardTerm):
    """
    Reward base quaternion/orientation tracking.
    
    Args:
        base_quat: Current base quaternion (B, 4)
        ref_base_quat: Reference base quaternion (B, 4)
    """

    required_keys = ("base_quat", "ref_base_quat")

    def _compute(self, base_quat: torch.Tensor, ref_base_quat: torch.Tensor) -> torch.Tensor:
        # Compute quaternion error using angle-axis
        from genesis.utils.geom import transform_quat_by_quat, inv_quat
        
        # q_error = q_ref^-1 * q_current
        ref_quat_inv = inv_quat(ref_base_quat)
        quat_error = transform_quat_by_quat(ref_quat_inv, base_quat)
        
        # Convert to angle (2 * arccos(w))
        angle_error = 2.0 * torch.acos(torch.clamp(quat_error[:, 0].abs(), -1.0, 1.0))
        return -(angle_error ** 2)


class BaseAngVelReward(RewardTerm):
    """
    Reward base angular velocity tracking.
    
    Args:
        base_ang_vel: Current base angular velocity (B, 3)
        ref_base_ang_vel: Reference base angular velocity (B, 3)
    """

    required_keys = ("base_ang_vel", "ref_base_ang_vel")

    def _compute(self, base_ang_vel: torch.Tensor, ref_base_ang_vel: torch.Tensor) -> torch.Tensor:
        base_ang_vel_error = torch.square(base_ang_vel - ref_base_ang_vel).sum(dim=-1)
        return -base_ang_vel_error


class TrackingLinkPosGlobalReward(RewardTerm):
    """
    Reward tracking link global position matching.
    
    Args:
        tracking_link_pos_global: Current tracking link positions (B, N, 3)
        ref_tracking_link_pos_global: Reference tracking link positions (B, N, 3)
        tracking_link_pos_global_weights: Per-link weights (N,)
    """

    required_keys = (
        "tracking_link_pos_global",
        "ref_tracking_link_pos_global",
        "tracking_link_pos_global_weights",
    )

    def _compute(
        self,
        tracking_link_pos_global: torch.Tensor,
        ref_tracking_link_pos_global: torch.Tensor,
        tracking_link_pos_global_weights: torch.Tensor,
    ) -> torch.Tensor:
        tracking_link_pos_error = (
            torch.square(tracking_link_pos_global - ref_tracking_link_pos_global).sum(dim=-1)
            * tracking_link_pos_global_weights[None, :]
        ).sum(dim=-1)
        return -tracking_link_pos_error


class TrackingLinkPosLocalReward(RewardTerm):
    """
    Reward tracking link local (yaw-aligned) position matching.
    
    Args:
        tracking_link_pos_local_yaw: Current tracking link positions in local frame (B, N, 3)
        ref_tracking_link_pos_local_yaw: Reference tracking link positions in local frame (B, N, 3)
        tracking_link_pos_local_weights: Per-link weights (N,)
        deviation_buf: Deviation buffer for adaptive weighting (B,)
    """

    required_keys = (
        "tracking_link_pos_local_yaw",
        "ref_tracking_link_pos_local_yaw",
        "tracking_link_pos_local_weights",
        "deviation_buf",
    )

    def _compute(
        self,
        tracking_link_pos_local_yaw: torch.Tensor,
        ref_tracking_link_pos_local_yaw: torch.Tensor,
        tracking_link_pos_local_weights: torch.Tensor,
        deviation_buf: torch.Tensor,
    ) -> torch.Tensor:
        tracking_link_pos_error = (
            torch.square(tracking_link_pos_local_yaw - ref_tracking_link_pos_local_yaw).sum(dim=-1)
            * tracking_link_pos_local_weights[None, :]
        ).sum(dim=-1)
        return -tracking_link_pos_error * deviation_buf


class TrackingLinkQuatReward(RewardTerm):
    """
    Reward tracking link quaternion/orientation matching.
    
    Args:
        tracking_link_quat_local_yaw: Current link quaternions in local frame (B, N, 4)
        ref_tracking_link_quat_local_yaw: Reference link quaternions in local frame (B, N, 4)
        tracking_link_quat_weights: Per-link weights (N,)
        deviation_buf: Deviation buffer for adaptive weighting (B,)
    """

    required_keys = (
        "tracking_link_quat_local_yaw",
        "ref_tracking_link_quat_local_yaw",
        "tracking_link_quat_weights",
        "deviation_buf",
    )

    def _compute(
        self,
        tracking_link_quat_local_yaw: torch.Tensor,
        ref_tracking_link_quat_local_yaw: torch.Tensor,
        tracking_link_quat_weights: torch.Tensor,
        deviation_buf: torch.Tensor,
    ) -> torch.Tensor:
        from genesis.utils.geom import transform_quat_by_quat, inv_quat
        
        # Compute quaternion error for each link
        B, N, _ = tracking_link_quat_local_yaw.shape
        ref_quat_inv = inv_quat(ref_tracking_link_quat_local_yaw.reshape(-1, 4))
        curr_quat = tracking_link_quat_local_yaw.reshape(-1, 4)
        quat_error = transform_quat_by_quat(ref_quat_inv, curr_quat).reshape(B, N, 4)
        
        # Convert to angle error
        angle_error = 2.0 * torch.acos(torch.clamp(quat_error[..., 0].abs(), -1.0, 1.0))
        tracking_link_quat_error = (angle_error * tracking_link_quat_weights[None, :]).sum(dim=-1)
        
        return -tracking_link_quat_error * deviation_buf


class TrackingLinkLinVelReward(RewardTerm):
    """
    Reward tracking link linear velocity matching.
    
    Args:
        tracking_link_lin_vel_global: Current link linear velocities (B, N, 3)
        ref_tracking_link_lin_vel_global: Reference link linear velocities (B, N, 3)
        deviation_buf: Deviation buffer for adaptive weighting (B,)
    """

    required_keys = (
        "tracking_link_lin_vel_global",
        "ref_tracking_link_lin_vel_global",
        "deviation_buf",
    )

    def _compute(
        self,
        tracking_link_lin_vel_global: torch.Tensor,
        ref_tracking_link_lin_vel_global: torch.Tensor,
        deviation_buf: torch.Tensor,
    ) -> torch.Tensor:
        tracking_link_lin_vel_error = torch.square(
            tracking_link_lin_vel_global - ref_tracking_link_lin_vel_global
        ).sum(dim=[-1, -2])
        return -tracking_link_lin_vel_error * deviation_buf

### ---- G1-Specific Regularization Penalties ---- ###

class DofVelPenalty(RewardTerm):
    """
    Penalize high joint velocities (energy efficiency).
    
    Args:
        dof_vel: Joint velocities (B, D)
    """

    required_keys = ("dof_vel",)

    def _compute(self, dof_vel: torch.Tensor) -> torch.Tensor:
        return -torch.sum(torch.square(dof_vel), dim=-1)


class ActionLimitPenalty(RewardTerm):
    """
    Penalize actions near limits.
    
    Args:
        actions: Actions (B, D)
    """

    required_keys = ("actions",)

    def __init__(self, scale: float = 1.0, limit: float = 0.9, name: str | None = None):
        super().__init__(scale, name)
        self.limit = limit

    def _compute(self, actions: torch.Tensor) -> torch.Tensor:
        # Penalize actions exceeding limit threshold
        near_limit = (torch.abs(actions) > self.limit).float()
        return -torch.sum(near_limit * torch.square(actions), dim=-1)


class AnkleTorquePenalty(RewardTerm):
    """
    Penalize ankle joint torques specifically (indices 4, 5, 10, 11 for G1 12-DOF).
    
    Args:
        torques: Joint torques (B, D)
    """

    required_keys = ("torques",)

    def _compute(self, torques: torch.Tensor) -> torch.Tensor:
        # G1 12-DOF ankle indices: left_ankle_pitch(4), left_ankle_roll(5),
        # right_ankle_pitch(10), right_ankle_roll(11)
        return -torch.sum(torch.square(torques[:, [4, 5, 10, 11]]), dim=-1)


class BodyAngVelXYPenalty(RewardTerm):
    """
    Penalize body angular velocity in roll and pitch (XY components).
    
    Args:
        body_ang_vel: Body angular velocity (B, 3) or base_ang_vel
    """

    required_keys = ("base_ang_vel",)

    def _compute(self, base_ang_vel: torch.Tensor) -> torch.Tensor:
        return -torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)


class BodyRollPenalty(RewardTerm):
    """
    Penalize body roll angle.
    
    Args:
        base_euler: Base euler angles [roll, pitch, yaw] (B, 3) in radians
    """

    required_keys = ("base_euler",)

    def _compute(self, base_euler: torch.Tensor) -> torch.Tensor:
        # Assuming base_euler is in degrees as per g1_env.py
        roll_rad = base_euler[:, 0] * torch.pi / 180.0
        return -torch.square(roll_rad)

class HipRollPenalty(RewardTerm):
    """
    Penalize hip roll joint positions (indices 1, 7 for G1 12-DOF).
    
    Args:
        dof_pos: Joint positions (B, D)
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:
        # G1 12-DOF hip roll indices: left_hip_roll(1), right_hip_roll(7)
        return -torch.sum(torch.abs(dof_pos[:, [1, 7]]), dim=-1)


class G1FeetSlidePenalty(RewardTerm):
    """
    Penalize foot sliding when in contact with ground.
    
    Args:
        feet_vel: Feet velocities (B, num_feet, 3)
        foot_contact_weighted: Weighted foot contact forces (B, num_feet)
    """

    required_keys = ("feet_vel", "foot_contact_weighted")
    
    feet_slide_height_threshold = 0.2

    def _compute(
        self, feet_vel: torch.Tensor, foot_contact_weighted: torch.Tensor
    ) -> torch.Tensor:
        # Check if feet are on ground (contact weighted > threshold)
        on_ground = foot_contact_weighted > 0.1
        # Get XY velocity magnitude
        feet_vel_xy = torch.norm(feet_vel[:, :, :2], dim=-1)
        # Penalize sliding when on ground
        return -torch.sum(on_ground * torch.square(feet_vel_xy), dim=1)


class MotionFeetAirTimePenalty(RewardTerm):
    """
    Penalize feet air time that's too short (encourages proper swing phase).
    
    Args:
        feet_first_contact: Binary indicator of first contact this timestep (B, num_feet)
        feet_air_time: Time feet have been in air (B, num_feet)
    """

    required_keys = ("feet_first_contact", "feet_air_time")
    
    target_feet_air_time = 0.4

    def _compute(
        self, feet_first_contact: torch.Tensor, feet_air_time: torch.Tensor
    ) -> torch.Tensor:
        # Only penalize at moment of contact if air time was too short
        pen_air_time = torch.sum(
            torch.clamp(self.target_feet_air_time - feet_air_time, min=0.0) * feet_first_contact,
            dim=1,
        )
        return -pen_air_time
### ---- Utility Functions ---- ###

def create_reward_dict(reward_configs: dict[str, dict]) -> dict[str, RewardTerm]:
    """
    Create a dictionary of reward terms from configuration.
    
    Args:
        reward_configs: Dict mapping reward names to config dicts with:
            - 'type': Class name (e.g., 'JointPosTrackingReward')
            - 'scale': Reward scale/weight
            - Additional kwargs for the reward class
    
    Returns:
        Dictionary of initialized reward terms
    """
    reward_dict = {}
    
    for name, config in reward_configs.items():
        reward_type = config.pop('type')
        scale = config.pop('scale', 1.0)
        
        # Get the reward class
        reward_class = globals().get(reward_type)
        if reward_class is None:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        # Instantiate with remaining kwargs
        reward_dict[name] = reward_class(scale=scale, name=name, **config)
    
    return reward_dict
