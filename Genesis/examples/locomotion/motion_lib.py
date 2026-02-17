"""Motion library for loading and sampling reference motion data from NPZ files."""

import numpy as np
import torch

import genesis as gs


class MotionLib:
    """
    Motion library for loading reference trajectories from NPZ files.
    
    Expected NPZ structure:
        - fps: int (e.g., 80)
        - joint_names: list of 12 joint names
        - body_names: list of 13 body link names
        - joint_pos: (num_frames, 12) - joint positions in radians
        - joint_vel: (num_frames, 12) - joint velocities in rad/s
        - body_pos_w: (num_frames, 13, 3) - body positions in world frame
        - body_quat_w: (num_frames, 13, 4) - body quaternions [w, x, y, z]
        - body_lin_vel_w: (num_frames, 13, 3) - body linear velocities
        - body_ang_vel_w: (num_frames, 13, 3) - body angular velocities
    """

    def __init__(
        self,
        motion_file: str,
        device: torch.device,
        tracking_link_names: list[str] | None = None,
    ):
        self.device = device
        self.motion_file = motion_file
        
        # Load NPZ data
        print(f"Loading motion data from: {motion_file}")
        data = np.load(motion_file, allow_pickle=True)
        
        # Metadata
        self.fps = int(data["fps"]) if "fps" in data else 80
        self.dt = 1.0 / self.fps
        self.joint_names = list(data["joint_names"])
        self.body_names = list(data["body_names"])
        
        # Motion data
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        
        self.num_frames = self.joint_pos.shape[0]
        self.num_joints = self.joint_pos.shape[1]
        self.num_bodies = self.body_pos_w.shape[1]
        
        # Motion duration
        self.motion_length = self.num_frames * self.dt
        
        # Tracking links
        self.tracking_link_names = tracking_link_names or []
        self.tracking_link_indices = []
        for name in self.tracking_link_names:
            if name in self.body_names:
                self.tracking_link_indices.append(self.body_names.index(name))
            else:
                print(f"Warning: Tracking link '{name}' not found in body_names")
        
        # For simplicity, treat as a single motion (can extend to multiple motions later)
        self.num_motions = 1
        self.motion_names = [motion_file.split("/")[-1].replace(".npz", "")]
        
        print(f"Loaded motion: {self.num_frames} frames at {self.fps} fps ({self.motion_length:.2f}s)")
        print(f"  Joints: {self.num_joints}, Bodies: {self.num_bodies}")
        print(f"  Tracking {len(self.tracking_link_indices)} links: {self.tracking_link_names}")

    def sample_motion_ids(self, n: int) -> torch.Tensor:
        """Sample motion IDs (always 0 for single motion)."""
        return torch.zeros(n, dtype=torch.long, device=self.device)

    def sample_motion_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Sample random start times within the motion."""
        n = len(motion_ids)
        # Random time in [0, motion_length)
        return torch.rand(n, device=self.device) * self.motion_length

    def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Get motion length(s) for given motion ID(s)."""
        return torch.full_like(motion_ids, self.motion_length, dtype=torch.float32)

    def get_joint_idx_by_name(self, name: str) -> int:
        """Get joint index by name."""
        return self.joint_names.index(name)

    def get_body_idx_by_name(self, name: str) -> int:
        """Get body link index by name."""
        return self.body_names.index(name)

    def _get_frame_indices(self, motion_times: torch.Tensor) -> torch.Tensor:
        """Convert motion times to frame indices with wrapping."""
        frame_idx = (motion_times / self.dt).long()
        frame_idx = frame_idx % self.num_frames
        return frame_idx

    def _interpolate_frames(
        self, data: torch.Tensor, motion_times: torch.Tensor
    ) -> torch.Tensor:
        """
        Linearly interpolate between frames.
        
        Args:
            data: (num_frames, ...) tensor
            motion_times: (B,) times to sample at
        
        Returns:
            (B, ...) interpolated data
        """
        # Get frame indices
        frame_float = motion_times / self.dt
        frame_idx0 = frame_float.long() % self.num_frames
        frame_idx1 = (frame_idx0 + 1) % self.num_frames
        alpha = (frame_float - frame_idx0.float()).unsqueeze(-1)
        
        # Interpolate
        data0 = data[frame_idx0]
        data1 = data[frame_idx1]
        
        # Handle multi-dimensional data
        while len(alpha.shape) < len(data0.shape):
            alpha = alpha.unsqueeze(-1)
        
        return data0 * (1 - alpha) + data1 * alpha

    def get_motion_frame(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Get robot state at given motion frame(s).
        
        Returns:
            base_pos: (B, 3)
            base_quat: (B, 4)
            base_lin_vel: (B, 3)
            base_ang_vel: (B, 3)
            dof_pos: (B, num_joints)
            dof_vel: (B, num_joints)
        """
        # Base is typically the first body (pelvis)
        base_pos = self._interpolate_frames(self.body_pos_w[:, 0, :], motion_times)
        base_quat = self._interpolate_frames(self.body_quat_w[:, 0, :], motion_times)
        base_lin_vel = self._interpolate_frames(self.body_lin_vel_w[:, 0, :], motion_times)
        base_ang_vel = self._interpolate_frames(self.body_ang_vel_w[:, 0, :], motion_times)
        
        # Normalize quaternion
        base_quat = base_quat / torch.norm(base_quat, dim=-1, keepdim=True)
        
        # Joint data
        dof_pos = self._interpolate_frames(self.joint_pos, motion_times)
        dof_vel = self._interpolate_frames(self.joint_vel, motion_times)
        
        return base_pos, base_quat, base_lin_vel, base_ang_vel, dof_pos, dof_vel

    def get_ref_motion_frame(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Get detailed reference motion frame including tracking links.
        
        Returns:
            base_pos: (B, 3)
            base_quat: (B, 4)
            base_lin_vel: (B, 3)
            base_ang_vel: (B, 3)
            base_ang_vel_local: (B, 3)
            dof_pos: (B, num_joints)
            dof_vel: (B, num_joints)
            link_pos_global: (B, num_tracking_links, 3)
            link_pos_local: (B, num_tracking_links, 3) - relative to base (yaw-aligned)
            link_quat_global: (B, num_tracking_links, 4)
            link_quat_local: (B, num_tracking_links, 4) - relative to base (yaw-aligned)
            link_lin_vel_global: (B, num_tracking_links, 3)
            link_lin_vel_local: (B, num_tracking_links, 3)
            link_ang_vel_global: (B, num_tracking_links, 3)
            link_ang_vel_local: (B, num_tracking_links, 3)
            foot_contact: (B, num_feet) - placeholder zeros
            foot_contact_weighted: (B, num_feet) - placeholder zeros
        """
        base_pos, base_quat, base_lin_vel, base_ang_vel, dof_pos, dof_vel = self.get_motion_frame(
            motion_ids, motion_times
        )
        
        # Get base angular velocity in local frame
        base_ang_vel_local = self._transform_by_quat_inv(base_ang_vel, base_quat)
        
        # Extract tracking link data
        if len(self.tracking_link_indices) > 0:
            link_indices = torch.tensor(self.tracking_link_indices, device=self.device)
            link_pos_global = self._interpolate_frames(
                self.body_pos_w[:, link_indices, :], motion_times
            )
            link_quat_global = self._interpolate_frames(
                self.body_quat_w[:, link_indices, :], motion_times
            )
            link_lin_vel_global = self._interpolate_frames(
                self.body_lin_vel_w[:, link_indices, :], motion_times
            )
            link_ang_vel_global = self._interpolate_frames(
                self.body_ang_vel_w[:, link_indices, :], motion_times
            )
            
            # Normalize quaternions
            link_quat_global = link_quat_global / torch.norm(link_quat_global, dim=-1, keepdim=True)
            
            # Compute yaw-aligned base quaternion (only rotation around Z)
            base_euler = self._quat_to_euler(base_quat)
            base_yaw = base_euler[:, 2]
            base_quat_yaw = self._quat_from_angle_axis(
                base_yaw, torch.tensor([0.0, 0.0, 1.0], device=self.device)
            )
            base_quat_yaw_inv = self._quat_inv(base_quat_yaw)
            
            # Transform links to local yaw-aligned frame
            link_pos_local = link_pos_global - base_pos.unsqueeze(1)
            link_pos_local = self._transform_by_quat(link_pos_local, base_quat_yaw_inv.unsqueeze(1))
            
            link_quat_local = self._quat_mul_batched(
                base_quat_yaw_inv.unsqueeze(1), link_quat_global
            )
            
            link_lin_vel_local = self._transform_by_quat(
                link_lin_vel_global, base_quat_yaw_inv.unsqueeze(1)
            )
            link_ang_vel_local = self._transform_by_quat(
                link_ang_vel_global, base_quat_yaw_inv.unsqueeze(1)
            )
        else:
            B = len(motion_ids)
            link_pos_global = torch.zeros((B, 0, 3), device=self.device)
            link_pos_local = torch.zeros((B, 0, 3), device=self.device)
            link_quat_global = torch.zeros((B, 0, 4), device=self.device)
            link_quat_local = torch.zeros((B, 0, 4), device=self.device)
            link_lin_vel_global = torch.zeros((B, 0, 3), device=self.device)
            link_lin_vel_local = torch.zeros((B, 0, 3), device=self.device)
            link_ang_vel_global = torch.zeros((B, 0, 3), device=self.device)
            link_ang_vel_local = torch.zeros((B, 0, 3), device=self.device)
        
        # Placeholder contact data (not in NPZ)
        B = len(motion_ids)
        foot_contact = torch.zeros((B, 2), device=self.device)
        foot_contact_weighted = torch.zeros((B, 2), device=self.device)
        
        return (
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            base_ang_vel_local,
            dof_pos,
            dof_vel,
            link_pos_global,
            link_pos_local,
            link_quat_global,
            link_quat_local,
            link_lin_vel_global,
            link_lin_vel_local,
            link_ang_vel_global,
            link_ang_vel_local,
            foot_contact,
            foot_contact_weighted,
        )

    def get_motion_future_obs(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        observed_steps: dict[str, list[float]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Get motion observations at current and future timesteps.
        
        Args:
            motion_ids: (B,) motion indices
            motion_times: (B,) current times
            observed_steps: dict mapping observation names to time offsets
        
        Returns:
            curr_motion_obs_dict: Current frame observations
            future_motion_obs_dict: Future frame observations at specified offsets
        """
        B = len(motion_ids)
        
        # Current frame
        (
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            base_ang_vel_local,
            dof_pos,
            dof_vel,
            link_pos_global,
            link_pos_local,
            link_quat_global,
            link_quat_local,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.get_ref_motion_frame(motion_ids, motion_times)
        
        curr_motion_obs_dict = {
            "ref_base_pos": base_pos,
            "ref_base_quat": base_quat,
            "ref_base_lin_vel": base_lin_vel,
            "ref_base_ang_vel": base_ang_vel,
            "ref_dof_pos": dof_pos,
            "ref_dof_vel": dof_vel,
            "ref_link_pos_global": link_pos_global,
            "ref_link_quat_global": link_quat_global,
        }
        
        # Future frames
        future_motion_obs_dict = {}
        for key, time_offsets in observed_steps.items():
            if len(time_offsets) == 0:
                continue
            
            future_data = []
            for offset in time_offsets:
                future_time = motion_times + offset
                # Wrap around if exceeds motion length
                future_time = future_time % self.motion_length
                
                if "dof" in key:
                    future_dof_pos = self._interpolate_frames(self.joint_pos, future_time)
                    future_data.append(future_dof_pos)
                elif "link" in key or "body" in key:
                    if len(self.tracking_link_indices) > 0:
                        link_indices = torch.tensor(self.tracking_link_indices, device=self.device)
                        future_link_pos = self._interpolate_frames(
                            self.body_pos_w[:, link_indices, :], future_time
                        )
                        future_data.append(future_link_pos)
            
            if future_data:
                future_motion_obs_dict[key] = torch.stack(future_data, dim=1)
        
        return curr_motion_obs_dict, future_motion_obs_dict

    def get_observed_steps(self, observed_steps: list[float]) -> dict[str, list[float]]:
        """Convert list of time offsets to dict format."""
        if not observed_steps:
            return {}
        return {"future_dof_pos": observed_steps}

    # Quaternion and transform utilities
    def _quat_inv(self, q: torch.Tensor) -> torch.Tensor:
        """Invert quaternion [w, x, y, z]."""
        return q * torch.tensor([1, -1, -1, -1], device=self.device)

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=-1)

    def _quat_mul_batched(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply batched quaternions with broadcasting."""
        return self._quat_mul(q1, q2)

    def _transform_by_quat(self, vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Rotate vector(s) by quaternion using v' = q * v * q^-1."""
        # Handle batched quaternions
        if quat.dim() == 3:  # (B, N_quat, 4) where N_quat may be 1 for broadcasting
            B = vec.shape[0]
            N = vec.shape[1] if vec.dim() == 3 else 1
            vec = vec.reshape(B, N, 3)
            # Broadcast quat if needed (B, 1, 4) -> (B, N, 4)
            if quat.shape[1] == 1 and N > 1:
                quat = quat.expand(B, N, 4)
            # Convert vector to quaternion [0, x, y, z]
            vec_quat = torch.cat([torch.zeros(B, N, 1, device=self.device), vec], dim=-1)
            quat_inv = self._quat_inv(quat)
            result = self._quat_mul(self._quat_mul(quat, vec_quat), quat_inv)
            return result[..., 1:]
        else:  # (B, 4)
            B = quat.shape[0]
            if vec.dim() == 3:  # (B, N, 3)
                N = vec.shape[1]
                vec_quat = torch.cat([torch.zeros(B, N, 1, device=self.device), vec], dim=-1)
                quat = quat.unsqueeze(1).expand(B, N, 4)
            else:  # (B, 3)
                vec_quat = torch.cat([torch.zeros(B, 1, device=self.device), vec], dim=-1)
            quat_inv = self._quat_inv(quat)
            result = self._quat_mul(self._quat_mul(quat, vec_quat), quat_inv)
            return result[..., 1:]

    def _transform_by_quat_inv(self, vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Rotate vector by inverse quaternion."""
        return self._transform_by_quat(vec, self._quat_inv(quat))

    def _quat_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.pi / 2,
            torch.asin(sinp)
        )
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw], dim=-1)

    def _quat_from_angle_axis(self, angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        """Create quaternion from angle and axis."""
        half_angle = angle * 0.5
        sin_half = torch.sin(half_angle)
        
        if axis.dim() == 1:
            # Single axis, batched angles
            axis = axis / torch.norm(axis)
            w = torch.cos(half_angle)
            xyz = axis * sin_half.unsqueeze(-1)
        else:
            # Batched axis
            axis = axis / torch.norm(axis, dim=-1, keepdim=True)
            w = torch.cos(half_angle)
            xyz = axis * sin_half.unsqueeze(-1)
        
        return torch.cat([w.unsqueeze(-1), xyz], dim=-1)
