"""G1 humanoid motion tracking environment for high-fidelity motion imitation."""

import math
from typing import Any

import torch

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from g1_env import G1Env, gs_rand
from motion_lib import MotionLib
import reward_terms


class G1TrackingEnv(G1Env):
    """
    Motion tracking environment that trains a policy to imitate reference motions.
    
    Extends G1Env with:
    - Reference motion loading and sampling
    - Phase-based observations for motion timing
    - Tracking rewards for joint positions, velocities, and end-effectors
    - Hard synchronization option for precise motion replay
    """

    def _init_tracking_buffers(self):
        """Initialize tracking-specific buffers after parent initialization."""
        # Load motion library
        motion_file = self.tracking_cfg.get("motion_file")
        tracking_link_names = self.tracking_cfg.get("tracking_link_names", [])
        
        if motion_file is None:
            raise ValueError("tracking_cfg must contain 'motion_file' path")
        
        self.motion_lib = MotionLib(
            motion_file=motion_file,
            device=gs.device,
            tracking_link_names=tracking_link_names,
        )
        
        # Motion state
        self.motion_ids = torch.zeros(self.num_envs, dtype=gs.tc_int, device=gs.device)
        self.motion_times = torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)
        self.motion_time_offsets = torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)
        self.motion_lengths = torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)
        
        # Reference state buffers
        self.ref_base_pos = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.ref_base_quat = torch.zeros((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.ref_base_quat[:, 0] = 1.0  # Identity quaternion
        self.ref_base_lin_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.ref_base_ang_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        self.ref_dof_vel = torch.zeros_like(self.dof_vel)
        
        # Phase variable for temporal encoding
        self.phase = torch.zeros((self.num_envs, 1), dtype=gs.tc_float, device=gs.device)
        
        # New observation components
        self.last_actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.diff_base_yaw = torch.zeros((self.num_envs, 1), dtype=gs.tc_float, device=gs.device)
        self.diff_base_pos_local_yaw = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel_local = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        
        # Tracking link states (if specified)
        num_tracking_links = len(tracking_link_names)
        self.tracking_link_names = tracking_link_names
        self.tracking_link_idx = []
        
        if num_tracking_links > 0:
            # Map tracking link names to robot link indices
            for name in tracking_link_names:
                try:
                    link = self.robot.get_link(name)
                    self.tracking_link_idx.append(link.idx_local)
                except:
                    print(f"Warning: Tracking link '{name}' not found in robot")
            
            self.tracking_link_idx = torch.tensor(
                self.tracking_link_idx, dtype=gs.tc_int, device=gs.device
            )
            
            self.tracking_link_pos = torch.zeros(
                (self.num_envs, len(self.tracking_link_idx), 3),
                dtype=gs.tc_float,
                device=gs.device,
            )
            self.ref_tracking_link_pos = torch.zeros_like(self.tracking_link_pos)
            
            # Observation components for tracking links
            self.diff_tracking_link_pos_local_yaw = torch.zeros_like(self.tracking_link_pos)
            self.diff_tracking_link_rotation_6D = torch.zeros(
                (self.num_envs, len(self.tracking_link_idx), 6),
                dtype=gs.tc_float,
                device=gs.device,
            )
            self.tracking_link_quat = torch.zeros(
                (self.num_envs, len(self.tracking_link_idx), 4),
                dtype=gs.tc_float,
                device=gs.device,
            )
            self.ref_tracking_link_quat = torch.zeros_like(self.tracking_link_quat)
        else:
            self.tracking_link_pos = torch.zeros((self.num_envs, 0, 3), dtype=gs.tc_float, device=gs.device)
            self.ref_tracking_link_pos = torch.zeros_like(self.tracking_link_pos)
            self.diff_tracking_link_pos_local_yaw = torch.zeros_like(self.tracking_link_pos)
            self.diff_tracking_link_rotation_6D = torch.zeros((self.num_envs, 0, 6), dtype=gs.tc_float, device=gs.device)
            self.tracking_link_quat = torch.zeros((self.num_envs, 0, 4), dtype=gs.tc_float, device=gs.device)
            self.ref_tracking_link_quat = torch.zeros_like(self.tracking_link_quat)
        
        # Difference terms for observations
        self.diff_base_pos = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.diff_base_quat = torch.zeros((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.diff_dof_pos = torch.zeros_like(self.dof_pos)
        self.diff_dof_vel = torch.zeros_like(self.dof_vel)
        
        # Termination thresholds for tracking errors
        self.termination_thresholds = self.tracking_cfg.get("termination_thresholds", {})
        
        # Hard reset option (sync robot state to reference)
        self.hard_reset_ratio = self.tracking_cfg.get("hard_reset_ratio", 0.8)
        
        print(f"Initialized tracking environment with:")
        print(f"  Motion: {motion_file}")
        print(f"  Duration: {self.motion_lib.motion_length:.2f}s ({self.motion_lib.num_frames} frames)")
        print(f"  Tracking {len(self.tracking_link_idx)} links")

    def _init_reward_terms(self):
        """Initialize RewardTerm instances from configuration."""
        self.reward_terms = {}
        
        # Get reward scales from config (now under 'scales' key)
        reward_scales = self.reward_cfg.get("scales", {})
        
        # If no 'scales' key, try legacy 'reward_scales' format
        if not reward_scales:
            reward_scales = self.reward_cfg.get("reward_scales", {})
        
        if not reward_scales:
            print("Warning: No reward scales found in reward_cfg['scales'] or reward_cfg['reward_scales']")
            return
        
        print(f"\nInitializing {len(reward_scales)} reward terms:")
        for name, scale in reward_scales.items():
            # Get the reward class from reward_terms module
            reward_class = getattr(reward_terms, name, None)
            if reward_class is None:
                print(f"  Warning: Reward class '{name}' not found in reward_terms.py, skipping")
                continue
            
            # Instantiate with scale multiplied by dt
            try:
                self.reward_terms[name] = reward_class(scale=scale * self.dt, name=name)
                print(f"  ✓ {name}: {scale * self.dt:.6f}")
            except Exception as e:
                print(f"  ✗ {name}: Failed to instantiate - {e}")
        
        # Initialize deviation buffer for adaptive weighting
        self.deviation_buf = torch.ones(self.num_envs, dtype=gs.tc_float, device=gs.device)
        
        # Initialize tracking link weights
        num_tracking_links = len(self.tracking_link_idx) if hasattr(self, 'tracking_link_idx') else 0
        self.tracking_link_pos_global_weights = torch.ones(num_tracking_links, dtype=gs.tc_float, device=gs.device)
        self.tracking_link_pos_local_weights = torch.ones(num_tracking_links, dtype=gs.tc_float, device=gs.device)
        self.tracking_link_quat_weights = torch.ones(num_tracking_links, dtype=gs.tc_float, device=gs.device)
        
        print(f"Reward system initialized with {len(self.reward_terms)} active terms\n")

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        tracking_cfg,
        randomization_cfg=None,
        show_viewer=False,
    ):
        # Store tracking config
        self.tracking_cfg = tracking_cfg
        
        # Temporarily replace reward_cfg to prevent parent from initializing method-based rewards
        # We'll use RewardTerm class-based system instead
        original_reward_cfg = reward_cfg
        temp_reward_cfg = {
            "reward_scales": {},  # Empty so parent doesn't try to create method-based rewards
            "scales": {},
        }
        # Copy other reward config parameters if they exist
        if "parameters" in reward_cfg:
            temp_reward_cfg["parameters"] = reward_cfg["parameters"]
        
        # Initialize parent with empty reward scales
        super().__init__(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=temp_reward_cfg,
            command_cfg=command_cfg,
            randomization_cfg=randomization_cfg,
            show_viewer=show_viewer,
        )
        
        # Restore original reward config
        self.reward_cfg = original_reward_cfg
        
        # Initialize tracking-specific buffers
        self._init_tracking_buffers()
        
        # Override episode length to match motion duration
        if self.tracking_cfg.get("match_motion_duration", True):
            motion_duration = self.motion_lib.motion_length
            self.max_episode_length = math.ceil(motion_duration / self.dt)
            print(f"  Episode length: {self.max_episode_length} steps ({motion_duration:.2f}s)")
        
        # Initialize RewardTerm-based reward system (overrides parent method-based system)
        self._init_reward_terms()

    def _reset_motion(self, envs_idx):
        """Sample new reference motions for specified environments."""
        n = len(envs_idx) if envs_idx is not None else self.num_envs
        idx = envs_idx if envs_idx is not None else torch.arange(self.num_envs, device=gs.device)
        
        # Sample motion IDs and start times
        motion_ids = self.motion_lib.sample_motion_ids(n)
        motion_times = self.motion_lib.sample_motion_times(motion_ids)
        
        self.motion_ids[idx] = motion_ids.to(gs.tc_int)
        self.motion_time_offsets[idx] = motion_times
        self.motion_lengths[idx] = self.motion_lib.get_motion_length(motion_ids)

    def _update_ref_motion(self, envs_idx=None):
        """Update reference motion state for current timestep."""
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device, dtype=gs.tc_int)
        
        motion_ids = self.motion_ids[envs_idx]
        motion_times = self.motion_time_offsets[envs_idx] + self.episode_length_buf[envs_idx].float() * self.dt
        
        # Get reference frame from motion library
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
            link_lin_vel_global,
            link_lin_vel_local,
            link_ang_vel_global,
            link_ang_vel_local,
            foot_contact,
            foot_contact_weighted,
        ) = self.motion_lib.get_ref_motion_frame(motion_ids, motion_times)
        
        # Update reference buffers
        self.ref_base_pos[envs_idx] = base_pos
        self.ref_base_quat[envs_idx] = base_quat
        self.ref_base_lin_vel[envs_idx] = base_lin_vel
        self.ref_base_ang_vel[envs_idx] = base_ang_vel
        self.ref_dof_pos[envs_idx] = dof_pos
        self.ref_dof_vel[envs_idx] = dof_vel
        
        if len(self.tracking_link_idx) > 0:
            self.ref_tracking_link_pos[envs_idx] = link_pos_global
            self.ref_tracking_link_quat[envs_idx] = link_quat_global
        
        # Update phase (normalized time in [0, 2π])
        phase = (motion_times / self.motion_lib.motion_length) * 2 * math.pi
        self.phase[envs_idx] = phase.unsqueeze(-1)
        
        # Compute differences for observations
        self.diff_base_pos[envs_idx] = self.ref_base_pos[envs_idx] - self.base_pos[envs_idx]
        self.diff_dof_pos[envs_idx] = self.ref_dof_pos[envs_idx] - self.dof_pos[envs_idx]
        self.diff_dof_vel[envs_idx] = self.ref_dof_vel[envs_idx] - self.dof_vel[envs_idx]

    def hard_sync_motion(self, envs_idx):
        """Synchronize robot state to current reference motion frame."""
        motion_ids = self.motion_ids[envs_idx]
        motion_times = self.motion_time_offsets[envs_idx] + self.episode_length_buf[envs_idx].float() * self.dt
        
        # Get full motion state
        base_pos, base_quat, base_lin_vel, base_ang_vel, dof_pos, dof_vel = (
            self.motion_lib.get_motion_frame(motion_ids, motion_times)
        )
        
        # Build full qpos: [base_pos(3), base_quat(4), dof_pos(12)]
        qpos = torch.cat([base_pos, base_quat, dof_pos], dim=-1)
        
        # Set robot position and joint states
        # Note: motors_dof_idx specifies which DOFs to set (12 actuated joints, not 18 total including base)
        self.robot.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)
        self.robot.set_dofs_velocity(dof_vel, dofs_idx_local=self.motors_dof_idx, envs_idx=envs_idx, skip_forward=False)

    def step(self, actions):
        """Step the environment with tracking updates."""
        # Store previous actions for observation
        self.last_actions = self.actions.clone()
        
        # Store actions
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        # Update episode buffers
        self.episode_length_buf += 1
        
        # Update robot state
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # Update tracking link positions and orientations
        if len(self.tracking_link_idx) > 0:
            for i, link_idx in enumerate(self.tracking_link_idx):
                link = self.robot.links[link_idx]
                self.tracking_link_pos[:, i, :] = link.get_pos()
                self.tracking_link_quat[:, i, :] = link.get_quat()

        # Update feet states (from parent)
        for i, link_name in enumerate(self.feet_links):
            link = self.robot.get_link(link_name)
            self.feet_pos[:, i, :] = link.get_pos()
            self.feet_vel[:, i, :] = link.get_vel()
        self.feet_heights[:, 0] = self.feet_pos[:, 0, 2]
        self.feet_heights[:, 1] = self.feet_pos[:, 1, 2]
        
        # Update contact forces and air time
        all_contact_forces = self.robot.get_links_net_contact_force()
        for i, link_idx in enumerate(self.feet_link_idx):
            if all_contact_forces.dim() == 2:
                contact_force = all_contact_forces[link_idx:link_idx+1, :]
            else:
                contact_force = all_contact_forces[:, link_idx, :]
            
            self.contact_forces[:, i, :] = contact_force
            contact_norm = torch.norm(contact_force, dim=-1)
            in_contact = contact_norm > 5.0
            self.foot_contact_weighted[:, i] = torch.clamp(contact_norm / 100.0, 0.0, 1.0)
            self.feet_air_time[:, i] += self.dt
            first_contact = in_contact & ~self.last_contacts[:, i]
            self.feet_first_contact[:, i] = first_contact
            self.latched_air_time[:, i] = torch.where(
                first_contact, self.feet_air_time[:, i], self.latched_air_time[:, i]
            )
            self.feet_air_time[:, i] *= ~in_contact
            self.last_contacts[:, i] = in_contact

        # Update reference motion AFTER state update
        self._update_ref_motion()

        # Compute rewards using RewardTerm system
        self._compute_tracking_rewards()

        # Apply domain randomization (pushes)
        if self.push_robots:
            self._push_robots()

        # Check termination
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        
        # Termination based on orientation
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_base_height_less_than"]
        
        # Termination based on tracking errors (if specified)
        if "joint_pos_error" in self.termination_thresholds:
            joint_error = torch.sum(torch.abs(self.diff_dof_pos), dim=-1)
            self.reset_buf |= joint_error > self.termination_thresholds["joint_pos_error"]
        
        if "base_height_error" in self.termination_thresholds:
            height_error = torch.abs(self.base_pos[:, 2] - self.ref_base_pos[:, 2])
            self.reset_buf |= height_error > self.termination_thresholds["base_height_error"]

        # Timeout flag
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # Reset if needed
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # Update observations
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        self.extras["observations"] = {"critic": self.obs_buf}

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _reset_idx(self, envs_idx=None):
        """Reset specified environments with motion initialization."""
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device, dtype=gs.tc_int)
        
        # Sample new reference motions
        self._reset_motion(envs_idx)
        
        # Decide whether to hard sync or soft reset
        n_hard = int(len(envs_idx) * self.hard_reset_ratio)
        if n_hard > 0:
            # Hard sync: Set robot state to exact reference frame
            perm = torch.randperm(len(envs_idx), device=gs.device)
            hard_idx = envs_idx[perm[:n_hard]]
            soft_idx = envs_idx[perm[n_hard:]]
            
            # Update reference for hard reset envs
            self._update_ref_motion(hard_idx)
            self.hard_sync_motion(hard_idx)
            
            # Soft reset for remaining envs
            if len(soft_idx) > 0:
                self._update_ref_motion(soft_idx)
                self.robot.set_qpos(self.init_qpos, envs_idx=soft_idx, zero_velocity=True, skip_forward=True)
        else:
            # All soft resets
            self._update_ref_motion(envs_idx)
            self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # Reset buffers (parent implementation)
        if envs_idx is None or len(envs_idx) == self.num_envs:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            self.feet_air_time.zero_()
            self.last_contacts.zero_()
        else:
            # Use integer indices with broadcasting for init values
            self.base_pos[envs_idx] = self.init_base_pos
            self.base_quat[envs_idx] = self.init_base_quat
            self.projected_gravity[envs_idx] = self.init_projected_gravity
            self.dof_pos[envs_idx] = self.init_dof_pos
            self.base_lin_vel[envs_idx] = 0.0
            self.base_ang_vel[envs_idx] = 0.0
            self.dof_vel[envs_idx] = 0.0
            self.actions[envs_idx] = 0.0
            self.last_actions[envs_idx] = 0.0
            self.last_dof_vel[envs_idx] = 0.0
            self.episode_length_buf[envs_idx] = 0
            self.reset_buf[envs_idx] = True
            self.feet_air_time[envs_idx] = 0.0
            self.last_contacts[envs_idx] = False

        # Log episode rewards
        n_envs = len(envs_idx)
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if n_envs > 0:
                mean = value[envs_idx].sum() / n_envs
            else:
                mean = torch.tensor(0.0, device=gs.device)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            value[envs_idx] = 0.0

    def _compute_observation_components(self):
        """Compute all observation components for the new observation structure."""
        # 1. Base angular velocity in local frame
        inv_base_quat = inv_quat(self.base_quat)
        self.base_ang_vel_local = transform_by_quat(self.base_ang_vel, inv_base_quat)
        
        # 2. Yaw difference between current and reference base orientation
        # Extract yaw from current base quat
        curr_yaw = torch.atan2(
            2 * (self.base_quat[:, 0] * self.base_quat[:, 3] + self.base_quat[:, 1] * self.base_quat[:, 2]),
            1 - 2 * (self.base_quat[:, 2]**2 + self.base_quat[:, 3]**2)
        )
        # Extract yaw from reference base quat
        ref_yaw = torch.atan2(
            2 * (self.ref_base_quat[:, 0] * self.ref_base_quat[:, 3] + self.ref_base_quat[:, 1] * self.ref_base_quat[:, 2]),
            1 - 2 * (self.ref_base_quat[:, 2]**2 + self.ref_base_quat[:, 3]**2)
        )
        # Compute yaw difference (wrap to [-pi, pi])
        self.diff_base_yaw = torch.remainder(curr_yaw - ref_yaw + math.pi, 2 * math.pi) - math.pi
        self.diff_base_yaw = self.diff_base_yaw.unsqueeze(-1)
        
        # 3. Base position difference in yaw-aligned local frame
        # Use current yaw to create rotation matrix
        cos_yaw = torch.cos(curr_yaw)
        sin_yaw = torch.sin(curr_yaw)
        diff_base_pos_global = self.base_pos - self.ref_base_pos
        self.diff_base_pos_local_yaw[:, 0] = cos_yaw * diff_base_pos_global[:, 0] + sin_yaw * diff_base_pos_global[:, 1]
        self.diff_base_pos_local_yaw[:, 1] = -sin_yaw * diff_base_pos_global[:, 0] + cos_yaw * diff_base_pos_global[:, 1]
        self.diff_base_pos_local_yaw[:, 2] = diff_base_pos_global[:, 2]
        
        # 4. Tracking link position differences in yaw-aligned local frame (if tracking links exist)
        if len(self.tracking_link_idx) > 0:
            for i in range(len(self.tracking_link_idx)):
                diff_link_pos_global = self.tracking_link_pos[:, i] - self.ref_tracking_link_pos[:, i]
                self.diff_tracking_link_pos_local_yaw[:, i, 0] = cos_yaw * diff_link_pos_global[:, 0] + sin_yaw * diff_link_pos_global[:, 1]
                self.diff_tracking_link_pos_local_yaw[:, i, 1] = -sin_yaw * diff_link_pos_global[:, 0] + cos_yaw * diff_link_pos_global[:, 1]
                self.diff_tracking_link_pos_local_yaw[:, i, 2] = diff_link_pos_global[:, 2]
            
            # 5. Tracking link rotation differences in 6D representation
            # Convert quaternion difference to 6D rotation representation
            for i in range(len(self.tracking_link_idx)):
                # Compute relative rotation: q_diff = q_ref^-1 * q_curr
                ref_quat_inv = inv_quat(self.ref_tracking_link_quat[:, i])
                diff_quat = transform_quat_by_quat(ref_quat_inv, self.tracking_link_quat[:, i])
                
                # Convert to rotation matrix and extract first two columns (6D representation)
                # 6D rotation representation: first two columns of rotation matrix
                # This is more continuous than quaternions for learning
                w, x, y, z = diff_quat[:, 0], diff_quat[:, 1], diff_quat[:, 2], diff_quat[:, 3]
                
                # First column of rotation matrix
                self.diff_tracking_link_rotation_6D[:, i, 0] = 1 - 2 * (y**2 + z**2)
                self.diff_tracking_link_rotation_6D[:, i, 1] = 2 * (x*y + w*z)
                self.diff_tracking_link_rotation_6D[:, i, 2] = 2 * (x*z - w*y)
                
                # Second column of rotation matrix
                self.diff_tracking_link_rotation_6D[:, i, 3] = 2 * (x*y - w*z)
                self.diff_tracking_link_rotation_6D[:, i, 4] = 1 - 2 * (x**2 + z**2)
                self.diff_tracking_link_rotation_6D[:, i, 5] = 2 * (y*z + w*x)

    def _update_observation(self):
        """Construct observation with new structure."""
        # Compute all observation components
        self._compute_observation_components()
        
        # Get motion observations from motion library
        motion_obs = self._get_motion_obs()
        
        # Build observation vector:
        # last_action(12) + dof_pos(12) + dof_vel(12) + base_ang_vel_local(3) + 
        # diff_base_yaw(1) + diff_base_pos_local_yaw(3) + 
        # diff_tracking_link_pos_local_yaw(N*3) + diff_tracking_link_rotation_6D(N*6) + 
        # projected_gravity(3) + motion_obs(varies)
        obs_components = [
            self.last_actions,  # 12
            self.dof_pos * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],  # 12
            self.base_ang_vel_local * self.obs_scales["ang_vel"],  # 3
            self.diff_base_yaw,  # 1
            self.diff_base_pos_local_yaw,  # 3
        ]
        
        # Add tracking link observations (if available)
        if len(self.tracking_link_idx) > 0:
            obs_components.append(self.diff_tracking_link_pos_local_yaw.reshape(self.num_envs, -1))  # N*3
            obs_components.append(self.diff_tracking_link_rotation_6D.reshape(self.num_envs, -1))  # N*6
        
        obs_components.extend([
            self.projected_gravity,  # 3
            motion_obs,  # varies based on config
        ])
        
        self.obs_buf = torch.cat(obs_components, dim=-1)
    
    def _get_motion_obs(self):
        """Get motion observations from motion library based on config."""
        obs_cfg = self.obs_cfg.get("motion_obs", {})
        observed_steps = obs_cfg.get("observed_steps", [])
        
        if len(observed_steps) == 0:
            # No future observations, return empty tensor
            return torch.zeros((self.num_envs, 0), dtype=gs.tc_float, device=gs.device)
        
        # Get current motion time
        motion_times = self.motion_time_offsets + self.episode_length_buf.float() * self.dt
        
        # Get observed frames from motion library
        curr_obs_dict, future_obs_dict = self.motion_lib.get_observed_motion_frames(
            self.motion_ids, motion_times, {"future_dof_pos": observed_steps}
        )
        
        # Flatten future observations into a single vector
        obs_list = []
        if "future_dof_pos" in future_obs_dict:
            # future_dof_pos shape: (B, num_steps, 12)
            future_dof = future_obs_dict["future_dof_pos"]
            obs_list.append(future_dof.reshape(self.num_envs, -1))
        
        if len(obs_list) == 0:
            return torch.zeros((self.num_envs, 0), dtype=gs.tc_float, device=gs.device)
        
        return torch.cat(obs_list, dim=-1)

    def reset(self):
        """Reset all environments."""
        self._reset_idx()
        self._update_observation()
        return self.obs_buf, None

    # ==================== Reward Computation ====================
    
    def _compute_tracking_rewards(self):
        """Compute rewards using RewardTerm system."""
        # Build state dictionary with all required keys
        state = self._build_reward_state()
        
        # Compute each reward term
        self.rew_buf.zero_()
        for name, reward_term in self.reward_terms.items():
            try:
                # Check if all required keys are available
                missing_keys = [k for k in reward_term.required_keys if k not in state]
                if missing_keys:
                    print(f"Warning: Reward '{name}' missing keys: {missing_keys}")
                    continue
                
                # Compute reward
                rew = reward_term(state)
                self.rew_buf += rew
                
                # Track episode sum
                if name not in self.episode_sums:
                    self.episode_sums[name] = torch.zeros(
                        self.num_envs, dtype=gs.tc_float, device=gs.device
                    )
                self.episode_sums[name] += rew
            except Exception as e:
                print(f"Error computing reward '{name}': {e}")
    
    def _build_reward_state(self) -> dict[str, torch.Tensor]:
        """Build state dictionary for reward computation."""
        # Compute weighted errors for tracking rewards
        dof_pos_error_weighted = torch.sum(
            torch.square(self.dof_pos - self.ref_dof_pos), dim=-1
        )
        dof_vel_error_weighted = torch.sum(
            torch.square(self.dof_vel - self.ref_dof_vel), dim=-1
        )
        
        # Compute torques using PD controller formula
        # torque = kp * (target_pos - current_pos) - kd * current_vel
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        pos_error = target_dof_pos - self.dof_pos
        
        # Get kp and kd values (they're set per joint in __init__)
        # We need to approximate based on joint type since they're not directly accessible
        kp_values = []
        kd_values = []
        for joint_name in self.env_cfg["joint_names"]:
            if "hip_yaw" in joint_name:
                kp_values.append(self.env_cfg["stiffness"]["hip_yaw"])
                kd_values.append(self.env_cfg["damping"]["hip_yaw"])
            elif "hip_roll" in joint_name:
                kp_values.append(self.env_cfg["stiffness"]["hip_roll"])
                kd_values.append(self.env_cfg["damping"]["hip_roll"])
            elif "hip_pitch" in joint_name:
                kp_values.append(self.env_cfg["stiffness"]["hip_pitch"])
                kd_values.append(self.env_cfg["damping"]["hip_pitch"])
            elif "knee" in joint_name:
                kp_values.append(self.env_cfg["stiffness"]["knee"])
                kd_values.append(self.env_cfg["damping"]["knee"])
            elif "ankle" in joint_name:
                kp_values.append(self.env_cfg["stiffness"]["ankle"])
                kd_values.append(self.env_cfg["damping"]["ankle"])
        
        kp_tensor = torch.tensor(kp_values, dtype=gs.tc_float, device=gs.device)
        kd_tensor = torch.tensor(kd_values, dtype=gs.tc_float, device=gs.device)
        
        self.torques = kp_tensor * pos_error - kd_tensor * self.dof_vel
        
        # Get tracking link states if available
        if len(self.tracking_link_idx) > 0:
            tracking_link_pos_global = self.tracking_link_pos
            ref_tracking_link_pos_global = self.ref_tracking_link_pos
            
            # Get link velocities from robot
            tracking_link_lin_vel_global = torch.zeros_like(tracking_link_pos_global)
            for i, link_idx in enumerate(self.tracking_link_idx):
                link = self.robot.links[link_idx]
                tracking_link_lin_vel_global[:, i, :] = link.get_vel()
            
            # Compute local frame positions (yaw-aligned)
            base_yaw = torch.atan2(
                2 * (self.base_quat[:, 0] * self.base_quat[:, 3] + self.base_quat[:, 1] * self.base_quat[:, 2]),
                1 - 2 * (self.base_quat[:, 2]**2 + self.base_quat[:, 3]**2)
            )
            cos_yaw = torch.cos(base_yaw)
            sin_yaw = torch.sin(base_yaw)
            
            # Transform to yaw-aligned local frame
            tracking_link_pos_local_yaw = torch.zeros_like(tracking_link_pos_global)
            ref_tracking_link_pos_local_yaw = torch.zeros_like(ref_tracking_link_pos_global)
            
            for i in range(tracking_link_pos_global.shape[1]):
                rel_pos = tracking_link_pos_global[:, i] - self.base_pos
                tracking_link_pos_local_yaw[:, i, 0] = cos_yaw * rel_pos[:, 0] + sin_yaw * rel_pos[:, 1]
                tracking_link_pos_local_yaw[:, i, 1] = -sin_yaw * rel_pos[:, 0] + cos_yaw * rel_pos[:, 1]
                tracking_link_pos_local_yaw[:, i, 2] = rel_pos[:, 2]
                
                ref_rel_pos = ref_tracking_link_pos_global[:, i] - self.ref_base_pos
                ref_tracking_link_pos_local_yaw[:, i, 0] = cos_yaw * ref_rel_pos[:, 0] + sin_yaw * ref_rel_pos[:, 1]
                ref_tracking_link_pos_local_yaw[:, i, 1] = -sin_yaw * ref_rel_pos[:, 0] + cos_yaw * ref_rel_pos[:, 1]
                ref_tracking_link_pos_local_yaw[:, i, 2] = ref_rel_pos[:, 2]
            
            # Placeholder for link quaternions and velocities (would need to be computed from robot)
            tracking_link_quat_local_yaw = torch.zeros(
                (self.num_envs, len(self.tracking_link_idx), 4),
                dtype=gs.tc_float, device=gs.device
            )
            # Get actual link quaternions from robot
            for i, link_idx in enumerate(self.tracking_link_idx):
                link = self.robot.links[link_idx]
                link_quat = link.get_quat()
                # Transform to yaw-aligned local frame (simplified - using global for now)
                tracking_link_quat_local_yaw[:, i, :] = link_quat
            
            # Get reference link quaternions and velocities from motion library
            # We'll call get_ref_motion_frame once for all tracking link data
            motion_ids_for_links = self.motion_ids
            motion_times_for_links = self.motion_time_offsets + self.episode_length_buf.float() * self.dt
            
            (
                _, _, _, _, _,  # base states
                _, _,  # dof_pos, dof_vel
                _,  # link_pos_global
                _,  # link_pos_local
                ref_link_quat_global,
                ref_link_quat_local,
                ref_link_lin_vel_global,
                _,  # link_lin_vel_local
                _,  # link_ang_vel_global
                _,  # link_ang_vel_local
                _,  # foot_contact
                _,  # foot_contact_weighted
            ) = self.motion_lib.get_ref_motion_frame(motion_ids_for_links, motion_times_for_links)
            
            # Use the local yaw-aligned quaternions and velocities from motion lib
            ref_tracking_link_quat_local_yaw = ref_link_quat_local
            ref_tracking_link_lin_vel_global = ref_link_lin_vel_global
        else:
            # Empty tensors if no tracking links
            tracking_link_pos_global = torch.zeros((self.num_envs, 0, 3), dtype=gs.tc_float, device=gs.device)
            ref_tracking_link_pos_global = torch.zeros_like(tracking_link_pos_global)
            tracking_link_pos_local_yaw = torch.zeros_like(tracking_link_pos_global)
            ref_tracking_link_pos_local_yaw = torch.zeros_like(tracking_link_pos_global)
            tracking_link_quat_local_yaw = torch.zeros((self.num_envs, 0, 4), dtype=gs.tc_float, device=gs.device)
            ref_tracking_link_quat_local_yaw = torch.zeros_like(tracking_link_quat_local_yaw)
            tracking_link_lin_vel_global = torch.zeros_like(tracking_link_pos_global)
            ref_tracking_link_lin_vel_global = torch.zeros_like(tracking_link_pos_global)
        
        # Get reference foot contact from the most recent motion library call in _update_ref_motion
        # We already have the motion state, so we can call get_ref_motion_frame once more for contact
        motion_ids_for_contact = self.motion_ids
        motion_times_for_contact = self.motion_time_offsets + self.episode_length_buf.float() * self.dt
        
        # Get reference motion data including foot contact
        (
            _, _, _, _, _,  # base states
            _, _,  # dof states
            _, _, _, _,  # link pos/quat global/local
            _, _, _, _,  # link vel global/local
            ref_foot_contact,
            ref_foot_contact_weighted,
        ) = self.motion_lib.get_ref_motion_frame(motion_ids_for_contact, motion_times_for_contact)
        
        # Build complete state dictionary
        state = {
            # Basic states
            "actions": self.actions,
            "last_actions": self.last_actions,
            "dof_pos": self.dof_pos,
            "dof_vel": self.dof_vel,
            "base_pos": self.base_pos,
            "base_quat": self.base_quat,
            "base_euler": self.base_euler,
            "base_lin_vel": self.base_lin_vel,
            "base_ang_vel": self.base_ang_vel,
            "projected_gravity": self.projected_gravity,
            "torques": self.torques,
            
            # Reference states
            "ref_base_pos": self.ref_base_pos,
            "ref_base_quat": self.ref_base_quat,
            "ref_base_ang_vel": self.ref_base_ang_vel,
            "ref_dof_pos": self.ref_dof_pos,
            "ref_dof_vel": self.ref_dof_vel,
            
            # Weighted errors
            "dof_pos_error_weighted": dof_pos_error_weighted,
            "dof_vel_error_weighted": dof_vel_error_weighted,
            "deviation_buf": self.deviation_buf,
            
            # Tracking link states
            "tracking_link_pos_global": tracking_link_pos_global,
            "ref_tracking_link_pos_global": ref_tracking_link_pos_global,
            "tracking_link_pos_global_weights": self.tracking_link_pos_global_weights,
            "tracking_link_pos_local_yaw": tracking_link_pos_local_yaw,
            "ref_tracking_link_pos_local_yaw": ref_tracking_link_pos_local_yaw,
            "tracking_link_pos_local_weights": self.tracking_link_pos_local_weights,
            "tracking_link_quat_local_yaw": tracking_link_quat_local_yaw,
            "ref_tracking_link_quat_local_yaw": ref_tracking_link_quat_local_yaw,
            "tracking_link_quat_weights": self.tracking_link_quat_weights,
            "tracking_link_lin_vel_global": tracking_link_lin_vel_global,
            "ref_tracking_link_lin_vel_global": ref_tracking_link_lin_vel_global,
            
            # Foot contact states
            "foot_contact_weighted": self.foot_contact_weighted,
            "ref_foot_contact": ref_foot_contact,
            "ref_foot_contact_weighted": ref_foot_contact_weighted,
            "feet_vel": self.feet_vel,
            "feet_first_contact": self.feet_first_contact,
            "feet_air_time": self.latched_air_time,
        }
        
        return state

    # ==================== Legacy Tracking Reward Functions (Unused) ====================
    # These are kept for reference but not used - RewardTerm system is used instead
    
    # def _reward_JointPosTrackingReward(self):
    #     """Reward for tracking reference joint positions."""
    #     error = torch.sum(torch.square(self.dof_pos - self.ref_dof_pos), dim=-1)
    #     sigma = self.reward_cfg.get("joint_pos_tracking_sigma", 0.1)
    #     return torch.exp(-error / sigma)

    # def _reward_JointVelTrackingReward(self):
    #     \"\"\"Reward for tracking reference joint velocities.\"\"\"
    #     error = torch.sum(torch.square(self.dof_vel - self.ref_dof_vel), dim=-1)
    #     sigma = self.reward_cfg.get(\"joint_vel_tracking_sigma\", 0.5)
    #     return torch.exp(-error / sigma)

    # def _reward_BaseHeightTrackingReward(self):
    #     \"\"\"Reward for tracking reference base height.\"\"\"
    #     error = torch.square(self.base_pos[:, 2] - self.ref_base_pos[:, 2])
    #     sigma = self.reward_cfg.get(\"base_height_tracking_sigma\", 0.1)
    #     return torch.exp(-error / sigma)

    # def _reward_BasePositionTrackingReward(self):
    #     \"\"\"Reward for tracking reference base XYZ position.\"\"\"
    #     error = torch.sum(torch.square(self.base_pos - self.ref_base_pos), dim=-1)
    #     sigma = self.reward_cfg.get(\"base_pos_tracking_sigma\", 0.25)
    #     return torch.exp(-error / sigma)

    # def _reward_EndEffectorTrackingReward(self):
    #     \"\"\"Reward for tracking end-effector (feet) positions.\"\"\"
    #     if len(self.tracking_link_idx) == 0:
    #         return torch.zeros(self.num_envs, device=gs.device)
    #     
    #     error = torch.sum(
    #         torch.square(self.tracking_link_pos - self.ref_tracking_link_pos), dim=-1
    #     ).mean(dim=-1)
    #     sigma = self.reward_cfg.get(\"end_effector_tracking_sigma\", 0.1)
    #     return torch.exp(-error / sigma)
