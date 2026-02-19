"""G1TrackingEnv — BeyondMimic-style motion-tracking environment.

Observation vector (138 dims):
  robot_anchor_ori_w      6   2-column rotation matrix of robot anchor (heading)
  robot_anchor_lin_vel_w  3   pelvis linear velocity (world frame)
  robot_anchor_ang_vel_w  3   pelvis angular velocity (world frame)
  robot_body_pos_b       39   13 bodies × 3 – positions in robot anchor frame
  robot_body_ori_b       78   13 bodies × 6 – 2-col rotation mats in anchor frame
  motion_anchor_pos_b     3   motion anchor position relative to robot anchor
  motion_anchor_ori_b     6   motion anchor orientation relative to robot anchor
  ─────────────────────────
  Total                 138

Reward terms (auto-discovered by _reward_ prefix, all scaled × dt):
  GlobalAnchorPositionTracking     – global pelvis XY tracking
  GlobalAnchorOrientationTracking  – global pelvis heading tracking
  RelativeBodyPositionTracking     – body pose shape (mean over 13 bodies)
  RelativeBodyOrientationTracking  – body orientation shape
  GlobalBodyLinVelTracking         – body linear velocity matching
  GlobalBodyAngVelTracking         – body angular velocity matching
  FeetContactTime                  – reward sustained foot contact
  AlivBonus                        – constant per-step bonus
  ActionRatePenalty                – penalise action jerk
  DofLimitPenalty                  – penalise joint limit violations
"""

from __future__ import annotations

import math

import torch

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from tracking.motion_lib import (
    MotionLib,
    yaw_quat_from_quat,
    rot_mat_2col,
    quat_error_magnitude,
    subtract_frame_transforms,
    _rotate_by_quat,
    _inv_quat,
)


class G1TrackingEnv:
    """Motion-tracking environment for the G1 12-DOF humanoid.

    Does NOT inherit G1Env – clean-slate implementation that shares the same
    Genesis scene setup and PD-control idioms.
    """

    def __init__(
        self,
        num_envs: int,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        motion_cfg: dict,
        show_viewer: bool = False,
    ):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]           # 138
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]   # 12
        self.device = gs.device

        self.simulate_action_latency = True
        self.dt = 0.02                              # 50 Hz control
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg   = env_cfg
        self.obs_cfg   = obs_cfg
        self.reward_cfg = reward_cfg
        self.motion_cfg = motion_cfg

        self.obs_scales   = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # ── Genesis scene ────────────────────────────────────────────────────
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=30,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="../g1_files/g1_12dof_package/g1_12dof/g1_12dof.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            )
        )

        self.scene.build(n_envs=num_envs)

        # ── Joint / DOF indices ───────────────────────────────────────────────
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # ── PD gains ─────────────────────────────────────────────────────────
        kp_values, kd_values = [], []
        for name in self.env_cfg["joint_names"]:
            if "hip_yaw" in name:
                kp_values.append(self.env_cfg["stiffness"]["hip_yaw"])
                kd_values.append(self.env_cfg["damping"]["hip_yaw"])
            elif "hip_roll" in name:
                kp_values.append(self.env_cfg["stiffness"]["hip_roll"])
                kd_values.append(self.env_cfg["damping"]["hip_roll"])
            elif "hip_pitch" in name:
                kp_values.append(self.env_cfg["stiffness"]["hip_pitch"])
                kd_values.append(self.env_cfg["damping"]["hip_pitch"])
            elif "knee" in name:
                kp_values.append(self.env_cfg["stiffness"]["knee"])
                kd_values.append(self.env_cfg["damping"]["knee"])
            elif "ankle" in name:
                kp_values.append(self.env_cfg["stiffness"]["ankle"])
                kd_values.append(self.env_cfg["damping"]["ankle"])
        self.robot.set_dofs_kp(kp_values, self.motors_dof_idx)
        self.robot.set_dofs_kv(kd_values, self.motors_dof_idx)

        # ── Global constants ─────────────────────────────────────────────────
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.init_base_pos  = torch.tensor(env_cfg["base_init_pos"],  dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)

        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            dtype=gs.tc_float, device=gs.device,
        )

        # ── Body link indices for all 13 bodies ──────────────────────────────
        # Genesis body order matches NPZ body_names order (both from the URDF).
        self.num_bodies = len(env_cfg["body_names"])
        self.body_link_idx = torch.tensor(
            [self.robot.get_link(name).idx_local for name in env_cfg["body_names"]],
            dtype=gs.tc_int, device=gs.device,
        )

        # Foot link indices for contact tracking (last 2 bodies: ankle roll links)
        self.feet_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.feet_link_idx = torch.tensor(
            [self.robot.get_link(name).idx_local for name in self.feet_names],
            dtype=gs.tc_int, device=gs.device,
        )

        # ── MotionLib ─────────────────────────────────────────────────────────
        self.motion_lib = MotionLib(motion_cfg["file"], device=gs.device)
        self.joint_idx_map = self.motion_lib.build_joint_idx_map(env_cfg["joint_names"])

        # ── Buffers ──────────────────────────────────────────────────────────
        N = num_envs
        B = self.num_bodies
        f = gs.tc_float
        d = gs.device

        self.obs_buf           = torch.zeros((N, self.num_obs),   dtype=f, device=d)
        self.rew_buf           = torch.zeros((N,),                dtype=f, device=d)
        self.reset_buf         = torch.ones((N,),                 dtype=gs.tc_bool, device=d)
        self.episode_length_buf = torch.zeros((N,),               dtype=gs.tc_int,  device=d)

        # Robot state
        self.base_pos          = torch.zeros((N, 3),    dtype=f, device=d)
        self.base_quat         = torch.zeros((N, 4),    dtype=f, device=d)
        self.base_lin_vel      = torch.zeros((N, 3),    dtype=f, device=d)
        self.base_ang_vel      = torch.zeros((N, 3),    dtype=f, device=d)
        self.projected_gravity = torch.zeros((N, 3),    dtype=f, device=d)
        self.base_euler        = torch.zeros((N, 3),    dtype=f, device=d)

        self.dof_pos           = torch.zeros((N, 12),   dtype=f, device=d)
        self.dof_vel           = torch.zeros((N, 12),   dtype=f, device=d)
        self.actions           = torch.zeros((N, 12),   dtype=f, device=d)
        self.last_actions      = torch.zeros((N, 12),   dtype=f, device=d)

        # All-body state (N, 13, ...)
        self.all_body_pos      = torch.zeros((N, B, 3), dtype=f, device=d)
        self.all_body_quat     = torch.zeros((N, B, 4), dtype=f, device=d)
        self.all_body_lin_vel  = torch.zeros((N, B, 3), dtype=f, device=d)
        self.all_body_ang_vel  = torch.zeros((N, B, 3), dtype=f, device=d)

        # Robot anchor (pelvis XY + yaw)
        self.robot_anchor_pos  = torch.zeros((N, 3),    dtype=f, device=d)
        self.robot_anchor_quat = torch.zeros((N, 4),    dtype=f, device=d)
        self.robot_anchor_quat[:, 0] = 1.0   # identity

        # Motion-reference state (set each step by get_frame)
        self.ref_anchor_pos_w          = torch.zeros((N, 3),    dtype=f, device=d)
        self.ref_anchor_quat_w         = torch.zeros((N, 4),    dtype=f, device=d)
        self.ref_body_pos_relative_w   = torch.zeros((N, B, 3), dtype=f, device=d)
        self.ref_body_quat_relative_w  = torch.zeros((N, B, 4), dtype=f, device=d)
        self.ref_body_lin_vel_w        = torch.zeros((N, B, 3), dtype=f, device=d)
        self.ref_body_ang_vel_w        = torch.zeros((N, B, 3), dtype=f, device=d)

        # Robot body-relative state (computed each step for obs + rewards)
        self.robot_body_pos_relative_w  = torch.zeros((N, B, 3), dtype=f, device=d)
        self.robot_body_quat_relative_w = torch.zeros((N, B, 4), dtype=f, device=d)

        # Motion time per env
        self.motion_time   = torch.zeros((N,),  dtype=f, device=d)
        self.start_frame   = torch.zeros((N,),  dtype=gs.tc_int, device=d)

        # Contact tracking for FeetContactTime reward
        self.last_contacts         = torch.zeros((N, 2), dtype=gs.tc_bool, device=d)
        self.feet_contact_time     = torch.zeros((N, 2), dtype=f,          device=d)
        # how long the foot was in contact before it last lifted
        self.last_contact_duration = torch.zeros((N, 2), dtype=f,          device=d)
        # True on the step a foot transitions from contact → air (set by _update_contact_state)
        self.first_air             = torch.zeros((N, 2), dtype=gs.tc_bool, device=d)

        self.extras = {"observations": {}}

        # ── Reward function discovery ────────────────────────────────────────
        self.reward_functions, self.episode_sums = {}, {}
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((N,), dtype=f, device=d)

    # ────────────────────────────────────────────────────────────────────────
    # rsl_rl interface
    # ────────────────────────────────────────────────────────────────────────

    def reset(self):
        self._reset_idx(None)
        self._update_observation()
        return self.obs_buf, None

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # ────────────────────────────────────────────────────────────────────────
    # Step
    # ────────────────────────────────────────────────────────────────────────

    def step(self, actions: torch.Tensor):
        self.actions = actions.clamp(
            -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(
            target_dof_pos[:, self.actions_dof_idx], slice(6, 18)
        )
        self.scene.step()

        # ── Update robot state buffers ─────────────────────────────────────
        self.episode_length_buf += 1

        self.base_pos  = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True, degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel      = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel      = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # All body states (N, 13, ...)
        self.all_body_pos     = self.robot.get_links_pos(self.body_link_idx)
        self.all_body_quat    = self.robot.get_links_quat(self.body_link_idx)
        self.all_body_lin_vel = self.robot.get_links_vel(self.body_link_idx)
        self.all_body_ang_vel = self.robot.get_links_ang(self.body_link_idx)

        # ── Robot anchor frame ────────────────────────────────────────────
        self.robot_anchor_pos  = self.base_pos.clone()
        self.robot_anchor_pos[:, 2] = 0.0
        self.robot_anchor_quat = yaw_quat_from_quat(self.base_quat)

        # ── Robot body-relative state ─────────────────────────────────────
        # body_pos_relative_w[i] = R_anchor^T @ (body_pos_w[i] - anchor_pos)
        N, B = self.num_envs, self.num_bodies
        inv_anchor_q = _inv_quat(self.robot_anchor_quat)  # (N, 4)
        delta_pos = self.all_body_pos - self.robot_anchor_pos[:, None, :]   # (N, B, 3)
        inv_aq_exp = inv_anchor_q[:, None, :].expand(N, B, 4).reshape(N * B, 4)
        self.robot_body_pos_relative_w = _rotate_by_quat(
            delta_pos.reshape(N * B, 3), inv_aq_exp
        ).reshape(N, B, 3)

        # body_quat_relative_w = inv_anchor_quat * body_quat
        from tracking.motion_lib import _quat_mul
        bq_flat = self.all_body_quat.reshape(N * B, 4)
        self.robot_body_quat_relative_w = _quat_mul(inv_aq_exp, bq_flat).reshape(N, B, 4)

        # ── Advance motion time and get reference frame ───────────────────
        self.motion_time = (self.motion_time + self.dt).clamp(
            max=self.motion_lib.duration
        )
        ref = self.motion_lib.get_frame(self.motion_time)
        self.ref_anchor_pos_w         = ref["anchor_pos_w"]
        self.ref_anchor_quat_w        = ref["anchor_quat_w"]
        self.ref_body_pos_relative_w  = ref["body_pos_relative_w"]
        self.ref_body_quat_relative_w = ref["body_quat_relative_w"]
        self.ref_body_lin_vel_w       = ref["body_lin_vel_w"]
        self.ref_body_ang_vel_w       = ref["body_ang_vel_w"]

        # ── Contact tracking ─────────────────────────────────────────────
        self._update_contact_state()

        # ── Rewards ──────────────────────────────────────────────────────
        self.rew_buf.zero_()
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ── Termination ──────────────────────────────────────────────────
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg[
            "termination_if_pitch_greater_than"
        ]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg[
            "termination_if_roll_greater_than"
        ]
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg[
            "termination_if_base_height_less_than"
        ]
        # End of motion sequence
        self.reset_buf |= self.motion_time >= self.motion_lib.duration

        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ────────────────────────────────────────────────────────────────────────
    # Reset
    # ────────────────────────────────────────────────────────────────────────

    def _reset_idx(self, envs_idx):
        """Reset environments at envs_idx (bool mask or None for all)."""
        if envs_idx is None:
            # Full reset — reset everything
            n_reset = self.num_envs
            reset_mask = None
            indices = None
        else:
            n_reset = int(envs_idx.sum().item())
            if n_reset == 0:
                return
            reset_mask = envs_idx
            indices = envs_idx.nonzero(as_tuple=False).squeeze(-1)

        # Sample random start frames, ensuring at least 2 s of motion remain.
        # Sampling all the way to the end produces 6-step episodes that give
        # near-zero gradient signal and waste compute.
        min_ep_frames = int(2.0 * self.motion_lib.fps)   # 100 frames at fps=50
        max_start = max(1, self.motion_lib.num_frames - 1 - min_ep_frames)
        new_start_frames = torch.randint(
            0,
            max_start,
            (n_reset,),
            device=gs.device,
        )
        new_times = new_start_frames.float() / self.motion_lib.fps
        init_state = self.motion_lib.get_init_state(new_start_frames)

        # Build full qpos: [base_pos(3), base_quat(4), joint_pos(12)]
        joint_pos_reordered = init_state["joint_pos"][:, self.joint_idx_map]
        qpos = torch.cat(
            [init_state["base_pos"], init_state["base_quat"], joint_pos_reordered],
            dim=-1,
        )

        if indices is None:
            self.robot.set_qpos(qpos, zero_velocity=True, skip_forward=True)
        else:
            self.robot.set_qpos(qpos, envs_idx=indices, zero_velocity=True, skip_forward=True)

        # Restore base lin/ang velocity from reference.
        # Base DOFs are 0-5: [lin_vel x,y,z, ang_vel x,y,z] (MuJoCo/Genesis free-joint convention).
        # set_qpos(..., zero_velocity=True) zeroed them; we restore them here so
        # episodes starting mid-motion (e.g. during jump) have correct momentum.
        base_vel = torch.cat(
            [init_state["base_lin_vel"], init_state["base_ang_vel"]], dim=-1
        )  # (n_reset, 6)
        base_dof_idx = torch.arange(6, dtype=gs.tc_int, device=gs.device)
        if indices is None:
            self.robot.set_dofs_velocity(base_vel, base_dof_idx)
        else:
            self.robot.set_dofs_velocity(base_vel, base_dof_idx, envs_idx=indices)

        # Set joint velocities from reference
        joint_vel_reordered = init_state["joint_vel"][:, self.joint_idx_map]
        if indices is None:
            self.robot.set_dofs_velocity(joint_vel_reordered, self.motors_dof_idx)
        else:
            self.robot.set_dofs_velocity(
                joint_vel_reordered, self.motors_dof_idx, envs_idx=indices
            )

        # Update time buffers
        if reset_mask is None:
            self.motion_time.copy_(new_times)
            self.start_frame.copy_(new_start_frames)
            self.actions.zero_()
            self.last_actions.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            self.feet_contact_time.zero_()
            self.last_contacts.zero_()
            self.last_contact_duration.zero_()
            self.first_air.zero_()
        else:
            self.motion_time[reset_mask] = new_times
            self.start_frame[reset_mask] = new_start_frames
            self.actions.masked_fill_(reset_mask[:, None], 0.0)
            self.last_actions.masked_fill_(reset_mask[:, None], 0.0)
            self.episode_length_buf.masked_fill_(reset_mask, 0)
            self.reset_buf.masked_fill_(reset_mask, True)
            self.feet_contact_time.masked_fill_(reset_mask[:, None], 0.0)
            self.last_contacts.masked_fill_(reset_mask[:, None], False)
            self.last_contact_duration.masked_fill_(reset_mask[:, None], 0.0)
            self.first_air.masked_fill_(reset_mask[:, None], False)

        # Log episode sums
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if reset_mask is None:
                mean_val = value.mean()
                value.zero_()
            else:
                n = max(n_reset, 1)
                mean_val = value[reset_mask].sum() / n
                value.masked_fill_(reset_mask, 0.0)
            self.extras["episode"]["rew_" + key] = (
                mean_val / self.env_cfg["episode_length_s"]
            )

    # ────────────────────────────────────────────────────────────────────────
    # Observation
    # ────────────────────────────────────────────────────────────────────────

    def _update_observation(self):
        """Build 138-dim BeyondMimic observation vector.

        Scaling applied to non-bounded components so all obs entries land in
        approximately [−1, +1] for the first network layer:
          lin_vel   × 0.2   (±5 m/s walking+jump → ±1.0)
          ang_vel   × 0.1   (±10 rad/s landing   → ±1.0)
          body_pos  × 0.5   (±1.5 m anchor frame  → ±0.75)
          anchor_pos× 0.2   (±2 m drift possible  → ±0.4)
        Rotation-matrix components are already in [−1, +1] — no scaling.
        """
        N, B = self.num_envs, self.num_bodies

        lin_vel_scale  = self.obs_scales.get("lin_vel",    0.2)
        ang_vel_scale  = self.obs_scales.get("ang_vel",    0.1)
        body_pos_scale = self.obs_scales.get("body_pos",   0.5)
        anc_pos_scale  = self.obs_scales.get("anchor_pos", 0.2)

        # 1. robot_anchor_ori_w  (6) – 2-col rotation matrix of anchor heading
        anchor_ori = rot_mat_2col(self.robot_anchor_quat)                      # (N, 6)

        # 2. robot_anchor_lin_vel_w  (3) – pelvis linear velocity (world frame)
        robot_lin_vel_w = self.robot.get_vel() * lin_vel_scale                 # (N, 3)

        # 3. robot_anchor_ang_vel_w  (3) – pelvis angular velocity (world frame)
        robot_ang_vel_w = self.robot.get_ang() * ang_vel_scale                 # (N, 3)

        # 4. robot_body_pos_b  (N, 13, 3) → flatten to (N, 39)
        body_pos_b = (
            self.robot_body_pos_relative_w.reshape(N, B * 3) * body_pos_scale
        )

        # 5. robot_body_ori_b  (N, 13, 6) → flatten to (N, 78)
        body_ori_b = rot_mat_2col(
            self.robot_body_quat_relative_w.reshape(N * B, 4)
        ).reshape(N, B * 6)

        # 6. motion_anchor_pos_b  (3) – motion anchor position in robot anchor frame
        mot_anchor_pos_b, _ = subtract_frame_transforms(
            self.robot_anchor_pos,
            self.robot_anchor_quat,
            self.ref_anchor_pos_w,
        )                                                                         # (N, 3)
        mot_anchor_pos_b = mot_anchor_pos_b * anc_pos_scale

        # 7. motion_anchor_ori_b  (6) – motion anchor orientation in robot anchor frame
        _, mot_anchor_quat_b = subtract_frame_transforms(
            self.robot_anchor_pos,
            self.robot_anchor_quat,
            self.ref_anchor_pos_w,
            self.ref_anchor_quat_w,
        )
        mot_anchor_ori_b = rot_mat_2col(mot_anchor_quat_b)                       # (N, 6)

        self.obs_buf = torch.cat([
            anchor_ori,          # 6
            robot_lin_vel_w,     # 3
            robot_ang_vel_w,     # 3
            body_pos_b,          # 39
            body_ori_b,          # 78
            mot_anchor_pos_b,    # 3
            mot_anchor_ori_b,    # 6
        ], dim=-1)               # 138

    # ────────────────────────────────────────────────────────────────────────
    # Contact state
    # ────────────────────────────────────────────────────────────────────────

    def _update_contact_state(self):
        """Track per-foot contact duration for FeetContactTime reward.

        Genesis batch API always returns (N, num_links, 3) — never 2D.
        Using all_cf[:, link_idx, :] consistently avoids the broken dim==2
        fallback which produced shape (1, 3) → norm → (1,) instead of (N,).
        """
        all_cf = self.robot.get_links_net_contact_force()  # (N, num_links, 3)
        for i, link_idx in enumerate(self.feet_link_idx):
            force = all_cf[:, link_idx, :]                  # (N, 3)
            in_contact = torch.norm(force, dim=-1) > 5.0    # (N,)

            # Detect lift-off BEFORE resetting the timer (was in contact, now is not)
            first_air = self.last_contacts[:, i] & ~in_contact
            self.first_air[:, i] = first_air

            # Save accumulated contact duration at the moment of lift-off
            self.last_contact_duration[:, i] = torch.where(
                first_air,
                self.feet_contact_time[:, i],
                self.last_contact_duration[:, i],
            )

            # Accumulate contact time while in contact; reset on lift-off
            self.feet_contact_time[:, i] = torch.where(
                in_contact,
                self.feet_contact_time[:, i] + self.dt,
                torch.zeros_like(self.feet_contact_time[:, i]),
            )

            self.last_contacts[:, i] = in_contact

    # ────────────────────────────────────────────────────────────────────────
    # Reward functions  (all return (N,) tensor; scales multiplied externally)
    # ────────────────────────────────────────────────────────────────────────

    def _reward_GlobalAnchorPositionTracking(self) -> torch.Tensor:
        """exp(-||motion_anchor_pos - robot_anchor_pos||² / std²)."""
        std = self.reward_cfg.get("anchor_pos_std", 0.5)
        err = torch.sum(
            torch.square(self.ref_anchor_pos_w - self.robot_anchor_pos), dim=-1
        )
        return torch.exp(-err / (std * std))

    def _reward_GlobalAnchorOrientationTracking(self) -> torch.Tensor:
        """exp(-quat_err(anchor_quat, robot_anchor_quat)² / std²)."""
        std = self.reward_cfg.get("anchor_ori_std", 0.5)
        err = quat_error_magnitude(self.ref_anchor_quat_w, self.robot_anchor_quat) ** 2
        return torch.exp(-err / (std * std))

    def _reward_RelativeBodyPositionTracking(self) -> torch.Tensor:
        """exp(-mean_bodies(||ref_body_pos_rel - robot_body_pos_rel||²) / std²)."""
        std = self.reward_cfg.get("body_pos_std", 0.1)
        err = torch.sum(
            torch.square(
                self.ref_body_pos_relative_w - self.robot_body_pos_relative_w
            ),
            dim=-1,
        )  # (N, 13)
        return torch.exp(-err.mean(-1) / (std * std))

    def _reward_RelativeBodyOrientationTracking(self) -> torch.Tensor:
        """exp(-mean_bodies(quat_err²) / std²)."""
        std = self.reward_cfg.get("body_ori_std", 0.5)
        N, B = self.num_envs, self.num_bodies
        err = quat_error_magnitude(
            self.ref_body_quat_relative_w.reshape(N * B, 4),
            self.robot_body_quat_relative_w.reshape(N * B, 4),
        ).reshape(N, B) ** 2
        return torch.exp(-err.mean(-1) / (std * std))

    def _reward_GlobalBodyLinVelTracking(self) -> torch.Tensor:
        """exp(-mean_bodies(||ref_lin_vel - robot_lin_vel||²) / std²)."""
        std = self.reward_cfg.get("body_lin_vel_std", 0.5)
        err = torch.sum(
            torch.square(self.ref_body_lin_vel_w - self.all_body_lin_vel), dim=-1
        )  # (N, 13)
        return torch.exp(-err.mean(-1) / (std * std))

    def _reward_GlobalBodyAngVelTracking(self) -> torch.Tensor:
        """exp(-mean_bodies(||ref_ang_vel - robot_ang_vel||²) / std²)."""
        std = self.reward_cfg.get("body_ang_vel_std", 1.0)
        err = torch.sum(
            torch.square(self.ref_body_ang_vel_w - self.all_body_ang_vel), dim=-1
        )  # (N, 13)
        return torch.exp(-err.mean(-1) / (std * std))

    def _reward_FeetContactTime(self) -> torch.Tensor:
        """Reward feet that sustained contact above a minimum threshold.
        Fires on the step the foot lifts (first_air) if the preceding
        contact duration met or exceeded the threshold.
        """
        threshold = self.reward_cfg.get("feet_contact_threshold", 0.3)
        reward = torch.sum(
            (self.last_contact_duration >= threshold).float() * self.first_air.float(),
            dim=-1,
        )
        return reward

    def _reward_FallPenalty(self) -> torch.Tensor:
        """One-shot penalty on the step the robot falls.

        Fires when any fall condition is met — same thresholds as termination.
        Returns −1.0 for fallen envs, 0.0 otherwise.  With scale=100.0 the
        effective penalty is −2.0 per fall (≈ 4× the max per-step positive reward).
        The robot resets immediately after, so this fires exactly once per fall.
        """
        fall = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_base_height_less_than"])
        )
        return -fall.float()

    def _reward_AlivBonus(self) -> torch.Tensor:
        return torch.ones(self.num_envs, dtype=gs.tc_float, device=gs.device)

    def _reward_ActionRatePenalty(self) -> torch.Tensor:
        return -torch.sum(torch.square(self.actions - self.last_actions), dim=-1)

    def _reward_DofLimitPenalty(self) -> torch.Tensor:
        lower, upper = self.robot.get_dofs_limit()
        lo = lower[self.motors_dof_idx]
        hi = upper[self.motors_dof_idx]
        soft = 0.9
        violation = -(self.dof_pos - lo * soft).clamp(max=0.0)
        violation += (self.dof_pos - hi * soft).clamp(min=0.0)
        return -torch.sum(violation, dim=-1)
