import math

import torch

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class G1Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, randomization_cfg=None, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # 1-step latency matches real robot
        self.dt = 0.02                       # 50 Hz control
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.randomization_cfg = randomization_cfg or {}

        self.enable_command_resampling = env_cfg.get("enable_command_resampling", True)

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # ── Genesis scene ─────────────────────────────────────────────────────
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
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
            ),
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

        self.robot.set_dofs_kp(kp_values, self.motors_dof_idx)
        self.robot.set_dofs_kv(kd_values, self.motors_dof_idx)

        # ── Global constants ──────────────────────────────────────────────────
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # ── Buffers ───────────────────────────────────────────────────────────
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.obs_buf = torch.empty((self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits = [
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        ]
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.extras = dict()
        self.extras["observations"] = dict()

        # ── Feet tracking ─────────────────────────────────────────────────────
        self.feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.feet_link_idx = torch.tensor(
            [self.robot.get_link(name).idx_local for name in self.feet_links],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.feet_pos = torch.zeros((self.num_envs, 2, 3), dtype=gs.tc_float, device=gs.device)
        self.feet_heights = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.feet_vel = torch.zeros((self.num_envs, 2, 3), dtype=gs.tc_float, device=gs.device)
        self.contact_forces = torch.zeros((self.num_envs, 2, 3), dtype=gs.tc_float, device=gs.device)
        self.foot_contact_weighted = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)

        # ── Air time and contact tracking ─────────────────────────────────────
        self.feet_air_time = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.last_contacts = torch.zeros((self.num_envs, 2), dtype=gs.tc_bool, device=gs.device)
        self.feet_first_contact = torch.zeros((self.num_envs, 2), dtype=gs.tc_bool, device=gs.device)
        self.latched_air_time = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)

        # ── Gait phase tracking ───────────────────────────────────────────────
        # Each foot has an independent phase [0, 1): 0–0.4 = stance, 0.4–1.0 = swing.
        # Left and right are initialized 0.5 apart (antiphase) for walking gait.
        self.foot_phases = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.gait_period = 1.0

        # Desired joint angles computed from gait phase each step
        self.desired_ankle_angles = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.desired_knee_angles = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        # left_ankle_pitch_joint (4), right_ankle_pitch_joint (10)
        self.ankle_joint_indices = [4, 10]

        # ── Domain randomization ──────────────────────────────────────────────
        self.add_noise = self.randomization_cfg.get("add_noise", True)
        self.noise_scale_vec = torch.zeros(self.num_obs, dtype=gs.tc_float, device=gs.device)

        noise_scales = self.randomization_cfg.get("noise_scales", {
            "dof_pos": 0.01,
            "dof_vel": 1.5,
            "ang_vel": 0.2,
            "gravity": 0.05,
            "commands": 0.0,
        })

        # Obs layout: ang_vel(0:3) | gravity(3:6) | commands(6:9) | dof_pos(9:21) | dof_vel(21:33) | actions(33:45)
        self.noise_scale_vec[0:3]  = noise_scales["ang_vel"]
        self.noise_scale_vec[3:6]  = noise_scales["gravity"]
        self.noise_scale_vec[6:9]  = noise_scales["commands"]
        self.noise_scale_vec[9:21] = noise_scales["dof_pos"]
        self.noise_scale_vec[21:33] = noise_scales["dof_vel"]

        self.push_robots = self.randomization_cfg.get("push_robots", False)
        self.push_interval_s = self.randomization_cfg.get("push_interval_s", 15)
        self.last_push_step = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)

        self.randomize_friction = self.randomization_cfg.get("randomize_friction", False)
        self.randomize_base_mass = self.randomization_cfg.get("randomize_base_mass", False)
        self.randomize_motor_strength = self.randomization_cfg.get("randomize_motor_strength", False)

        self.kp_scale = torch.ones((self.num_envs, 1), dtype=gs.tc_float, device=gs.device)
        self.kd_scale = torch.ones((self.num_envs, 1), dtype=gs.tc_float, device=gs.device)

        # ── Reward function discovery ─────────────────────────────────────────
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        self._randomize_physics_props()

    # ────────────────────────────────────────────────────────────────────────
    # rsl_rl interface
    # ────────────────────────────────────────────────────────────────────────

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.obs_buf, None

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf

        if self.add_noise:
            noise = (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            return self.obs_buf + noise, self.extras

        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # ────────────────────────────────────────────────────────────────────────
    # Step
    # ────────────────────────────────────────────────────────────────────────

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        # ── Update robot state ────────────────────────────────────────────────
        self.episode_length_buf += 1
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

        # ── Update feet state ─────────────────────────────────────────────────
        for i, link_name in enumerate(self.feet_links):
            link = self.robot.get_link(link_name)
            self.feet_pos[:, i, :] = link.get_pos()
            self.feet_vel[:, i, :] = link.get_vel()
        self.feet_heights[:, 0] = self.feet_pos[:, 0, 2]
        self.feet_heights[:, 1] = self.feet_pos[:, 1, 2]

        # ── Update contact and air time ───────────────────────────────────────
        all_contact_forces = self.robot.get_links_net_contact_force()

        for i, link_idx in enumerate(self.feet_link_idx):
            contact_force = all_contact_forces[:, link_idx, :]
            self.contact_forces[:, i, :] = contact_force
            contact_norm = torch.norm(contact_force, dim=-1)
            in_contact = contact_norm > 5.0
            self.foot_contact_weighted[:, i] = torch.clamp(contact_norm / 100.0, 0.0, 1.0)

            self.feet_air_time[:, i] += self.dt

            first_contact = in_contact & ~self.last_contacts[:, i]
            self.feet_first_contact[:, i] = first_contact

            # Latch air time at the moment of touchdown (before resetting the timer)
            self.latched_air_time[:, i] = torch.where(
                first_contact,
                self.feet_air_time[:, i],
                self.latched_air_time[:, i]
            )

            self.feet_air_time[:, i] *= ~in_contact
            self.last_contacts[:, i] = in_contact

        # ── Gait phase and desired joint angles ───────────────────────────────
        self._update_foot_phases()
        self._compute_desired_ankle_angles()
        self._compute_desired_knee_angles()

        # ── Rewards ───────────────────────────────────────────────────────────
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.push_robots:
            self._push_robots()

        if self.enable_command_resampling:
            self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # ── Termination ───────────────────────────────────────────────────────
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_base_height_less_than"]

        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ────────────────────────────────────────────────────────────────────────
    # Reset
    # ────────────────────────────────────────────────────────────────────────

    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
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
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # Phase reset: left foot at 0.1 (flat-foot), right at 0.6 (antiphase offset 0.5).
        # Starting at 0.1 avoids an immediate ankle penalty spike that phase 0.0 (heel-strike) would trigger.
        if envs_idx is None:
            self.foot_phases[:, 0] = 0.1
            self.foot_phases[:, 1] = 0.6
            self.desired_ankle_angles.zero_()
            self.desired_knee_angles.zero_()
        else:
            self.foot_phases[envs_idx, 0] = 0.1
            self.foot_phases[envs_idx, 1] = 0.6
            self.desired_ankle_angles.masked_fill_(envs_idx[:, None], 0.0)
            self.desired_knee_angles.masked_fill_(envs_idx[:, None], 0.0)

        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            value.masked_fill_(envs_idx, 0.0)

        self._resample_commands(envs_idx)

    # ────────────────────────────────────────────────────────────────────────
    # Observation
    # ────────────────────────────────────────────────────────────────────────

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],                      # 3
                self.projected_gravity,                                               # 3
                self.commands * self.commands_scale,                                  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],                            # 12
                self.actions,                                                         # 12
            ),
            dim=-1,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Commands
    # ────────────────────────────────────────────────────────────────────────

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        # Zero commands 10% of the time to encourage standing and better foot tracking rewards
        zero_mask = torch.rand((self.num_envs,), device=gs.device) < 0.1
        commands[zero_mask] = 0.0
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    # ────────────────────────────────────────────────────────────────────────
    # Domain randomization
    # ────────────────────────────────────────────────────────────────────────

    def _randomize_physics_props(self):
        """Randomize motor gains, mass, and friction. Called once at init."""
        if not (self.randomize_friction or self.randomize_base_mass or self.randomize_motor_strength):
            return

        if self.randomize_motor_strength:
            stiffness_range = self.randomization_cfg.get("motor_strength_range", [0.8, 1.2])
            damping_range = self.randomization_cfg.get("motor_strength_range", [0.8, 1.2])
            self.kp_scale = torch.rand(self.num_envs, 1, dtype=gs.tc_float, device=gs.device) * \
                            (stiffness_range[1] - stiffness_range[0]) + stiffness_range[0]
            self.kd_scale = torch.rand(self.num_envs, 1, dtype=gs.tc_float, device=gs.device) * \
                            (damping_range[1] - damping_range[0]) + damping_range[0]

        # Mass and friction randomization require Genesis API support; stubs retained
        if self.randomize_base_mass:
            pass

        if self.randomize_friction:
            pass

    def _push_robots(self):
        """Apply random XY velocity perturbations to the robot base periodically."""
        push_interval = int(self.push_interval_s / self.dt)
        envs_to_push = (self.episode_length_buf - self.last_push_step) >= push_interval

        if envs_to_push.any():
            max_vel_change = self.randomization_cfg.get("max_push_vel_xy", 1.0)
            xy_perturbation = (torch.rand(self.num_envs, 2, dtype=gs.tc_float, device=gs.device) * 2 - 1)
            xy_perturbation[~envs_to_push] = 0.0

            curr_vel = self.robot.get_vel()
            curr_vel[:, :2] += xy_perturbation * max_vel_change
            self.robot.set_vel(curr_vel)

            self.last_push_step[envs_to_push] = self.episode_length_buf[envs_to_push]

    # ────────────────────────────────────────────────────────────────────────
    # Gait phase and biomechanical joint trajectories
    # ────────────────────────────────────────────────────────────────────────

    def _update_foot_phases(self):
        """Advance foot phases and snap to stance-start on touchdown.

        Phase [0, 0.4) = stance, [0.4, 1.0) = swing.
        Adaptive period: 1.5 s at rest → 1.0 s at 1.5 m/s.
        Both feet snap to phase 0.0 on first contact when the phase is within
        circular distance 0.3 of that target, correcting small drift.
        """
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        speed_ratio = torch.clamp(cmd_mag / 1.5, 0.0, 1.0)
        adaptive_period = 1.5 - (speed_ratio * 0.50)
        phase_increment = self.dt / adaptive_period
        is_moving = cmd_mag > 0.1

        self.foot_phases += phase_increment.unsqueeze(1) * is_moving.unsqueeze(1)
        self.foot_phases = torch.fmod(self.foot_phases, 1.0)

        snap_target = torch.tensor(0.0, device=gs.device, dtype=gs.tc_float)
        for i in range(2):
            phase_dist = torch.minimum(
                torch.abs(self.foot_phases[:, i] - snap_target),
                1.0 - torch.abs(self.foot_phases[:, i] - snap_target)
            )
            near_target = phase_dist < 0.3
            should_snap = self.feet_first_contact[:, i] & near_target & is_moving
            self.foot_phases[:, i] = torch.where(should_snap, snap_target, self.foot_phases[:, i])

    def _compute_desired_ankle_angles(self):
        """Compute desired ankle pitch angles for heel-strike to toe-off motion.

        Seven-segment biomechanical trajectory per foot:
          heel_angle    = +0.08 rad  (dorsiflexion at heel strike, ~5 deg)
          pushoff_angle = -0.28 rad  (plantarflexion at toe-off, ~16 deg)
          swing_angle   = +0.07 rad  (dorsiflexion for foot clearance)
          land_angle    = +0.05 rad  (dorsiflexion held during pre-landing)
        """
        heel_angle    =  0.08
        pushoff_angle = -0.28
        swing_angle   =  0.07
        land_angle    =  0.05

        for i in range(2):
            phase = self.foot_phases[:, i]

            hs_mask       = phase < 0.05
            hs_flat_mask  = (phase >= 0.05) & (phase < 0.08)
            hs_flat_angle = heel_angle * (1.0 - (phase - 0.05) / 0.03)
            flat_mask     = (phase >= 0.08) & (phase < 0.33)
            pushoff_mask  = (phase >= 0.33) & (phase < 0.40)
            pushoff_ramp  = pushoff_angle * ((phase - 0.33) / 0.07)
            early_swing_mask  = (phase >= 0.40) & (phase < 0.55)
            early_prog        = (phase - 0.40) / 0.15
            early_swing_angle = pushoff_angle + early_prog * (swing_angle - pushoff_angle)
            mid_swing_mask = (phase >= 0.55) & (phase < 0.65)
            landing_mask   = phase >= 0.65

            desired_angle = torch.zeros_like(phase)
            desired_angle = torch.where(hs_mask,          heel_angle,        desired_angle)
            desired_angle = torch.where(hs_flat_mask,     hs_flat_angle,     desired_angle)
            desired_angle = torch.where(flat_mask,        0.0,               desired_angle)
            desired_angle = torch.where(pushoff_mask,     pushoff_ramp,      desired_angle)
            desired_angle = torch.where(early_swing_mask, early_swing_angle, desired_angle)
            desired_angle = torch.where(mid_swing_mask,   swing_angle,       desired_angle)
            desired_angle = torch.where(landing_mask,     land_angle,        desired_angle)

            self.desired_ankle_angles[:, i] = desired_angle

    def _compute_desired_knee_angles(self):
        """Compute desired knee flexion based on human gait biomechanics.

        G1 knee convention: 0.0 = straight, positive = bent.
        Three-segment profile:
          Stance  (0.0–0.4):  sin bump, peak ~0.35 rad at phase 0.20
          Flex    (0.4–0.65): sin rise to ~1.1 rad at phase 0.65
          Extend  (0.65–1.0): (1−t)^1.5 power-law drop — synchronized with ankle
                               pre-landing dorsiflexion; terminal target 0.05 rad
        """
        self.desired_knee_angles.zero_()

        for i in range(2):
            phase = self.foot_phases[:, i]

            stance_mask     = phase < 0.4
            stance_progress = (phase / 0.4) * math.pi
            stance_angle    = 0.1 + 0.25 * torch.sin(stance_progress)

            swing_flex_mask = (phase >= 0.4) & (phase < 0.65)
            flex_progress   = (phase - 0.4) / 0.25
            flex_angle      = 0.1 + 1.0 * torch.sin(flex_progress * math.pi / 2)

            ext_progress = (phase - 0.65) / 0.35
            ext_angle    = 0.05 + 1.05 * (1.0 - ext_progress) ** 1.5

            self.desired_knee_angles[:, i] = torch.where(
                stance_mask, stance_angle,
                torch.where(swing_flex_mask, flex_angle, ext_angle)
            )

    # ────────────────────────────────────────────────────────────────────────
    # Reward functions  (all return (N,) tensor; scales multiplied externally)
    # ────────────────────────────────────────────────────────────────────────

    def _reward_LinVelXYReward(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg.get("tracking_sigma", 0.25))

    def _reward_AngVelZReward(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg.get("tracking_sigma", 0.25))

    def _reward_LinVelZPenalty(self):
        return -torch.square(self.base_lin_vel[:, 2])

    def _reward_AngVelXYPenalty(self):
        return -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_OrientationPenalty(self):
        return -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_ActionRatePenalty(self):
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_moving = cmd_mag > 0.1
        scale = (torch.where(is_moving, 1.0, 10.0))
        return -torch.sum(torch.square(self.last_actions - self.actions), dim=1) * scale

    def _reward_ActionLimitPenalty(self):
        action_limit = self.reward_cfg.get("action_limit", 1.0)
        near_limit = (torch.abs(self.actions) > action_limit * 0.9).float()
        return -torch.sum(near_limit * torch.square(self.actions), dim=1)

    def _reward_HipYawPenalty(self):
        # left_hip_yaw_joint (2), right_hip_yaw_joint (8)
        return -torch.sum(torch.abs(self.dof_pos[:, [2, 8]]), dim=-1)

    def _reward_HipRollPenalty(self):
        # left_hip_roll_joint (1), right_hip_roll_joint (7)
        return -torch.sum(torch.abs(self.dof_pos[:, [1, 7]]), dim=-1)

    def _reward_BodyRollPenalty(self):
        return -torch.square(self.base_euler[:, 0] * math.pi / 180.0)

    def _reward_FeetAirTimePenalty(self):
        """Penalize air time deviation from target, applied at moment of touchdown."""
        target_air_time = self.reward_cfg.get("target_feet_air_time", 0.6)
        pen_air_time = torch.sum(
            torch.abs(self.latched_air_time - target_air_time) * self.feet_first_contact.float(), dim=1
        )
        return -pen_air_time

    def _reward_G1FeetSlidePenalty(self):
        on_ground = self.foot_contact_weighted > 0.1
        feet_vel_xy = torch.norm(self.feet_vel[:, :, :2], dim=-1)
        return -torch.sum(on_ground * torch.square(feet_vel_xy), dim=1)

    def _reward_FeetOrientationPenalty(self):
        left_foot_quat  = self.robot.get_link(self.feet_links[0]).get_quat()
        right_foot_quat = self.robot.get_link(self.feet_links[1]).get_quat()
        left_gravity_local  = transform_by_quat(self.global_gravity, inv_quat(left_foot_quat))
        right_gravity_local = transform_by_quat(self.global_gravity, inv_quat(right_foot_quat))
        # XY components of projected gravity should be zero when the foot is flat
        left_dev  = -torch.sum(torch.square(left_gravity_local[:, :2]),  dim=1)
        right_dev = -torch.sum(torch.square(right_gravity_local[:, :2]), dim=1)
        return left_dev + right_dev

    def _reward_DofPosLimitPenalty(self):
        all_lower_limits = self.robot.get_dofs_limit()[0]
        all_upper_limits = self.robot.get_dofs_limit()[1]
        lower_limits = all_lower_limits[self.motors_dof_idx]
        upper_limits = all_upper_limits[self.motors_dof_idx]
        soft_ratio = 0.9
        # Penalize if within 10% of a hard joint limit.
        # Example: limit=-1.0, soft=-0.9. If pos=-0.95, violation=0.05
        out_of_limits = -(self.dof_pos - lower_limits * soft_ratio).clip(max=0.0)
        out_of_limits += (self.dof_pos - upper_limits * soft_ratio).clip(min=0.0)
        return -torch.sum(out_of_limits, dim=1)

    def _reward_FootPhaseReward(self):
        """Reward matching foot contact state to expected gait phase.

        Phase < 0.4 = stance (foot should be grounded).
        Phase >= 0.4 = swing (foot should be airborne).
        Extra penalty when both feet are grounded during a phase where at least
        one should be in swing, preventing the lazy double-stance solution.
        """
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_moving = cmd_mag > 0.1
        stance_threshold = 0.4

        left_should_contact  = self.foot_phases[:, 0] < stance_threshold
        right_should_contact = self.foot_phases[:, 1] < stance_threshold
        left_in_contact  = self.last_contacts[:, 0]
        right_in_contact = self.last_contacts[:, 1]

        left_correct  = (left_should_contact  == left_in_contact).float()
        right_correct = (right_should_contact == right_in_contact).float()

        both_should_not_contact = ~left_should_contact | ~right_should_contact
        both_actually_contact   = left_in_contact & right_in_contact
        double_stance_penalty   = (both_should_not_contact & both_actually_contact).float()

        phase_alignment = (left_correct + right_correct) / 2.0 - double_stance_penalty
        return phase_alignment * is_moving

    def _reward_StandingKneeReward(self):
        """Reward default knee angle (0.3 rad) when standing still.

        Targets default_dof_pos to avoid fighting the PD controller at rest.
        """
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_standing = cmd_mag < 0.1
        # knee joints: left_knee_joint (3), right_knee_joint (9)
        knee_angles = self.dof_pos[:, [3, 9]]
        knee_error = torch.sum(torch.square(knee_angles - 0.3), dim=1)
        return torch.exp(-knee_error / 0.1) * is_standing

    def _reward_AnkleTrackingPenalty(self):
        """Penalize deviation from desired ankle angles for natural heel-strike motion.

        exp(-error / 0.1) − 1 per ankle: 0 when perfect, −1 when error is large.
        Max total penalty = −2 (both ankles fully off target). Only during walking.
        """
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_moving = cmd_mag > 0.1
        ankle_angles = self.dof_pos[:, self.ankle_joint_indices]
        # Per-ankle squared error, shape: (num_envs, 2)
        per_ankle_error = torch.square(ankle_angles - self.desired_ankle_angles)
        per_ankle_penalty = torch.exp(-per_ankle_error / 0.1) - 1.0
        penalty = torch.sum(per_ankle_penalty, dim=1)
        return penalty * is_moving

    def _reward_FootExtensionReward(self):
        """During swing, reward the foot being extended in front of the base.

        Corrects stomping gait by encouraging forward leg reach before heel-strike.
        Target forward distance scales with command speed. Swing gate: phase > 0.4.

        Target = 0.1 + 0.15 × cmd_x (m): 0.1 m at rest, 0.4 m at 2 m/s.
        Gaussian kernel, sigma=0.04: ~50% reward at 0.2 m error.
        Max total = 2.0 (both feet perfectly on target).
        """
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_moving = (cmd_mag > 0.1).float()
        inv_base_quat = inv_quat(self.base_quat)
        total_reward = torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)

        for i in range(2):
            phase    = self.foot_phases[:, i]
            in_swing = (phase > 0.4).float()
            rel_world  = self.feet_pos[:, i, :] - self.base_pos
            rel_body   = transform_by_quat(rel_world, inv_base_quat)
            foot_fwd   = rel_body[:, 0]  # X = forward in body frame
            target_fwd = 0.1 + 0.15 * torch.clamp(self.commands[:, 0], 0.0, 2.0)
            error = (foot_fwd - target_fwd) ** 2
            total_reward += torch.exp(-error / 0.04) * in_swing * is_moving

        return total_reward

    def _reward_StandStillVelocityPenalty(self):
        """Penalize any joint or base motion when commanded to stand still."""
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_standing = cmd_mag < 0.1
        total_penalty = (
            torch.sum(torch.square(self.dof_vel), dim=1)
            + torch.sum(torch.square(self.base_lin_vel), dim=1)
            + torch.sum(torch.square(self.base_ang_vel), dim=1)
        )
        return -total_penalty * is_standing

    def _reward_StandStillContactReward(self):
        """Reward having both feet on the ground when commanded to stand still."""
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_standing = cmd_mag < 0.1
        both_feet_contact = (self.last_contacts[:, 0] & self.last_contacts[:, 1]).float()
        return both_feet_contact * is_standing

    def _reward_KneeRegularizationReward(self):
        """Reward tracking the biomechanical knee flexion profile during walking.

        Two-tier sigma: loose (0.2) during flexion/stance for shape tracking,
        tight (0.04) during extension (phase >= 0.65) to drive precise
        pre-landing straightening at heel contact.
        """
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_moving = cmd_mag > 0.1
        # knee joints: left_knee_joint (3), right_knee_joint (9)
        current_knees = self.dof_pos[:, [3, 9]]
        error = torch.square(current_knees - self.desired_knee_angles)

        total_reward = torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)
        for i in range(2):
            phase = self.foot_phases[:, i]
            in_extension = (phase >= 0.65).float()
            sigma = 0.04 * in_extension + 0.2 * (1.0 - in_extension)
            total_reward += torch.exp(-error[:, i] / sigma)

        return total_reward * is_moving.float()
