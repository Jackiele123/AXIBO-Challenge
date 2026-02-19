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

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.randomization_cfg = randomization_cfg or {}
        
        # Enable command resampling by default (can be disabled for evaluation)
        self.enable_command_resampling = env_cfg.get("enable_command_resampling", True)

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
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

        # add plain
        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        # add robot - using the G1 12DOF model
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="../g1_files/g1_12dof_package/g1_12dof/g1_12dof.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # PD control parameters - map joint-specific stiffness and damping
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

        # Define global gravity direction vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # Initial state
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

        # initialize buffers
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
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        # Feet tracking for rewards
        self.feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.feet_link_idx = torch.tensor(
            [self.robot.get_link(name).idx_local for name in self.feet_links],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.feet_pos = torch.zeros((self.num_envs, 2, 3), dtype=gs.tc_float, device=gs.device)
        self.feet_heights = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.feet_vel = torch.zeros((self.num_envs, 2, 3), dtype=gs.tc_float, device=gs.device)
        
        # Contact tracking
        self.contact_forces = torch.zeros((self.num_envs, 2, 3), dtype=gs.tc_float, device=gs.device)
        self.foot_contact_weighted = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        
        # Feet air time tracking
        self.feet_air_time = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.last_contacts = torch.zeros((self.num_envs, 2), dtype=gs.tc_bool, device=gs.device)
        self.feet_first_contact = torch.zeros((self.num_envs, 2), dtype=gs.tc_bool, device=gs.device)
        self.latched_air_time = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        
        # Foot phase tracking (0 to 1, continuous)
        self.foot_phases = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        # 0 = left, 1 = right
        # Phase description: 0->0.5 = stance, 0.5->1.0 = swing
        # Phase offset: left and right should be 0.5 apart for walking gait
        self.gait_period = 1.0  # seconds for a full gait cycle (adjustable based on speed)
        
        # Desired ankle angles for heel-strike motion (pitch)
        self.desired_ankle_angles = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.desired_knee_angles = torch.zeros((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.ankle_joint_indices = [4, 10]  # left_ankle_pitch_joint, right_ankle_pitch_joint

        # Domain randomization - observation noise
        self.add_noise = self.randomization_cfg.get("add_noise", True)
        self.noise_scale_vec = torch.zeros(self.num_obs, dtype=gs.tc_float, device=gs.device)
        
        # Define noise scales (tune these based on real sensor specs)
        noise_scales = self.randomization_cfg.get("noise_scales", {
            "dof_pos": 0.01,    # +/- 0.01 rad
            "dof_vel": 1.5,     # +/- 1.5 rad/s
            "ang_vel": 0.2,     # +/- 0.2 rad/s
            "gravity": 0.05,    # +/- 0.05
            "commands": 0.0,    # No noise on commands usually
        })
        
        # Construct the noise vector (indices must match obs_buf concatenation order)
        # 0:3 ang_vel, 3:6 gravity, 6:9 commands, 9:21 dof_pos, 21:33 dof_vel, 33:45 actions
        self.noise_scale_vec[0:3] = noise_scales["ang_vel"]
        self.noise_scale_vec[3:6] = noise_scales["gravity"]
        self.noise_scale_vec[6:9] = noise_scales["commands"]
        self.noise_scale_vec[9:21] = noise_scales["dof_pos"]
        self.noise_scale_vec[21:33] = noise_scales["dof_vel"]
        # No noise on actions (33:45)

        # Domain randomization - pushing
        self.push_robots = self.randomization_cfg.get("push_robots", False)
        self.push_interval_s = self.randomization_cfg.get("push_interval_s", 15)
        self.last_push_step = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)

        # Domain randomization - physics properties
        self.randomize_friction = self.randomization_cfg.get("randomize_friction", False)
        self.randomize_base_mass = self.randomization_cfg.get("randomize_base_mass", False)
        self.randomize_motor_strength = self.randomization_cfg.get("randomize_motor_strength", False)
        
        # Motor strength randomization scales
        self.kp_scale = torch.ones((self.num_envs, 1), dtype=gs.tc_float, device=gs.device)
        self.kd_scale = torch.ones((self.num_envs, 1), dtype=gs.tc_float, device=gs.device)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        # Apply domain randomization
        self._randomize_physics_props()

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        # set all commands zero, randomly 10% of the time, to encourage standing and better foot tracking rewards
        zero_mask = torch.rand((self.num_envs,), device=gs.device) < 0.1
        commands[zero_mask] = 0.0
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        # update buffers
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

        # Update feet states
        for i, link_name in enumerate(self.feet_links):
            link = self.robot.get_link(link_name)
            self.feet_pos[:, i, :] = link.get_pos()
            self.feet_vel[:, i, :] = link.get_vel()
        self.feet_heights[:, 0] = self.feet_pos[:, 0, 2]  # Left foot Z
        self.feet_heights[:, 1] = self.feet_pos[:, 1, 2]  # Right foot Z
        
        # Update contact forces and air time
        # Get all link contact forces at once (shape: [n_envs, n_links, 3] or [n_links, 3])
        all_contact_forces = self.robot.get_links_net_contact_force()
        
        for i, link_idx in enumerate(self.feet_link_idx):
            # 1. Get Contact Data
            # Extract contact force for this foot link
            if all_contact_forces.dim() == 2:  # Single env case: [n_links, 3]
                contact_force = all_contact_forces[link_idx:link_idx+1, :]
            else:  # Multi env case: [n_envs, n_links, 3]
                contact_force = all_contact_forces[:, link_idx, :]
            
            self.contact_forces[:, i, :] = contact_force
            contact_norm = torch.norm(contact_force, dim=-1)
            
            # 2. Determine Contact State (Use a robust threshold like 5.0N)
            in_contact = contact_norm > 5.0
            self.foot_contact_weighted[:, i] = torch.clamp(contact_norm / 100.0, 0.0, 1.0)

            # 3. Increment Air Time (Simulate time passing)
            self.feet_air_time[:, i] += self.dt

            # 4. Detect "First Contact" (Touchdown)
            # True if we are touching now, but weren't touching last step
            first_contact = in_contact & ~self.last_contacts[:, i]
            self.feet_first_contact[:, i] = first_contact

            # 5. Capture the Air Time (Latching)
            # If we just landed, copy the current timer to the buffer.
            # We do this BEFORE resetting the timer in step 6.
            self.latched_air_time[:, i] = torch.where(
                first_contact,
                self.feet_air_time[:, i],
                self.latched_air_time[:, i]
            )

            # 6. Reset Air Time
            # If we are on the ground, reset timer to 0
            self.feet_air_time[:, i] *= ~in_contact
            
            # 7. Update Last Contact for next step
            self.last_contacts[:, i] = in_contact
        
        # Update foot phases
        self._update_foot_phases()
        
        # Compute desired ankle angles for heel-strike motion
        self._compute_desired_ankle_angles()
        self._compute_desired_knee_angles()

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Apply random pushes
        if self.push_robots:
            self._push_robots()

        # resample commands (only if enabled)
        if self.enable_command_resampling:
            self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # check termination and reset - humanoids are more sensitive to orientation
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_base_height_less_than"]

        # Compute timeout
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # Reset environment if necessary
        self._reset_idx(self.reset_buf)

        # update observations
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        
        if self.add_noise:
            # Uniform noise [-1, 1] * scale
            noise = (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            return self.obs_buf + noise, self.extras
            
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def _reset_idx(self, envs_idx=None):
        # reset state
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # reset buffers
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_pos.copy_(self.init_base_pos)
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
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)
            
        # Reset phase tracking
        if envs_idx is None:
            # Start in flat-foot phase (0.1) so desired ankle = 0 rad, matching default_dof_pos.
            # Phase 0.0 = heel-strike (desired +0.15 rad) would cause an immediate penalty spike.
            # Right foot offset by 0.5 for walking gait antiphase: 0.1 + 0.5 = 0.6
            self.foot_phases[:, 0] = 0.1
            self.foot_phases[:, 1] = 0.6
            self.desired_ankle_angles.zero_()
            self.desired_knee_angles.zero_()
        else:
            self.foot_phases[envs_idx, 0] = 0.1
            self.foot_phases[envs_idx, 1] = 0.6
            self.desired_ankle_angles.masked_fill_(envs_idx[:, None], 0.0)
            self.desired_knee_angles.masked_fill_(envs_idx[:, None], 0.0)
        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            value.masked_fill_(envs_idx, 0.0)

        # random sample command upon reset
        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.obs_buf, None

    # ------------ domain randomization functions ----------------
    def _randomize_physics_props(self):
        """Randomize friction, mass, and motor gains. Called once during __init__."""
        if not (self.randomize_friction or self.randomize_base_mass or self.randomize_motor_strength):
            return

        # 1. Randomize Motor Gains (Kp/Kd)
        if self.randomize_motor_strength:
            stiffness_range = self.randomization_cfg.get("motor_strength_range", [0.8, 1.2])  # +/- 20%
            damping_range = self.randomization_cfg.get("motor_strength_range", [0.8, 1.2])
            
            self.kp_scale = torch.rand(self.num_envs, 1, dtype=gs.tc_float, device=gs.device) * \
                            (stiffness_range[1] - stiffness_range[0]) + stiffness_range[0]
            self.kd_scale = torch.rand(self.num_envs, 1, dtype=gs.tc_float, device=gs.device) * \
                            (damping_range[1] - damping_range[0]) + damping_range[0]

        # 2. Randomize Base Mass
        if self.randomize_base_mass:
            added_mass_range = self.randomization_cfg.get("added_mass_range", [-1.0, 3.0])  # +/- kg
            # Note: Genesis may not support per-env mass randomization easily
            # This is a placeholder for when the API supports it
            pass

        # 3. Randomize Friction
        if self.randomize_friction:
            friction_range = self.randomization_cfg.get("friction_range", [0.5, 1.25])
            # Note: Genesis may not support per-env friction randomization easily
            # This is a placeholder for when the API supports it
            pass

    def _update_foot_phases(self):
        """Update foot phase based on contacts and command velocity."""
        # Calculate command speed
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        
        # Adaptive gait period: faster when speed is higher
        # At 0 m/s: 1.5s period (slow)
        # At 1.5 m/s: 1s period (fast)
        speed_ratio = torch.clamp(cmd_mag / 1.5, 0.0, 1.0)
        adaptive_period = 1.5 - (speed_ratio * 0.50)
        
        # Phase increment per timestep
        phase_increment = self.dt / adaptive_period
        
        # Only update phase if we're commanding movement
        is_moving = cmd_mag > 0.1
        
        # Increment phase
        self.foot_phases += phase_increment.unsqueeze(1) * is_moving.unsqueeze(1)
        
        # Wrap phase to [0, 1)
        self.foot_phases = torch.fmod(self.foot_phases, 1.0)
        
        # On first contact (landing), snap each foot's phase to 0.0 (start of stance).
        # Both feet use the same target and circular distance to handle the 1.0->0.0 wrap.
        #
        # Why 0.0 for both feet?
        # Each foot's phase independently tracks its own gait cycle. Both feet contact
        # near their own phase 0.0 (after wrapping from ~1.0). The 0.5 antiphase offset
        # is maintained through initialization and natural phase advancement — it does
        # NOT need a different snap target per foot.
        #
        # The previous right_target_phase=0.5 caused a critical bug: on the first step
        # of every episode, both feet are in contact and first_contact fires for both.
        # Right foot at phase 0.6 had circular distance 0.1 from 0.5 (<0.3 threshold),
        # so it immediately snapped to 0.5 (swing zone >=0.4). This made every
        # phase-dependent reward treat the right foot as "should be swinging" even
        # during stance, causing persistent right-leg stiffness.
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
        """Compute desired ankle angles for heel-strike to toe-off motion.

        Biomechanically corrected (2026-02-18):
          heel_angle=+0.08 rad (+4.6 deg dorsiflexion at heel strike, human: 0-5 deg)
          pushoff_angle=-0.28 rad (-16 deg plantarflexion at toe-off, human: 15-20 deg)
          swing_angle=+0.07 rad (+4 deg dorsiflexion for foot clearance; was -0.15 — sign fixed)
          land_angle=+0.05 rad (+2.9 deg dorsiflexion held from phase 0.65 onward)
        Smooth ramps replace instantaneous steps at phase boundaries.
        """
        heel_angle    =  0.08   # dorsiflexion at heel strike
        pushoff_angle = -0.28   # plantarflexion at push-off
        swing_angle   =  0.07   # dorsiflexion during swing for foot clearance
        land_angle    =  0.05   # dorsiflexion for pre-landing

        for i in range(2):  # left, right
            phase = self.foot_phases[:, i]

            # Heel strike (0.00-0.05): brief dorsiflexion
            hs_mask = phase < 0.05

            # Heel-to-flat ramp (0.05-0.08): 3% smooth transition to neutral
            hs_flat_mask = (phase >= 0.05) & (phase < 0.08)
            hs_flat_angle = heel_angle * (1.0 - (phase - 0.05) / 0.03)

            # Flat foot (0.08-0.33): neutral stance
            flat_mask = (phase >= 0.08) & (phase < 0.33)

            # Push-off ramp (0.33-0.40): 7% ramp to full plantarflexion
            pushoff_mask = (phase >= 0.33) & (phase < 0.40)
            pushoff_ramp = pushoff_angle * ((phase - 0.33) / 0.07)

            # Early swing recovery (0.40-0.55): ramp plantarflexion -> dorsiflexion
            early_swing_mask = (phase >= 0.40) & (phase < 0.55)
            early_prog = (phase - 0.40) / 0.15
            early_swing_angle = pushoff_angle + early_prog * (swing_angle - pushoff_angle)

            # Mid-swing clearance (0.55-0.65): hold dorsiflexion for foot clearance
            mid_swing_mask = (phase >= 0.55) & (phase < 0.65)

            # Pre-landing (0.65-1.00): maintain dorsiflexion for heel-strike prep
            landing_mask = phase >= 0.65

            # Combine all phases
            desired_angle = torch.zeros_like(phase)
            desired_angle = torch.where(hs_mask, heel_angle, desired_angle)
            desired_angle = torch.where(hs_flat_mask, hs_flat_angle, desired_angle)
            desired_angle = torch.where(flat_mask, 0.0, desired_angle)
            desired_angle = torch.where(pushoff_mask, pushoff_ramp, desired_angle)
            desired_angle = torch.where(early_swing_mask, early_swing_angle, desired_angle)
            desired_angle = torch.where(mid_swing_mask, swing_angle, desired_angle)
            desired_angle = torch.where(landing_mask, land_angle, desired_angle)

            self.desired_ankle_angles[:, i] = desired_angle
        
    def _push_robots(self):
        """Apply random external forces to robot base periodically."""
        push_interval = int(self.push_interval_s / self.dt)
        
        # Check which envs are due for a push
        envs_to_push = (self.episode_length_buf - self.last_push_step) >= push_interval
        
        if envs_to_push.any():
            max_vel_change = self.randomization_cfg.get("max_push_vel_xy", 1.0)  # m/s
            
            # Generate random push direction and magnitude
            xy_perturbation = (torch.rand(self.num_envs, 2, dtype=gs.tc_float, device=gs.device) * 2 - 1)
            xy_perturbation[~envs_to_push] = 0.0  # Only push selected envs
            
            # Apply velocity perturbation
            curr_vel = self.robot.get_vel()
            curr_vel[:, :2] += xy_perturbation * max_vel_change
            self.robot.set_vel(curr_vel)
            
            # Update last push time
            self.last_push_step[envs_to_push] = self.episode_length_buf[envs_to_push]

    # ------------ reward functions----------------
    def _reward_LinVelXYReward(self):
        """Reward tracking linear velocity commands (XY)"""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg.get("tracking_sigma", 0.25))

    def _reward_AngVelZReward(self):
        """Reward tracking angular velocity command (Yaw)"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg.get("tracking_sigma", 0.25))
    
    def _reward_LinVelZPenalty(self):
        """Penalize Z-axis base linear velocity"""
        return -torch.square(self.base_lin_vel[:, 2])
    
    def _reward_AngVelXYPenalty(self):
        """Penalize angular velocity in roll and pitch"""
        return -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_OrientationPenalty(self):
        """Penalize non-upright orientation"""
        return -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_ActionRatePenalty(self):
        """Penalize changes in actions"""
        return -torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_ActionLimitPenalty(self):
        """Penalize actions near limits"""
        action_limit = self.reward_cfg.get("action_limit", 1.0)
        near_limit = (torch.abs(self.actions) > action_limit * 0.9).float()
        return -torch.sum(near_limit * torch.square(self.actions), dim=1)
    
    def _reward_HipYawPenalty(self):
        """Penalize hip yaw DOF positions (indices 2, 8)"""
        # left_hip_yaw_joint, right_hip_yaw_joint
        return -torch.sum(torch.abs(self.dof_pos[:, [2, 8]]), dim=-1)
    
    def _reward_HipRollPenalty(self):
        """Penalize hip roll DOF positions (indices 1, 7)"""
        # left_hip_roll_joint, right_hip_roll_joint
        return -torch.sum(torch.abs(self.dof_pos[:, [1, 7]]), dim=-1)
    
    def _reward_BodyRollPenalty(self):
        """Penalize body roll"""
        # Extract roll from euler angles
        return -torch.square(self.base_euler[:, 0] * math.pi / 180.0)  # Convert to radians for penalty
    
    def _reward_FeetAirTimePenalty(self):
        """Penalize feet air time deviation from target (only applied at moment of contact)"""
        target_air_time = self.reward_cfg.get("target_feet_air_time", 0.6)
        
        # Compute penalty: absolute deviation from target, masked by first contact
        pen_air_time = torch.sum(
            torch.abs(self.latched_air_time - target_air_time) * self.feet_first_contact.float(), dim=1
        )
        return -pen_air_time
    # def _reward_FeetAirTimePenalty(self):
    #     """
    #     Adaptive Air Time Penalty (Inverse scaling).
    #     1. Only applies when moving (cmd > 0.1).
    #     2. Target scales INVERSELY with speed: 
    #        - 0.6s at 0.0 m/s (Slow swings)
    #        - 0.2s at 1.5 m/s (Fast, rapid stepping)
    #     """
    #     # Calculate command speed (XY magnitude)
    #     cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        
    #     # 1. Mask: Only penalize if we WANT to walk (cmd > 0.1 m/s)
    #     is_moving_cmd = cmd_mag > 0.1
        
    #     # 2. Adaptive Target Calculation
    #     # Clamp speed to 1.5 max
    #     speed_ratio = torch.clamp(cmd_mag / 1.5, 0.0, 1.0)
        
    #     # Inverse Interpolation: Start at 0.8, subtract up to 0.6 to reach 0.2
    #     # Ratio 0.0 -> 1.0 - 0.0 = 1.0s
    #     # Ratio 1.0 -> 1.0 - 0.8 = 0.2s
    #     adaptive_target = 0.6 - (speed_ratio * 0.4)
        
    #     # Expand target to shape [num_envs, 2]
    #     target_air_time = adaptive_target.unsqueeze(1).repeat(1, 2)
        
    #     # 3. Calculate Reward
    #     air_time_error = torch.abs(self.latched_air_time - target_air_time)
        
    #     # Mask by first_contact AND is_moving_cmd
    #     valid_mask = self.feet_first_contact & is_moving_cmd.unsqueeze(1)
        
    #     pen_air_time = torch.sum(air_time_error * valid_mask.float(), dim=1)
    #     return -pen_air_time
    
    def _reward_G1FeetSlidePenalty(self):
        """Penalize feet sliding when in contact"""
        # Check if feet are on ground
        on_ground = self.foot_contact_weighted > 0.1
        # Get XY velocity
        feet_vel_xy = torch.norm(self.feet_vel[:, :, :2], dim=-1)
        # Penalize sliding when on ground
        return -torch.sum(on_ground * torch.square(feet_vel_xy), dim=1)
    
    def _reward_FeetOrientationPenalty(self):
        """Penalize feet orientation deviation from flat"""
        left_foot_quat = self.robot.get_link(self.feet_links[0]).get_quat()
        right_foot_quat = self.robot.get_link(self.feet_links[1]).get_quat()
        
        left_gravity_local = transform_by_quat(self.global_gravity, inv_quat(left_foot_quat))
        right_gravity_local = transform_by_quat(self.global_gravity, inv_quat(right_foot_quat))
        
        # XY components should be zero if flat
        left_dev = -torch.sum(torch.square(left_gravity_local[:, :2]), dim=1)
        right_dev = -torch.sum(torch.square(right_gravity_local[:, :2]), dim=1)
        
        return left_dev + right_dev
    
    def _reward_DofPosLimitPenalty(self):
        """Penalize DOF positions near limits"""
        # Get limits for all DOFs and index to get only the controlled joints
        all_lower_limits = self.robot.get_dofs_limit()[0]
        all_upper_limits = self.robot.get_dofs_limit()[1]
        
        # Extract limits for controlled joints only
        lower_limits = all_lower_limits[self.motors_dof_idx]
        upper_limits = all_upper_limits[self.motors_dof_idx]
        
        # Soft limit buffer (e.g., 90% of range)
        # Penalize if we get within 10% of the hard limit
        soft_ratio = 0.9
        
        # Check lower violation
        # If limit is -1.0, soft is -0.9. If pos is -0.95, error is -0.05
        out_of_limits = -(self.dof_pos - lower_limits * soft_ratio).clip(max=0.0)
        
        # Check upper violation
        out_of_limits += (self.dof_pos - upper_limits * soft_ratio).clip(min=0.0)
        
        return -torch.sum(out_of_limits, dim=1)
    
    # def _reward_FeetContactForceLimitPenalty(self):
    #     """Penalize feet contact forces above limit"""
    #     force_limit = self.reward_cfg.get("contact_force_limit", 1.0)
    #     above_limit = torch.clamp(self.foot_contact_weighted - force_limit, min=0.0)
    #     return torch.sum(torch.square(above_limit), dim=1)
    
    def _reward_FootPhaseReward(self):
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_moving = cmd_mag > 0.1
        
        # Calculate stance phase threshold based on speed (60% stance, 40% swing)
        stance_threshold = 0.4
        
        # Determine expected contact state
        left_should_contact = self.foot_phases[:, 0] < stance_threshold
        right_should_contact = self.foot_phases[:, 1] < stance_threshold
        
        # Get actual contact state
        left_in_contact = self.last_contacts[:, 0]
        right_in_contact = self.last_contacts[:, 1]
        
        # Reward: match expected contact state
        left_correct = (left_should_contact == left_in_contact).float()
        right_correct = (right_should_contact == right_in_contact).float()
        
        # IMPORTANT: Penalize having both feet on ground when at least one should be in swing
        both_should_not_contact = ~left_should_contact | ~right_should_contact  # At least one in swing
        both_actually_contact = left_in_contact & right_in_contact  # Both on ground
        double_stance_penalty = (both_should_not_contact & both_actually_contact).float()
        
        # Combined reward: high when contacts match phase AND no double stance during swing
        phase_alignment = (left_correct + right_correct) / 2.0 - double_stance_penalty
        
        # Only apply when moving
        return phase_alignment * is_moving
    
    def _reward_StandingKneeReward(self):
        """Reward straight knees when standing still (zero command)."""
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_standing = cmd_mag < 0.1  # Standing still threshold
        
        # Knee joint indices: left_knee_joint (3), right_knee_joint (9)
        knee_angles = self.dof_pos[:, [3, 9]]
        
        # Desired knee angle when standing: match default_dof_pos (0.3 rad)
        # Avoids fighting the PD controller which targets default pose
        desired_standing_angle = 0.3
        knee_error = torch.sum(torch.square(knee_angles - desired_standing_angle), dim=1)
        
        # Exponential reward: high when knees are straight
        knee_reward = torch.exp(-knee_error / 0.1)
        
        # Only apply when standing
        return knee_reward * is_standing
    
    def _reward_AnkleTrackingPenalty(self):
        """Penalize deviation from desired ankle angles for natural heel-strike motion."""
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_moving = cmd_mag > 0.1
        
        # Get current ankle angles (pitch)
        ankle_angles = self.dof_pos[:, self.ankle_joint_indices]

        # Compute per-ankle squared error (shape: num_envs x 2)
        per_ankle_error = torch.square(ankle_angles - self.desired_ankle_angles)

        # Apply exp kernel per-ankle then sum:
        # error = 0 (perfect) -> exp(0) - 1 = 0 per ankle
        # error = large       -> exp(-large) - 1 -> -1 per ankle (max -2 total)
        # sigma = 0.1 rad^2 (~0.32 rad) gives a wider gradient region than 0.05
        per_ankle_penalty = torch.exp(-per_ankle_error / 0.1) - 1.0
        penalty = torch.sum(per_ankle_penalty, dim=1)

        # Only penalize when moving
        return penalty * is_moving
    
    def _reward_StandStillVelocityPenalty(self):
        """Penalize any joint or base movement when commanded to stand still."""
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_standing = cmd_mag < 0.1  # Standing still threshold
        
        # Penalize joint velocities
        joint_vel_penalty = torch.sum(torch.square(self.dof_vel), dim=1)
        
        # Penalize base linear velocity (XY plane mainly, but include Z)
        base_lin_vel_penalty = torch.sum(torch.square(self.base_lin_vel), dim=1)
        
        # Penalize base angular velocity
        base_ang_vel_penalty = torch.sum(torch.square(self.base_ang_vel), dim=1)
        
        # Total penalty
        total_penalty = joint_vel_penalty + base_lin_vel_penalty + base_ang_vel_penalty
        
        # Only apply when standing
        return -total_penalty * is_standing
    
    def _reward_StandStillContactReward(self):
        """Reward having both feet on the ground when commanded to stand still."""
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        is_standing = cmd_mag < 0.1  # Standing still threshold
        
        # Check if both feet are in contact
        both_feet_contact = (self.last_contacts[:, 0] & self.last_contacts[:, 1]).float()
        
        # Only reward when standing
        return both_feet_contact * is_standing

    def _compute_desired_knee_angles(self):
        """
        Compute desired knee flexion based on human gait biomechanics.
        G1 Knee Joint (Revolute): 0.0 = Straight, +Pos = Bent.

        Human Gait Cycle Approximation:
        - Phase 0.0-0.4 (Stance): sin bump, peak ~0.35 rad (20 deg) at phase 0.20
        - Phase 0.4-0.7 (Swing Flexion): sin-based rise to ~1.1 rad (63 deg) at phase 0.70
        - Phase 0.7-1.0 (Swing Extension): power-law drop (1-t)^1.5, faster than flexion
          -> phase 0.85: ~0.45 rad (26 deg) vs 0.81 rad (46 deg) with symmetric sin

        Boundary aligned with FootPhaseReward and _compute_desired_ankle_angles: 0.4
        Asymmetric swing: sin flexion + (1-t)^1.5 extension for faster knee drop.
        """
        self.desired_knee_angles.zero_()

        for i in range(2):  # Left and Right
            phase = self.foot_phases[:, i]

            # 1. Stance Phase (0.0 to 0.4): slight bend then extend for support
            stance_mask = phase < 0.4
            stance_progress = (phase / 0.4) * math.pi
            stance_angle = 0.1 + 0.25 * torch.sin(stance_progress)

            # 2. Swing Flexion (0.4 to 0.7): rapid knee lift via sin(0 -> pi/2)
            swing_flex_mask = (phase >= 0.4) & (phase < 0.7)
            flex_progress = (phase - 0.4) / 0.3  # 0 to 1
            flex_angle = 0.1 + 1.0 * torch.sin(flex_progress * math.pi / 2)
            # phase 0.40: 0.10 rad; phase 0.70: 1.10 rad

            # 3. Swing Extension (0.7 to 1.0): faster power-law drop
            ext_progress = (phase - 0.7) / 0.3  # 0 to 1
            ext_angle = 0.1 + 1.0 * (1.0 - ext_progress) ** 1.5
            # phase 0.70: 1.10 rad; phase 0.85: ~0.45 rad; phase 1.0: 0.10 rad

            # Combine: stance | swing_flex | swing_ext
            self.desired_knee_angles[:, i] = torch.where(
                stance_mask, stance_angle,
                torch.where(swing_flex_mask, flex_angle, ext_angle)
            )
    def _reward_KneeRegularizationReward(self):
            """
            Reward tracking natural human-like knee flexion profile.
            Only applies when moving.
            """
            cmd_mag = torch.norm(self.commands[:, :2], dim=1)
            is_moving = cmd_mag > 0.1
            
            # Knee indices: 3 (Left), 9 (Right)
            current_knees = self.dof_pos[:, [3, 9]]
            
            # Calculate error (squared difference)
            error = torch.square(current_knees - self.desired_knee_angles)
            
            # Convert to reward (Gaussian kernel)
            # Sigma = 0.2 (tolerant, we want the "shape" not perfect tracking)
            reward = torch.exp(-error / 0.2)
            
            # Sum both knees and mask
            return torch.sum(reward, dim=1) * is_moving.float()