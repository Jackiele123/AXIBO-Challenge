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
    # def _reward_FeetAirTimePenalty(self):
    #     """Penalize feet air time deviation from target (only applied at moment of contact)"""
    #     target_air_time = self.reward_cfg.get("target_feet_air_time", 0.4)
        
    #     # Compute penalty: absolute deviation from target, masked by first contact
    #     pen_air_time = torch.sum(
    #         torch.abs(self.latched_air_time - target_air_time) * self.feet_first_contact.float(), dim=1
    #     )
    #     return -pen_air_time
    def _reward_FeetAirTimePenalty(self):
        """
        Adaptive Air Time Penalty (Inverse scaling).
        1. Only applies when moving (cmd > 0.1).
        2. Target scales INVERSELY with speed: 
           - 0.6s at 0.0 m/s (Slow swings)
           - 0.2s at 1.5 m/s (Fast, rapid stepping)
        """
        # Calculate command speed (XY magnitude)
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        
        # 1. Mask: Only penalize if we WANT to walk (cmd > 0.1 m/s)
        is_moving_cmd = cmd_mag > 0
        
        # 2. Adaptive Target Calculation
        # Clamp speed to 1.5 max
        speed_ratio = torch.clamp(cmd_mag / 1.5, 0.0, 1.0)
        
        # Inverse Interpolation: Start at 0.8, subtract up to 0.6 to reach 0.2
        # Ratio 0.0 -> 1.0 - 0.0 = 1.0s
        # Ratio 1.0 -> 1.0 - 0.8 = 0.2s
        adaptive_target = 0.6 - (speed_ratio * 0.4)
        
        # Expand target to shape [num_envs, 2]
        target_air_time = adaptive_target.unsqueeze(1).repeat(1, 2)
        
        # 3. Calculate Reward
        air_time_error = torch.abs(self.latched_air_time - target_air_time)
        
        # Mask by first_contact AND is_moving_cmd
        valid_mask = self.feet_first_contact & is_moving_cmd.unsqueeze(1)
        
        pen_air_time = torch.sum(air_time_error * valid_mask.float(), dim=1)
        return -pen_air_time
    
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
    
    def _reward_StraightKneeStandReward(self):
        """
        Encourages straight knees (0.0 rad) ONLY when standing still.
        Prevents the robot from crouching while waiting for commands.
        """
        # Define "Standing Still" command threshold
        is_standing = torch.norm(self.commands[:, :2], dim=1) < 0.1
        
        # Get Knee Indices (G1 12-DOF standard order: HipP, HipR, HipY, Knee...)
        # Indices: Left Knee = 3, Right Knee = 9
        knee_indices = [3, 9]
        knee_pos = self.dof_pos[:, knee_indices]
        
        # Calculate deviation from 0.0 (Straight)
        # We assume 0.0 is straight in the URDF. If straight is 0.3, change target here.
        knee_error = torch.sum(torch.square(knee_pos), dim=1)
        
        # Only apply penalty when standing
        return -knee_error * is_standing.float()
    
    def _reward_HeelStrikePenalty(self):
        """
        Encourages Heel-First contact.
        At the moment of impact (first_contact), the foot pitch should be positive (toes up).
        """
        # 1. Get Foot Orientation
        left_foot_quat = self.robot.get_link(self.feet_links[0]).get_quat()
        right_foot_quat = self.robot.get_link(self.feet_links[1]).get_quat()
        
        # Convert to Euler [Roll, Pitch, Yaw]
        left_rpy = quat_to_xyz(left_foot_quat, rpy=True)
        right_rpy = quat_to_xyz(right_foot_quat, rpy=True)
        
        # Stack pitches: Shape [num_envs, 2]
        foot_pitch = torch.stack([left_rpy[:, 1], right_rpy[:, 1]], dim=1)
    
        target_heel_pitch = 0.4 
        # 3. Calculate Error
        # We only care if pitch is LESS than target (flat or toes down). 
        # If pitch is > 0.4 (super toes up), that's fine/exaggerated but acceptable.
        pitch_error = torch.square(torch.clamp(target_heel_pitch - foot_pitch, min=0.0))
        
        # 4. Apply only on First Contact
        return -torch.sum(pitch_error * self.feet_first_contact.float(), dim=1)
    # def _reward_FeetContactForceLimitPenalty(self):
    #     """Penalize feet contact forces above limit"""
    #     force_limit = self.reward_cfg.get("contact_force_limit", 1.0)
    #     above_limit = torch.clamp(self.foot_contact_weighted - force_limit, min=0.0)
    #     return torch.sum(torch.square(above_limit), dim=1)
