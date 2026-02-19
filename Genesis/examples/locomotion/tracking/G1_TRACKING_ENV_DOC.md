# G1TrackingEnv — Motion Tracking Environment Documentation

`g1_tracking_env.py` | Config: `configs/g1_tracking_config.yaml`

---

## Summary

`G1TrackingEnv` trains the Unitree G1 12-DOF humanoid to replicate reference motions from a motion capture NPZ file. The implementation closely follows the **BeyondMimic** methodology (IsaacLab), adapted to Genesis. All body poses are tracked in an **anchor-relative frame** that removes global position and heading drift from the reward signal.

The environment uses an **asymmetric actor-critic** design: the policy (actor) receives a compact 53-dimensional observation suitable for sim-to-real transfer, while the critic receives a privileged 170-dimensional observation that includes full body-space state.

**Termination:** roll/pitch > 60°, base height < 0.20 m (allows jumps), or motion sequence end.

---

## Observation Spaces

### Policy Observation (Actor, 53 dims)

| Component | Dims | Description |
|-----------|------|-------------|
| Phase encoding | 2 | `[sin, cos]` of normalized motion time |
| Motion anchor pos (body frame) | 3 | Reference anchor XY+Z in robot anchor frame |
| Motion anchor ori (body frame) | 6 | Reference anchor heading, 2-column rotation matrix |
| Base linear velocity | 3 | Pelvis lin vel in body frame |
| Base angular velocity | 3 | Pelvis ang vel in body frame |
| Joint position delta | 12 | `dof_pos − default_dof_pos` |
| Joint velocity | 12 | `dof_vel` |
| Last actions | 12 | Previous policy output |

Uniform noise is added during training (`add_noise: true`).

### Privileged Observation (Critic, 170 dims)

Extends the policy obs with full body-space state (no noise):

| Component | Dims | Description |
|-----------|------|-------------|
| All policy dims | 53 | Same as above, no noise |
| Body positions (body frame) | 39 | 13 bodies × 3, in anchor-yaw frame |
| Body orientations (body frame) | 78 | 13 bodies × 6, 2-column rotation matrices |

### Anchor Frame

`anchor_pos = [pelvis_x, pelvis_y, 0]`, `anchor_quat = yaw_only(pelvis_quat)`. All body-relative positions are expressed as `R_yaw^T @ (body_pos_w − anchor_pos)`.

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_actions` | 12 | Controlled DOFs |
| `episode_length_s` | 15.0 s | Max episode (> motion duration ~12 s) |
| `action_scale` | 0.25 | Scales policy output to joint residuals |
| `dt` | 0.02 s | 50 Hz control |
| `base_init_pos` | [0, 0, 0.80] m | Initial pelvis height |
| `termination_roll/pitch` | 60° | Looser than G1Env to allow jump dynamics |
| `termination_height` | 0.20 m | Min pelvis height |
| `num_obs` | 53 | Policy observation dims |
| `num_privileged_obs` | 170 | Critic observation dims |
| `add_noise` | true | Uniform obs noise during training |

**PD Gains:** Identical to G1Env (hip 100/2, knee 150/4, ankle 40/2).

**Reference-State Initialization:** Each reset samples a random start frame from `[0, num_frames − 1 − 2×fps]`, guaranteeing ≥ 2 s of motion per episode. The robot is initialized with the reference joint positions, velocities, and base state.

**Observation scales:**

| Scale | Value |
|-------|-------|
| `lin_vel` | 0.2 |
| `ang_vel` | 0.25 |
| `dof_pos` | 1.0 |
| `dof_vel` | 0.05 |
| `anchor_pos` | 0.2 |
| `body_pos` | 0.5 |

---

## Reward Weights

All scales are multiplied by `dt = 0.02` at init. All tracking terms use `exp(-error / σ²)` kernels.

| Reward | Scale | σ | Description |
|--------|-------|---|-------------|
| `GlobalAnchorPositionTracking` | 3.0 | 0.5 m | Pelvis XY position match |
| `GlobalAnchorOrientationTracking` | 3.0 | 0.5 rad | Pelvis heading match |
| `RelativeBodyPositionTracking` | 10.0 | 0.15 m | Body shape (mean over 13 bodies) |
| `RelativeBodyOrientationTracking` | 5.0 | 0.5 rad | Body orientation shape |
| `GlobalBodyLinVelTracking` | 2.0 | 1.0 m/s | Body linear velocity match |
| `GlobalBodyAngVelTracking` | 2.0 | 2.0 rad/s | Body angular velocity match |
| `FeetContactTime` | 1.0 | — | Reward sustained contact ≥ 0.3 s (fires on lift-off) |
| `AlivBonus` | 0.5 | — | Constant per-step survival bonus |
| `FallPenalty` | 100.0 | — | −1.0 on fall step (effective −2.0/fall) |
| `ActionRatePenalty` | 0.01 | — | Penalize action jerk |
| `DofLimitPenalty` | 50.0 | — | Penalize joints near soft limits (90%) |

---

## Motion Data

- **File:** `tracking/configs/walk_jump_bothfeet_4_small_12dof_80fps.npz`
- **FPS:** 50 (despite the filename; `dt = 0.02` matches one frame exactly)
- **Duration:** ~12.1 s (607 frames)
- **Joint order:** Matches `g1_config.yaml` → identity permutation at runtime

---

## Training

Run from `Genesis/examples/locomotion/tracking/`:

```bash
python g1_tracking_train.py                             # default: 4096 envs
python g1_tracking_train.py -e g1-track-v2 -B 2048 --max_iterations 2000
python g1_tracking_train.py -e g1-track --resume --ckpt 500
python g1_tracking_eval.py  -e g1-track --ckpt 500
```

Logs and checkpoints: `logs/<exp_name>/model_<iter>.pt`

Saved pickle contains 5 items: `[env_cfg, obs_cfg, reward_cfg, motion_cfg, train_cfg]`.
