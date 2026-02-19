# G1Env — Walking Environment Documentation

`g1_env.py` | Config: `configs/g1_config.yaml`

---

## Summary

`G1Env` trains the Unitree G1 12-DOF humanoid to walk using PPO. The environment implements velocity command tracking combined with a suite of biomechanically-motivated rewards to produce natural heel-toe gait. Control runs at 50 Hz via PD position control with 1-step simulated action latency.

**Observation (45 dims):**
`ang_vel(3) | projected_gravity(3) | commands(3) | dof_pos_delta(12) | dof_vel(12) | actions(12)`

**Action space:** 12 joint position residuals, scaled by 0.25 and added to `default_dof_pos`.

**Termination:** roll > 20°, pitch > 20°, base height < 0.5 m, or episode timeout.

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_actions` | 12 | Controlled DOFs |
| `episode_length_s` | 20.0 s | Max episode duration |
| `resampling_time_s` | 4.0 s | Command resample interval |
| `action_scale` | 0.25 | Scales policy output to joint residuals |
| `clip_actions` | 100.0 | Action clipping bounds |
| `dt` | 0.02 s | Control timestep (50 Hz) |
| `substeps` | 2 | Physics substeps per control step |
| `base_init_pos` | [0, 0, 0.80] m | Initial pelvis height |
| `termination_roll` | 20° | Roll limit before reset |
| `termination_pitch` | 20° | Pitch limit before reset |
| `termination_height` | 0.5 m | Min pelvis height before reset |
| `zero_command_prob` | 0.1 | Probability of zero command per resample |
| `lin_vel_x_range` | [−1.0, 1.5] m/s | Forward/back command range |

**PD Gains:**

| Joint group | Kp | Kd |
|-------------|----|----|
| hip_pitch | 100 | 2 |
| hip_roll | 100 | 2 |
| hip_yaw | 100 | 2 |
| knee | 150 | 4 |
| ankle | 40 | 2 |

**Default joint angles:**

| Joint | Default (rad) |
|-------|--------------|
| hip_pitch | −0.1 |
| hip_roll | 0.0 |
| hip_yaw | 0.0 |
| knee | 0.3 |
| ankle_pitch | 0.0 |
| ankle_roll | 0.0 |

---

## Gait Phase System

Each foot has an independent phase variable (0 → 1). Phase 0–0.4 = stance, 0.4–1.0 = swing. The phase advances at a rate driven by an **adaptive gait period** that scales from 1.5 s (stationary) to 1.0 s (1.5 m/s). The left and right feet are initialized 0.5 apart (antiphase) and maintain this offset via natural advancement.

On first contact (touchdown), the phase is snapped to 0.0 if it is within a circular distance of 0.3 from that target, correcting any small drift.

### Desired Ankle Trajectory

| Phase | State | Target |
|-------|-------|--------|
| 0.00–0.05 | Heel strike | +0.08 rad |
| 0.05–0.08 | Ramp to flat | +0.08 → 0 |
| 0.08–0.33 | Flat stance | 0 rad |
| 0.33–0.40 | Push-off | 0 → −0.28 rad |
| 0.40–0.55 | Swing recovery | −0.28 → +0.07 rad |
| 0.55–0.65 | Mid-swing clearance | +0.07 rad |
| 0.65–1.00 | Pre-landing | +0.05 rad |

### Desired Knee Trajectory

| Phase | State | Profile |
|-------|-------|---------|
| 0.0–0.4 | Stance | sin bump, peak 0.35 rad at phase 0.2 |
| 0.4–0.7 | Swing flexion | sin rise, 0.1 → 1.1 rad |
| 0.7–1.0 | Swing extension | `(1−t)^1.5` drop, 1.1 → 0.1 rad |

---

## Reward Weights

All scales are multiplied by `dt = 0.02` at init.

| Reward | Scale | Description |
|--------|-------|-------------|
| `LinVelXYReward` | 10.0 | Track XY velocity command (Gaussian, σ=0.25) |
| `AngVelZReward` | 10.0 | Track yaw command (Gaussian, σ=0.25) |
| `LinVelZPenalty` | 20.0 | Penalize vertical base velocity |
| `AngVelXYPenalty` | 0.5 | Penalize roll/pitch angular velocity |
| `OrientationPenalty` | 100.0 | Penalize non-upright projected gravity |
| `ActionRatePenalty` | 0.01 | Penalize action changes (jerk) |
| `ActionLimitPenalty` | 0.1 | Penalize actions near saturation (limit=2.0) |
| `HipYawPenalty` | 5.0 | Penalize hip yaw deviation from zero |
| `HipRollPenalty` | 5.0 | Penalize hip roll deviation from zero |
| `BodyRollPenalty` | 100.0 | Penalize body roll angle |
| `FeetAirTimePenalty` | 100.0 | Penalize air time deviation from 0.6 s target |
| `G1FeetSlidePenalty` | 2.0 | Penalize foot sliding while in contact |
| `FeetOrientationPenalty` | 0.0 | (Disabled) Foot flatness penalty |
| `DofPosLimitPenalty` | 100.0 | Penalize joints near soft limits (90% of range) |
| `FootPhaseReward` | 6.0 | Reward contact/swing matching gait phase |
| `StandingKneeReward` | 2.0 | Reward default knee angle when standing |
| `AnkleTrackingPenalty` | 6.0 | Penalize ankle angle deviation from trajectory |
| `StandStillVelocityPenalty` | 0.05 | Penalize motion when commanded to stand |
| `StandStillContactReward` | 0.5 | Reward both feet down when standing |
| `KneeRegularizationReward` | 5.0 | Reward knee tracking biomechanical profile |
| `FootExtensionReward` | 3.0 | Reward forward foot reach during swing |

---

## Training

Run from `Genesis/examples/locomotion/`:

```bash
python g1_train.py                              # default: 4096 envs, 5000 iters
python g1_train.py -e g1-v2 -B 2048 --max_iterations 1000
python g1_train.py -e g1-walking --resume --ckpt 500
python g1_eval.py  -e g1-walking --ckpt 499
```

Logs and checkpoints: `logs/<exp_name>/model_<iter>.pt`
