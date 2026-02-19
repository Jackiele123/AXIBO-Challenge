# AXIBO Challenge — Technical Report
## G1 Humanoid Locomotion: Walking and Motion Tracking with Genesis RL

---

## Overview

This report describes the implementation of two reinforcement learning environments for training the Unitree G1 12-DOF humanoid robot to walk and track reference motions. Both environments are built on the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics simulator using PPO from `rsl-rl-lib`. All training runs at 50 Hz (dt = 0.02 s) with PD position control.

---

## Task 2 — Biomechanically-Motivated Walking (`G1Env`)

### 2.1 Phase-Tracked Foot Angle Rewards (Heel-Toe Dynamics)

The first challenge in producing natural locomotion was that a naive velocity-tracking reward produces a shuffling gait with no ground clearance and no heel-strike geometry. To address this, a continuous **gait phase** variable (0 → 1) was introduced for each foot, advancing at a rate inversely proportional to the gait period. The period itself adapts with command speed: 1.5 s at rest, 1.0 s at 1.5 m/s, giving the robot an implicit clock signal that drives all subsequent biomechanical rewards.

The phase is used to drive **desired ankle angles** through a seven-segment trajectory that mirrors the human gait cycle:

| Phase range | Ankle state | Target angle |
|-------------|-------------|-------------|
| 0.00–0.05 | Heel strike | +0.08 rad (dorsiflexion) |
| 0.05–0.08 | Heel-to-flat ramp | +0.08 → 0 rad |
| 0.08–0.33 | Flat-foot stance | 0 rad |
| 0.33–0.40 | Push-off ramp | 0 → −0.28 rad (plantarflexion) |
| 0.40–0.55 | Early swing recovery | −0.28 → +0.07 rad |
| 0.55–0.65 | Mid-swing clearance | +0.07 rad |
| 0.65–1.00 | Pre-landing | +0.05 rad |

An `AnkleTrackingPenalty` (Gaussian kernel, σ = 0.1 rad², scale = 6.0) penalizes deviation from these targets during walking. This directly shapes the ankle pitch joint and encourages the heel-strike to toe-off sequence.

A `FootPhaseReward` (scale = 6.0) simultaneously rewards the robot for matching expected contact states to the phase: each foot in stance when phase < 0.4, in swing when phase ≥ 0.4. A double-stance penalty fires when at least one foot should be in swing but both remain on the ground, which prevents the lazy solution of never lifting a foot.

### 2.2 Natural Knee Flexion

With ankle angles driven by the phase, the knee was addressed next. Human gait involves a characteristic **M-shaped knee angle profile**: a small bump during stance loading, a rapid rise to ~63° (1.1 rad) at peak swing flexion around 70% of the cycle, then an asymmetric power-law drop back to near-straight before landing.

`_compute_desired_knee_angles` implements this three-segment profile:

- **Stance (phase 0–0.4):** sinusoidal bump, peak 0.35 rad at phase 0.2 (loading response)
- **Swing flexion (phase 0.4–0.7):** sin-based rise from 0.1 → 1.1 rad
- **Swing extension (phase 0.7–1.0):** power-law `(1 − t)^1.5` drop back to 0.1 rad — faster than the symmetric sin, matching the rapid pre-landing knee extension in human gait

A `KneeRegularizationReward` (Gaussian kernel, σ = 0.2 rad², scale = 5.0) rewards the robot for tracking this profile during walking, and a separate `StandingKneeReward` (scale = 2.0) encourages the default 0.3 rad knee angle when standing still, which prevents the controller from fighting the PD controller at rest.

The stance-swing boundary is 0.4 across all three reward functions, ensuring consistency.

### 2.3 Reward Tuning — Reducing Joint Stiffness

Despite the ankle and knee rewards being biomechanically correct, initial training produced a stiff, robot-like gait where all joints moved minimally. The issue was traced to two overly conservative penalty parameters:

- **`ActionRatePenalty` (scale = 0.01):** The original default of 0.1 was penalizing action changes heavily enough that the policy converged on a strategy of near-zero actions (minimal deviation from default pose). The scale was reduced by 10× to 0.01, allowing the policy to make larger joint movements without excessive penalty.
- **`ActionLimitPenalty` (scale = 0.1, limit = 2.0):** The original `action_limit = 1.0` triggered penalties at 90% of that threshold (0.9), which fell inside the range the ankle and hip pitch joints needed for natural motion. Raising this threshold allowed freer joint range usage.

These two changes unlocked the ankle dorsiflexion and plantarflexion range needed for heel-toe motion to actually emerge during training.

### 2.4 Forward Stride Length — Hip Extension Reward

Even with correct ankle and knee rewards, the robot exhibited **heel stomping**: the swing foot returned approximately underfoot before landing, producing short choppy steps rather than a natural stride. The ankle trajectory and knee flexion profile were correct, but there was no signal driving the **forward reach** of the leg during swing.

`FootExtensionReward` was added to address this directly. During swing (phase > 0.4), each foot is rewarded for being positioned a target distance **in front of the base** in the robot's body frame:

```
target_fwd = 0.1 + 0.15 × clamp(cmd_x, 0, 2)  [metres]
reward = exp(-(foot_fwd - target_fwd)² / 0.04)
```

At 0 m/s (but moving): 0.1 m target; at 1 m/s: 0.25 m; at 2 m/s: 0.4 m. The Gaussian kernel with σ = 0.04 gives ~50% reward at 0.2 m error, which is forgiving enough for early training but tight enough to shape hip-pitch extension behavior. Scale = 3.0.

This reward works by requiring the hip pitch joint to extend the leg forward before touchdown, which is the primary biomechanical driver of stride length.

---

## Task 3 — BeyondMimic Motion Tracking (`G1TrackingEnv`)

### 3.1 Research Background

The following prior work was surveyed to inform the design:

- **[Unitree RL MJLab](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/main):** Reference implementation of locomotion rewards for Unitree robots in MuJoCo; used to cross-check joint indexing conventions and PD gain values.
- **[Whole Body Tracking (HybridRobotics)](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py#L110):** IsaacLab-based whole-body tracking setup; informed the asymmetric actor-critic observation design and the use of anchor-relative body positions.
- **[ASAP (LeCAR Lab)](https://github.com/LeCAR-Lab/ASAP):** Sim-to-real pipeline for agile humanoid motions; provided insight on reference-state initialization (RSI) and motion library design.

The core tracking methodology closely follows **BeyondMimic** (Unitree/IsaacLab), adapted from IsaacLab to Genesis.

### 3.2 Architecture

`G1TrackingEnv` is a clean-slate implementation (no inheritance from `G1Env`) that tracks reference body poses from a pre-recorded motion capture NPZ file. The robot is initialized at a random frame of the motion at each episode reset and must track the subsequent trajectory.

**Motion Library (`MotionLib`)** loads the NPZ file and provides:
- `get_frame(motion_time)`: interpolated body poses, velocities, and anchor frame at arbitrary time
- `get_init_state(start_frames)`: joint positions, velocities, and base state for reference-state initialization

**Anchor Frame:** To remove global position and heading drift from the reward signal, all body positions and orientations are expressed relative to an **anchor frame** defined as `(pelvis_xy, 0, yaw_only_quat)`. This means the reward measures pose shape and heading independently of where in the world the robot currently stands.

**Asymmetric Actor-Critic:** The policy (actor) receives a compact 53-dimensional observation sufficient for sim-to-real transfer. The critic receives a privileged 170-dimensional observation that includes full body-space state, enabling the value function to learn a much richer representation during training:

| Stream | Dims | Contents |
|--------|------|----------|
| Policy (actor) | 53 | phase(2) + anchor_pos_b(3) + anchor_ori_b(6) + lin_vel(3) + ang_vel(3) + dof_pos(12) + dof_vel(12) + actions(12) |
| Privileged (critic) | 170 | policy(53) + body_pos_b(39) + body_ori_b(78) − phase(2) reused |

**Reference-State Initialization (RSI):** At each reset, a random start frame is sampled from [0, `num_frames − 1 − 2×fps`], guaranteeing at least 2 seconds of motion per episode. The robot is initialized with exact joint positions and velocities from the reference, providing a warm start that avoids wasting the first seconds of each episode recovering from the default pose.

### 3.3 Reward Design

All reward terms use Gaussian exponential kernels: `exp(-error / σ²)`. Scales are listed as configured (multiplied by dt = 0.02 at runtime):

| Reward | Description | σ | Scale |
|--------|-------------|---|-------|
| `GlobalAnchorPositionTracking` | Pelvis XY position error | 0.5 m | 3.0 |
| `GlobalAnchorOrientationTracking` | Pelvis heading error | 0.5 rad | 3.0 |
| `RelativeBodyPositionTracking` | Mean body shape error (13 bodies) | 0.15 m | 10.0 |
| `RelativeBodyOrientationTracking` | Mean body orientation error | 0.5 rad | 5.0 |
| `GlobalBodyLinVelTracking` | Mean body linear velocity | 1.0 m/s | 2.0 |
| `GlobalBodyAngVelTracking` | Mean body angular velocity | 2.0 rad/s | 2.0 |
| `FeetContactTime` | Reward sustained foot contact ≥ 0.3 s | — | 1.0 |
| `AlivBonus` | Constant per-step bonus | — | 0.5 |
| `FallPenalty` | −1 on fall step (effective: −2.0/fall) | — | 100.0 |
| `ActionRatePenalty` | Action jerk penalty | — | 0.01 |
| `DofLimitPenalty` | Joint limit soft-boundary penalty | — | 50.0 |

The `body_pos_std = 0.15` (tighter than the 0.5 default) ensures the body shape reward provides a strong gradient, while `body_lin_vel_std = 1.0` and `body_ang_vel_std = 2.0` are widened to accommodate jump dynamics and tolerate early-training velocity errors.
---
Find more about each environment in their respective code files:
- `G1Env`: `Genesis/examples/locomotion/g1_env.py`
- `G1TrackingEnv`: `Genesis/examples/locomotion/tracking/g1_tracking_env.py`


## References

1. Genesis Physics Simulator — https://github.com/Genesis-Embodied-AI/Genesis
2. Unitree RL MJLab — https://github.com/unitreerobotics/unitree_rl_mjlab
3. HybridRobotics Whole Body Tracking — https://github.com/HybridRobotics/whole_body_tracking
4. LeCAR-Lab ASAP — https://github.com/LeCAR-Lab/ASAP
5. BeyondMimic (IsaacLab) — motion tracking reward design and asymmetric AC observations
