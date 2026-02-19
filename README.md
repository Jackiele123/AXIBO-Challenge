# AXIBO Challenge — G1 Humanoid RL

Reinforcement learning for a **Unitree G1 humanoid robot** (12-DOF) using the
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics simulator and
`rsl-rl-lib` PPO.

Two trained policies are included:

| Policy | Experiment folder | Description |
|--------|------------------|-------------|
| Motion tracking | `logs/g1-tracking-v1` | BeyondMimic-style tracking of a walk+jump reference motion |
| Walking (velocity-command) | *(train separately)* | Velocity-commanded locomotion with biomechanical gait rewards |

---

## Requirements

```bash
pip install rsl-rl-lib==2.2.4   # must be this exact package, not rsl-rl
```

All eval commands below must be run from the locomotion working directory:

```bash
cd Genesis/examples/locomotion
```

---

## Evaluate the motion-tracking policy

Runs the included `g1-tracking-v1` checkpoint. The reference motion plays
automatically and loops; tracking RMSE is printed each step.

```bash
# from Genesis/examples/locomotion/
python tracking/g1_tracking_eval.py -e g1-tracking-v1 --ckpt 49999
```

The viewer opens automatically. Press **ESC** to quit.

---

## Evaluate the walking policy

Requires a trained checkpoint in `logs/<exp_name>/`.

```bash
# from Genesis/examples/locomotion/
python g1_eval.py -e g1-walking --ckpt <iteration>
```

Replace `<iteration>` with the checkpoint number (e.g. `4999`).

**Keyboard controls in the viewer:**

| Key | Action |
|-----|--------|
| `↑` / `W` | Move forward |
| `↓` / `S` | Move backward |
| `←` / `A` | Strafe left |
| `→` / `D` | Strafe right |
| `Q` / `E` | Rotate left / right |
| `R` | Reset robot |
| `ESC` | Quit |

---

## Training from scratch

```bash
# Motion-tracking policy
cd Genesis/examples/locomotion/tracking
python g1_tracking_train.py -e my-tracking-run

# Velocity-command walking policy
cd Genesis/examples/locomotion
python g1_train.py -e my-walking-run
```

See `Genesis/examples/locomotion/configs/g1_config.yaml` and
`Genesis/examples/locomotion/tracking/configs/g1_tracking_config.yaml` for all
hyperparameters.
