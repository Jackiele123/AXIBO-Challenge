"""Evaluation / visualisation for the G1 BeyondMimic tracking policy.

The motion drives itself and loops automatically when it completes.
Phase progress and per-body tracking RMSE are printed each step.

Usage (from tracking/):
    python g1_tracking_eval.py -e g1-tracking --ckpt 499

Press ESC in the viewer to quit.
"""

import argparse
import math
import os
import pickle

import torch

# ── working directory: must be locomotion/ for relative URDF paths ──────────
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importlib import metadata
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs

from g1_tracking_env import G1TrackingEnv
from motion_lib import quat_error_magnitude


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1-tracking")
    parser.add_argument("--ckpt",           type=int, default=499)
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="CRITICAL")

    log_dir = f"logs/{args.exp_name}"
    cfg_data = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    env_cfg, obs_cfg, reward_cfg, motion_cfg, train_cfg = cfg_data

    # Disable reward computation and observation noise during eval
    reward_cfg["reward_scales"] = {}
    obs_cfg["add_noise"] = False

    env = G1TrackingEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        motion_cfg=motion_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    is_running = [True]

    print("\n" + "=" * 60)
    print("G1 BeyondMimic Tracking Evaluation")
    print("=" * 60)
    print(f"  Motion duration : {env.motion_lib.duration:.2f} s")
    print(f"  FPS             : {env.motion_lib.fps}")
    print(f"  Frames          : {env.motion_lib.num_frames}")
    print(f"  Bodies          : {env.motion_lib.num_bodies}")
    print("=" * 60 + "\n")

    obs, _ = env.reset()
    step = 0

    try:
        with torch.no_grad():
            while is_running[0]:
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                step += 1

                # ── Diagnostics every 10 steps ────────────────────────────
                if step % 10 == 0:
                    t = env.motion_time[0].item()
                    phase = t / env.motion_lib.duration
                    pct   = phase * 100.0

                    # Body position RMSE across 13 bodies
                    pos_err = torch.sqrt(torch.mean(
                        torch.sum(
                            torch.square(
                                env.ref_body_pos_relative_w
                                - env.robot_body_pos_relative_w
                            ),
                            dim=-1,
                        )
                    )).item()

                    # Body orientation RMSE
                    N, B = 1, env.num_bodies
                    ori_err = torch.sqrt(torch.mean(
                        quat_error_magnitude(
                            env.ref_body_quat_relative_w.reshape(N * B, 4),
                            env.robot_body_quat_relative_w.reshape(N * B, 4),
                        ) ** 2
                    )).item()

                    # Anchor position error
                    anc_err = torch.norm(
                        env.ref_anchor_pos_w[0] - env.robot_anchor_pos[0]
                    ).item()

                    print(
                        f"t={t:5.2f}s  phase={pct:5.1f}%  "
                        f"pos_rmse={pos_err:.3f}m  "
                        f"ori_rmse={math.degrees(ori_err):.1f}°  "
                        f"anchor_err={anc_err:.3f}m"
                    )

                if dones[0]:
                    print("\n── Sequence ended, restarting ──\n")
                    obs, _ = env.reset()
                    step = 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Done.")


if __name__ == "__main__":
    main()
