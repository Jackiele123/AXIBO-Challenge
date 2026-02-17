"""Evaluation script for G1 humanoid motion tracking with visualization."""

import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from g1_tracking_env import G1TrackingEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate G1 motion tracking policy")
    parser.add_argument("-e", "--exp_name", type=str, default="g1-tracking",
                        help="Experiment name")
    parser.add_argument("--ckpt", type=int, default=499,
                        help="Checkpoint iteration to load")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of environments to visualize")
    parser.add_argument("--hard_sync", action="store_true",
                        help="Hard sync robot to reference motion at start")
    parser.add_argument("--loop", action="store_true",
                        help="Loop evaluation continuously")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    
    # Load configurations
    try:
        cfg_data = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    except FileNotFoundError:
        print(f"Error: Configuration file not found in {log_dir}")
        print("Please ensure the experiment has been trained.")
        return
    
    # Handle both old and new config formats
    if len(cfg_data) == 7:
        env_cfg, obs_cfg, reward_cfg, command_cfg, randomization_cfg, tracking_cfg, train_cfg = cfg_data
    elif len(cfg_data) == 6:
        # Old format without tracking_cfg
        print("Warning: Old config format detected. Using default tracking config.")
        env_cfg, obs_cfg, reward_cfg, command_cfg, randomization_cfg, train_cfg = cfg_data
        # Create minimal tracking config
        tracking_cfg = {
            "motion_file": "/home/jackiele/projects/AXIBO-Challenge/Genesis/examples/locomotion/configs/walk_jump_bothfeet_4_small_12dof_80fps.npz",
            "tracking_link_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "include_ref_states": True,
            "include_phase": True,
            "include_tracking_error": False,
            "hard_reset_ratio": 0.8,
            "match_motion_duration": True,
            "termination_thresholds": {},
        }
    else:
        print(f"Error: Unexpected config format with {len(cfg_data)} items")
        return
    
    # Disable randomization for evaluation
    randomization_cfg["add_noise"] = False
    randomization_cfg["push_robots"] = False
    
    # Disable reward computation for cleaner output
    reward_cfg["reward_scales"] = {}
    
    # Override hard reset ratio if requested
    if args.hard_sync:
        tracking_cfg["hard_reset_ratio"] = 1.0
        print("Hard sync enabled: Robot will start exactly at reference motion")
    
    # Create environment
    print(f"Loading experiment: {args.exp_name}")
    print(f"Motion file: {tracking_cfg['motion_file']}")
    
    env = G1TrackingEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        tracking_cfg=tracking_cfg,
        randomization_cfg=randomization_cfg,
        show_viewer=True,
    )

    # Load policy
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    
    try:
        runner.load(resume_path)
        print(f"Loaded checkpoint: {resume_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {resume_path}")
        return
    
    policy = runner.get_inference_policy(device=gs.device)

    print("\n" + "="*80)
    print("G1 Motion Tracking Evaluation")
    print("="*80)
    print(f"  Experiment: {args.exp_name}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Motion duration: {env.motion_lib.motion_length:.2f}s")
    print(f"  Episode length: {env.max_episode_length} steps")
    print(f"  Hard sync: {args.hard_sync}")
    print("="*80)
    print("\nPress 'ESC' to quit, 'R' to reset environment")
    print("Press 'I' in viewer for additional controls")
    print("="*80 + "\n")

    obs, _ = env.reset()
    episode_count = 0
    step_count = 0
    
    # Tracking metrics
    joint_errors = []
    base_height_errors = []
    
    try:
        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                
                # Compute tracking errors for logging
                joint_error = torch.mean(torch.abs(env.diff_dof_pos)).item()
                height_error = torch.mean(torch.abs(env.base_pos[:, 2] - env.ref_base_pos[:, 2])).item()
                
                joint_errors.append(joint_error)
                base_height_errors.append(height_error)
                step_count += 1
                
                # Print progress every 50 steps
                if step_count % 50 == 0:
                    avg_joint_err = sum(joint_errors[-50:]) / len(joint_errors[-50:])
                    avg_height_err = sum(base_height_errors[-50:]) / len(base_height_errors[-50:])
                    phase = env.phase[0].item()
                    print(f"Step {step_count:4d} | Joint error: {avg_joint_err:.4f} rad | "
                          f"Height error: {avg_height_err:.4f} m | Phase: {phase:.2f}")
                
                if dones[0]:
                    episode_count += 1
                    avg_joint_err = sum(joint_errors) / len(joint_errors) if joint_errors else 0
                    avg_height_err = sum(base_height_errors) / len(base_height_errors) if base_height_errors else 0
                    
                    print("\n" + "-"*80)
                    print(f"Episode {episode_count} complete!")
                    print(f"  Average joint tracking error: {avg_joint_err:.4f} rad")
                    print(f"  Average height tracking error: {avg_height_err:.4f} m")
                    print("-"*80 + "\n")
                    
                    # Reset tracking
                    joint_errors = []
                    base_height_errors = []
                    step_count = 0
                    
                    if not args.loop:
                        print("Evaluation complete. Use --loop to continue.")
                        break
                    
                    obs, _ = env.reset()
                    print(f"Starting episode {episode_count + 1}...")
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    
    finally:
        print("\nEvaluation finished")
        if episode_count > 0:
            print(f"Total episodes completed: {episode_count}")


if __name__ == "__main__":
    main()

"""
Usage Examples:

# Basic evaluation with visualization
python examples/locomotion/g1_tracking_eval.py -e g1-tracking --ckpt 500

# Evaluate with hard synchronization at start
python examples/locomotion/g1_tracking_eval.py -e g1-tracking --ckpt 1000 --hard_sync

# Continuous evaluation loop
python examples/locomotion/g1_tracking_eval.py -e g1-tracking --ckpt 1000 --loop

# Evaluate with multiple environments
python examples/locomotion/g1_tracking_eval.py -e g1-tracking --ckpt 1000 --num_envs 4
"""
