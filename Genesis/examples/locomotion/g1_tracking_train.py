"""Training script for G1 humanoid motion tracking."""

import argparse
import os
import pickle
import shutil
import yaml
from importlib import metadata

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


def get_train_cfg(exp_name, max_iterations):
    """Load training configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), "configs", "g1_tracking_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_cfg = config["training"]
    
    # Build train_cfg_dict from YAML
    train_cfg_dict = {
        "algorithm": training_cfg["algorithm"],
        "init_member_classes": {},
        "policy": training_cfg["policy"],
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": training_cfg["runner"]["log_interval"],
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": training_cfg["runner"]["num_steps_per_env"],
        "save_interval": training_cfg["runner"]["save_interval"],
        "empirical_normalization": None,
        "seed": training_cfg["runner"]["seed"],
    }

    return train_cfg_dict


def get_cfgs():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), "configs", "g1_tracking_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract config sections
    env_cfg = config["env"]
    obs_cfg = config["observation"]
    
    # Extract reward config
    reward_cfg = config["rewards"]["parameters"]
    reward_cfg["reward_scales"] = config["rewards"]["scales"]
    
    command_cfg = config["commands"]
    randomization_cfg = config["randomization"]
    tracking_cfg = config["tracking"]
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg, randomization_cfg, tracking_cfg


def main():
    parser = argparse.ArgumentParser(description="Train G1 humanoid for motion tracking")
    parser.add_argument("-e", "--exp_name", type=str, default="g1-tracking", 
                        help="Experiment name for logging")
    parser.add_argument("-B", "--num_envs", type=int, default=4096,
                        help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=10000,
                        help="Maximum training iterations")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=int, default=-1,
                        help="Checkpoint iteration to resume from (-1 for latest)")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    
    # Load configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg, randomization_cfg, tracking_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    
    # Handle resume
    if args.resume:
        train_cfg["runner"]["resume"] = True
        train_cfg["runner"]["checkpoint"] = args.checkpoint
        print(f"Resuming training from checkpoint {args.checkpoint}")
    else:
        # Clean log directory if not resuming
        if os.path.exists(log_dir):
            response = input(f"Log directory {log_dir} exists. Delete it? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(log_dir)
                print(f"Deleted {log_dir}")
            else:
                print("Training cancelled")
                return
        os.makedirs(log_dir, exist_ok=True)

    # Save configurations
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, randomization_cfg, tracking_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Initialize Genesis
    print("Initializing Genesis...")
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=train_cfg["seed"],
        performance_mode=True
    )

    # Create tracking environment
    print("Creating tracking environment...")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Motion file: {tracking_cfg['motion_file']}")
    
    env = G1TrackingEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        tracking_cfg=tracking_cfg,
        randomization_cfg=randomization_cfg,
        show_viewer=False,
    )

    print(f"\nEnvironment created:")
    print(f"  Observation dim: {env.num_obs}")
    print(f"  Action dim: {env.num_actions}")
    print(f"  Episode length: {env.max_episode_length} steps")
    print(f"  Control frequency: {1.0/env.dt:.1f} Hz")
    
    print(f"\nReward scales:")
    for name, scale in sorted(reward_cfg["reward_scales"].items()):
        if scale != 0.0:
            print(f"  {name}: {scale}")

    # Create runner
    print("\nInitializing PPO runner...")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Start training
    print(f"\nStarting training for {args.max_iterations} iterations...")
    print(f"Logs will be saved to: {log_dir}")
    print("="*80)
    
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    print("="*80)
    print("Training complete!")
    print(f"Final model saved to: {log_dir}/model_{args.max_iterations}.pt")


if __name__ == "__main__":
    main()

"""
Usage Examples:

# Basic training with default settings
python examples/locomotion/g1_tracking_train.py

# Custom experiment name and parameters
python examples/locomotion/g1_tracking_train.py -e g1-walk-jump -B 2048 --max_iterations 5000

# Resume training from checkpoint
python examples/locomotion/g1_tracking_train.py -e g1-tracking --resume --checkpoint 1000

# Quick test run with fewer environments
python examples/locomotion/g1_tracking_train.py -e test -B 512 --max_iterations 100
"""
