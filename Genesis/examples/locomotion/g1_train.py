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

from g1_env import G1Env


def get_train_cfg(exp_name, max_iterations):
    """Load training configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), "configs", "g1_config.yaml")
    
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

# Configs from https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py
def get_cfgs():
    """Load configuration from YAML file"""
    # Load YAML config
    config_path = os.path.join(os.path.dirname(__file__), "configs", "g1_config.yaml")
    
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
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg, randomization_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--ckpt", type=int, default=500, help="Checkpoint to resume from")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, randomization_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Handle resume logic
    if args.resume:
        resume_path = f"{log_dir}/model_{args.ckpt}.pt"
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        train_cfg["runner"]["resume"] = True
        train_cfg["runner"]["resume_path"] = resume_path
        train_cfg["runner"]["load_run"] = args.exp_name
        train_cfg["runner"]["checkpoint"] = args.ckpt
        print(f"Resuming training from {resume_path}")
    else:
        # Only clear log directory if not resuming
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, randomization_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=train_cfg["seed"], performance_mode=True)

    env = G1Env(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        randomization_cfg=randomization_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading model checkpoint: {train_cfg['runner']['resume_path']}")
        runner.load(train_cfg["runner"]["resume_path"])
        print(f"Resuming from iteration {args.ckpt}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/g1_train.py

# training with custom parameters
python examples/locomotion/g1_train.py -e g1-walking-v1 -B 2048 --max_iterations 1000
"""
