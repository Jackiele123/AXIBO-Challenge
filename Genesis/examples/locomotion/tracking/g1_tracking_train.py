"""Training entry point for G1 BeyondMimic motion-tracking.

Run from Genesis/examples/locomotion/tracking/:
    python g1_tracking_train.py
    python g1_tracking_train.py -e g1-tracking-v2 -B 2048 --max_iterations 1000
    python g1_tracking_train.py -e g1-tracking --resume --ckpt 500

The script changes directory to Genesis/examples/locomotion/ at startup so
that all relative paths (URDF ../g1_files/..., log dirs, motion NPZ) match
the existing scripts.
"""

import argparse
import os
import pickle
import shutil
import yaml
from importlib import metadata

# ── working directory: must be locomotion/ for relative URDF paths ──────────
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ── config helpers ────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs", "g1_tracking_config.yaml",
)


def load_yaml() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_cfgs():
    cfg = load_yaml()
    env_cfg    = cfg["env"]
    obs_cfg    = cfg["observation"]
    motion_cfg = cfg["motion"]

    reward_cfg = cfg["rewards"]
    reward_cfg["reward_scales"] = reward_cfg.pop("scales")

    return env_cfg, obs_cfg, reward_cfg, motion_cfg


def get_train_cfg(exp_name: str, max_iterations: int) -> dict:
    cfg = load_yaml()
    training = cfg["training"]
    return {
        "algorithm": training["algorithm"],
        "init_member_classes": {},
        "policy": training["policy"],
        "runner": {
            "checkpoint":        -1,
            "experiment_name":   exp_name,
            "load_run":          -1,
            "log_interval":      training["runner"]["log_interval"],
            "max_iterations":    max_iterations,
            "record_interval":   -1,
            "resume":            False,
            "resume_path":       None,
            "run_name":          "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": training["runner"]["num_steps_per_env"],
        "save_interval":     training["runner"]["save_interval"],
        "empirical_normalization": None,
        "seed":              training["runner"]["seed"],
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",       type=str, default="g1-tracking")
    parser.add_argument("-B", "--num_envs",        type=int, default=4096)
    parser.add_argument("--max_iterations",        type=int, default=50000)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from a checkpoint")
    parser.add_argument("--ckpt",   type=int, default=500,
                        help="Checkpoint iteration to resume from")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, motion_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if args.resume:
        resume_path = f"{log_dir}/model_{args.ckpt}.pt"
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        train_cfg["runner"]["resume"]      = True
        train_cfg["runner"]["resume_path"] = resume_path
        train_cfg["runner"]["load_run"]    = args.exp_name
        train_cfg["runner"]["checkpoint"]  = args.ckpt
        print(f"Resuming from {resume_path}")
    else:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, motion_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=train_cfg["seed"],
        performance_mode=True,
    )

    env = G1TrackingEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        motion_cfg=motion_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if args.resume:
        print(f"Loading checkpoint: {train_cfg['runner']['resume_path']}")
        runner.load(train_cfg["runner"]["resume_path"])

    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
