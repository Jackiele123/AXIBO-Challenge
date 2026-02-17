"""
Keyboard Controls for G1 Humanoid:
↑ / w   - Move Forward
↓ / s   - Move Backward
← / a   - Move Left
→ / d   - Move Right
q       - Rotate Left (CCW)
e       - Rotate Right (CW)
r       - Reset Robot
esc     - Quit

Press 'i' in viewer to see additional viewer controls
"""

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
from genesis.vis.keybindings import Key, KeyAction, Keybind

from g1_env import G1Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1-walking")
    parser.add_argument("--ckpt", type=int, default=499)
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="CRITICAL", logger_verbose_time=False, performance_mode=False)

    log_dir = f"logs/{args.exp_name}"
    cfg_data = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    
    # Handle both old (5 items) and new (6 items) config formats
    if len(cfg_data) == 6:
        env_cfg, obs_cfg, reward_cfg, command_cfg, randomization_cfg, train_cfg = cfg_data
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfg_data
        randomization_cfg = {}  # No randomization for old checkpoints
    
    # Disable randomization during evaluation
    randomization_cfg["add_noise"] = False
    randomization_cfg["push_robots"] = False
    reward_cfg["reward_scales"] = {}
    
    # Disable automatic command resampling for manual keyboard control
    env_cfg["enable_command_resampling"] = False

    env = G1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        randomization_cfg=randomization_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    # Command velocities [lin_vel_x, lin_vel_y, ang_vel_z]
    commands = torch.zeros((1, 3), dtype=gs.tc_float, device=gs.device)
    
    # Control parameters
    lin_vel_step = 0.1  # m/s
    ang_vel_step = 0.1  # rad/s
    max_lin_vel = 1.5
    max_ang_vel = 1.5

    # Running state
    is_running = [True]  # Use list to make it mutable in callbacks

    def update_command(delta_x=0.0, delta_y=0.0, delta_yaw=0.0):
        """Update command velocities."""
        commands[0, 0] = torch.clamp(commands[0, 0] + delta_x, -max_lin_vel, max_lin_vel)
        commands[0, 1] = torch.clamp(commands[0, 1] + delta_y, -max_lin_vel, max_lin_vel)
        commands[0, 2] = torch.clamp(commands[0, 2] + delta_yaw, -max_ang_vel, max_ang_vel)
        env.commands[0] = commands[0]
        print(f"Commands: forward={commands[0, 0].item():.2f} m/s, lateral={commands[0, 1].item():.2f} m/s, yaw={commands[0, 2].item():.2f} rad/s")

    def stop_command():
        """Stop all movement."""
        commands.zero_()
        env.commands[0] = commands[0]
        print("Commands: STOPPED")

    def reset_env():
        """Reset the environment."""
        stop_command()
        env.reset()
        print("Environment reset")

    def quit_sim():
        """Quit simulation."""
        is_running[0] = False
        print("Quitting...")

    # Register keyboard controls
    env.scene.viewer.register_keybinds(
        # Forward/backward
        Keybind("forward", Key.UP, KeyAction.PRESS, callback=update_command, args=(lin_vel_step, 0, 0)),
        Keybind("backward", Key.DOWN, KeyAction.PRESS, callback=update_command, args=(-lin_vel_step, 0, 0)),
        
        # Left/right strafe
        Keybind("left", Key.LEFT, KeyAction.PRESS, callback=update_command, args=(0, lin_vel_step, 0)),
        Keybind("right", Key.RIGHT, KeyAction.PRESS, callback=update_command, args=(0, -lin_vel_step, 0)),
        
        # Rotation
        Keybind("rotate_left", Key.Q, KeyAction.PRESS, callback=update_command, args=(0, 0, ang_vel_step)),
        Keybind("rotate_right", Key.E, KeyAction.PRESS, callback=update_command, args=(0, 0, -ang_vel_step)),
        
        # Stop
        Keybind("stop", Key.SPACE, KeyAction.PRESS, callback=stop_command),
        
        # Reset
        Keybind("reset", Key._7, KeyAction.PRESS, callback=reset_env),
        
    )

    print("\n" + "="*60)
    print("G1 Humanoid Keyboard Control")
    print("="*60)
    print("Controls:")
    print("  ↑/W     : Increase forward velocity")
    print("  ↓/S     : Decrease forward velocity (backward)")
    print("  ←/A     : Increase left velocity")
    print("  →/D     : Increase right velocity")
    print("  Q       : Rotate left (CCW)")
    print("  E       : Rotate right (CW)")
    print("  SPACE   : Stop all movement")
    print("  R       : Reset environment")
    print("  ESC     : Quit")
    print("="*60 + "\n")

    obs, _ = env.reset()
    
    try:
        with torch.no_grad():
            while is_running[0]:
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                
                if dones[0]:
                    print("Episode finished, resetting...")
                    obs, _ = env.reset()
                    update_command(0,0,0)  # Reset commands on episode end
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        print("Simulation finished")


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/g1_eval.py -e g1-walking --ckpt 100

# evaluation with specific experiment
python examples/locomotion/g1_eval.py -e g1-walking-v1 --ckpt 500
"""
