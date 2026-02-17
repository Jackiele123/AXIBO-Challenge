"""
Utility script to inspect and validate NPZ motion files.
Helps verify the data structure matches expected format.
"""

import argparse
import numpy as np
import sys


def inspect_npz(filepath):
    """Inspect NPZ file structure and contents."""
    print(f"\n{'='*80}")
    print(f"Inspecting NPZ file: {filepath}")
    print(f"{'='*80}\n")
    
    try:
        data = np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return False
    except Exception as e:
        print(f"Error loading file: {e}")
        return False
    
    # List all keys
    print("Available keys:")
    for key in data.keys():
        print(f"  - {key}")
    print()
    
    # Expected keys
    expected_keys = [
        'fps', 'joint_names', 'body_names',
        'joint_pos', 'joint_vel',
        'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'
    ]
    
    missing_keys = [k for k in expected_keys if k not in data.keys()]
    if missing_keys:
        print(f"Warning: Missing expected keys: {missing_keys}\n")
    
    # Print metadata
    print("Metadata:")
    if 'fps' in data:
        fps = int(data['fps']) if hasattr(data['fps'], '__int__') else data['fps']
        print(f"  FPS: {fps}")
    
    if 'joint_names' in data:
        joint_names = list(data['joint_names'])
        print(f"  Joint names ({len(joint_names)}):")
        for i, name in enumerate(joint_names):
            print(f"    [{i:2d}] {name}")
    
    if 'body_names' in data:
        body_names = list(data['body_names'])
        print(f"\n  Body names ({len(body_names)}):")
        for i, name in enumerate(body_names):
            print(f"    [{i:2d}] {name}")
    
    print()
    
    # Print array shapes and statistics
    print("Data arrays:")
    
    array_keys = ['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 
                  'body_lin_vel_w', 'body_ang_vel_w']
    
    for key in array_keys:
        if key not in data:
            print(f"  {key}: MISSING")
            continue
        
        arr = data[key]
        print(f"  {key}:")
        print(f"    Shape: {arr.shape}")
        print(f"    Dtype: {arr.dtype}")
        print(f"    Range: [{arr.min():.4f}, {arr.max():.4f}]")
        print(f"    Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")
    
    # Calculate motion duration
    if 'joint_pos' in data and 'fps' in data:
        num_frames = data['joint_pos'].shape[0]
        fps = int(data['fps']) if hasattr(data['fps'], '__int__') else data['fps']
        duration = num_frames / fps
        print(f"\nMotion summary:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Frames: {num_frames}")
        print(f"  Frame rate: {fps} FPS")
        print(f"  Time step: {1/fps:.4f} s")
    
    # Validate data consistency
    print(f"\nData validation:")
    issues = []
    
    if 'joint_pos' in data and 'joint_vel' in data:
        if data['joint_pos'].shape != data['joint_vel'].shape:
            issues.append("joint_pos and joint_vel shapes don't match")
    
    if 'body_pos_w' in data and 'body_quat_w' in data:
        if data['body_pos_w'].shape[:-1] != data['body_quat_w'].shape[:-1]:
            issues.append("body_pos_w and body_quat_w frame/body counts don't match")
    
    if 'joint_pos' in data and 'joint_names' in data:
        expected_joints = len(data['joint_names'])
        actual_joints = data['joint_pos'].shape[1]
        if expected_joints != actual_joints:
            issues.append(f"joint_names count ({expected_joints}) doesn't match joint_pos ({actual_joints})")
    
    if 'body_pos_w' in data and 'body_names' in data:
        expected_bodies = len(data['body_names'])
        actual_bodies = data['body_pos_w'].shape[1]
        if expected_bodies != actual_bodies:
            issues.append(f"body_names count ({expected_bodies}) doesn't match body_pos_w ({actual_bodies})")
    
    # Check for NaN or Inf
    for key in array_keys:
        if key in data:
            arr = data[key]
            if np.isnan(arr).any():
                issues.append(f"{key} contains NaN values")
            if np.isinf(arr).any():
                issues.append(f"{key} contains Inf values")
    
    if issues:
        print("  Issues found:")
        for issue in issues:
            print(f"    ⚠ {issue}")
    else:
        print("  ✓ All validations passed")
    
    # Sample data preview
    if 'joint_pos' in data:
        print(f"\nSample joint positions (frame 0):")
        print(f"  {data['joint_pos'][0]}")
    
    if 'body_pos_w' in data:
        print(f"\nSample body position (frame 0, body 0):")
        print(f"  {data['body_pos_w'][0, 0]}")
    
    print(f"\n{'='*80}\n")
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Inspect NPZ motion files")
    parser.add_argument("filepath", type=str, 
                        help="Path to NPZ file")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare with another NPZ file")
    args = parser.parse_args()
    
    # Inspect primary file
    success = inspect_npz(args.filepath)
    
    # Compare if requested
    if args.compare:
        print("\n" + "="*80)
        print("COMPARISON WITH SECOND FILE")
        print("="*80)
        success2 = inspect_npz(args.compare)
        
        if success and success2:
            print("\nBoth files are valid!")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Usage:

# Inspect motion file
python examples/locomotion/inspect_motion.py ../g1_files/g1_12dof_package/walk_jump_bothfeet_4_small_12dof_80fps.npz

# Compare two motion files
python examples/locomotion/inspect_motion.py motion1.npz --compare motion2.npz
"""
