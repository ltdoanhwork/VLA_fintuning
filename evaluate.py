#!/usr/bin/env python3
"""
Evaluation Script: LIBERO + Isaac-GR00T
=======================================
Evaluate model performance using MSE against ground truth.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoint-20 --sample-idx 10

Author: ltdoanh
Date: 2025-11-24
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add Isaac-GR00T to path
sys.path.append('/home/serverai/ltdoanh/pi0_vggt/Isaac-GR00T')

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag

# Import config
from config import LiberoDataConfig


def main():
    parser = argparse.ArgumentParser(description="Evaluate LIBERO + Isaac-GR00T model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/libero_groot_training/checkpoint-20",
        help="Path to checkpoint"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./merged_libero_mask_depth_noops_lerobot_40",
        help="Path to dataset"
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Sample index to test"
    )
    
    args = parser.parse_args()

    print("=" * 80)
    print("🎯 Model Evaluation (MSE)")
    print("=" * 80)

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"\n❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"\n📁 Checkpoint: {args.checkpoint}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")

    # Load policy
    print(f"\n🤖 Loading policy...")
    data_config = LiberoDataConfig()

    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=8,
        device=device,
    )
    print("✅ Policy loaded!")

    # Load dataset WITHOUT transforms to get raw data including actions
    print(f"\n📦 Loading dataset (no transforms for GT)...")
    dataset_raw = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=data_config.modality_config(),
        video_backend="torchvision_av",
        video_backend_kwargs=None,
        transforms=None,  # Important: no transforms!
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )
    print(f"✅ Dataset loaded: {len(dataset_raw)} samples")

    # Get sample
    print(f"\n📝 Getting sample {args.sample_idx}...")
    if args.sample_idx >= len(dataset_raw):
        print(f"❌ Sample index {args.sample_idx} out of range")
        sys.exit(1)
        
    raw_sample = dataset_raw[args.sample_idx]

    # Run inference
    print(f"\n🎯 Running inference...")
    with torch.no_grad():
        prediction = policy.get_action(raw_sample)

    # Get predicted actions
    print(f"\n📊 Extracting predictions...")
    if any(key.startswith('action.') for key in prediction.keys()):
        pred_keys = sorted([k for k in prediction.keys() if k.startswith('action.')])
        pred_values = []
        for key in pred_keys:
            val = prediction[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            pred_values.append(val)
        pred_actions = np.concatenate(pred_values, axis=-1)  # (16, 7)
        print(f"   Predicted shape: {pred_actions.shape}")
        print(f"   First prediction (t=0): {pred_actions[0]}")
    else:
        print(f"   ❌ No action.* keys in prediction")
        sys.exit(1)

    # Get ground truth actions from raw sample
    print(f"\n📊 Extracting ground truth...")
    if 'action' in raw_sample:
        # Single 'action' key
        gt_actions = raw_sample['action']
        if isinstance(gt_actions, torch.Tensor):
            gt_actions = gt_actions.cpu().numpy()
        print(f"   Found 'action' key!")
    elif any(key.startswith('action.') for key in raw_sample.keys()):
        # Separate action.* keys
        gt_keys = sorted([k for k in raw_sample.keys() if k.startswith('action.')])
        gt_values = []
        for key in gt_keys:
            val = raw_sample[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            gt_values.append(val)
        gt_actions = np.concatenate(gt_values, axis=-1)  # (16, 7)
        print(f"   Found action.* keys!")
    else:
        print(f"   ❌ No action data in raw sample")
        sys.exit(1)

    print(f"   Ground truth shape: {gt_actions.shape}")
    print(f"   First GT (t=0): {gt_actions[0]}")

    # Calculate MSE
    print(f"\n✅ MSE Calculation:")

    # First action MSE
    mse_first = np.mean((pred_actions[0] - gt_actions[0]) ** 2)
    print(f"   MSE (first action): {mse_first:.6f}")

    # All actions MSE
    mse_all = np.mean((pred_actions - gt_actions) ** 2)
    print(f"   MSE (all actions): {mse_all:.6f}")

    # Per-dimension MSE
    mse_per_dim = np.mean((pred_actions - gt_actions) ** 2, axis=0)
    print(f"\n   Per-dimension MSE:")

    action_names = ['Gripper', 'Pitch', 'Roll', 'X', 'Y', 'Yaw', 'Z']
    for name, mse_val in zip(action_names, mse_per_dim):
        if mse_val < 0.01:
            status = "✅"
        elif mse_val < 0.1:
            status = "⚠️"
        else:
            status = "❌"
        print(f"      {status} {name}: {mse_val:.6f}")

    # Assessment
    print(f"\n📈 Assessment:")
    if mse_all < 0.001:
        print(f"   ⭐⭐⭐ Excellent! Model is very accurate (MSE < 0.001)")
    elif mse_all < 0.01:
        print(f"   ⭐⭐ Very Good! Model performs well (MSE < 0.01)")
    elif mse_all < 0.1:
        print(f"   ⭐ Good! Acceptable performance (MSE < 0.1)")
    else:
        print(f"   ⚠️ Needs Improvement! Consider training more (MSE > 0.1)")

    print("\n" + "=" * 80)
    print("✅ Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
