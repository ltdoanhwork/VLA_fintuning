#!/usr/bin/env python3
"""
Visualization Script: LIBERO + Isaac-GR00T
==========================================
Visualize predicted vs ground truth actions.

Usage:
    python visualize.py
    python visualize.py --num-samples 5 --show

Author: ltdoanh
Date: 2025-11-24
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add Isaac-GR00T to path
sys.path.append('/home/serverai/ltdoanh/pi0_vggt/Isaac-GR00T')

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag

# Import config
from config import LiberoDataConfig


def plot_actions(predicted, ground_truth, action_dim_names, save_path=None):
    """Plot predicted vs ground truth actions."""
    horizon = predicted.shape[0]
    n_dims = min(predicted.shape[1], len(action_dim_names))
    
    # Create subplots
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 3*n_dims))
    if n_dims == 1:
        axes = [axes]
    
    timesteps = np.arange(horizon)
    
    for i, (ax, name) in enumerate(zip(axes, action_dim_names[:n_dims])):
        # Plot predicted
        ax.plot(timesteps, predicted[:, i], 'b-', linewidth=2, label='Predicted', marker='o', markersize=4)
        
        # Plot ground truth
        ax.plot(timesteps, ground_truth[:, i], 'r--', linewidth=2, label='Ground Truth', marker='s', markersize=4)
        
        # Styling
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} - Predicted vs Ground Truth', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Calculate and show MSE
        mse = np.mean((predicted[:, i] - ground_truth[:, i]) ** 2)
        ax.text(0.98, 0.02, f'MSE: {mse:.6f}', 
                transform=ax.transAxes, 
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved plot to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize predicted vs ground truth actions")
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
        help="Sample index to visualize"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/visualizations",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 Action Visualization")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Device: {device}")
    
    # Load policy
    print(f"\n🤖 Loading policy from: {args.checkpoint}")
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
    
    # Load dataset WITHOUT transforms to get ground truth
    print(f"\n📦 Loading dataset (no transforms for GT)...")
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=data_config.modality_config(),
        video_backend="torchvision_av",
        video_backend_kwargs=None,
        transforms=None,  # Important: no transforms to preserve GT!
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Action dimension names
    action_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
    
    # Process samples
    print(f"\n🎯 Processing {args.num_samples} sample(s)...")
    print("=" * 80)
    
    for i in range(args.num_samples):
        sample_idx = args.sample_idx + i
        
        if sample_idx >= len(dataset):
            print(f"\n⚠️  Sample {sample_idx} exceeds dataset size")
            break
        
        print(f"\n📝 Sample {sample_idx}:")
        
        # Get sample
        sample = dataset[sample_idx]
        
        # Run inference
        with torch.no_grad():
            prediction = policy.get_action(sample)
        
        # Extract actions
        if any(key.startswith('action.') for key in prediction.keys()):
            # Denormalized format
            action_keys = sorted([k for k in prediction.keys() if k.startswith('action.')])
            
            action_values = []
            for key in action_keys:
                val = prediction[key]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                action_values.append(val)
            
            # Combine: (16,1) * 7 -> (16,7)
            pred_actions = np.concatenate(action_values, axis=-1)
        
        elif 'action_pred' in prediction:
            pred_actions = prediction['action_pred']
            if isinstance(pred_actions, torch.Tensor):
                pred_actions = pred_actions.cpu().numpy()
            pred_actions = pred_actions[0]  # Remove batch dim
        
        else:
            print(f"   ⚠️  Unexpected format")
            continue
        
        # Get ground truth from sample
        if 'action' in sample:
            gt_actions = sample['action']
            if isinstance(gt_actions, torch.Tensor):
                gt_actions = gt_actions.cpu().numpy()
        elif any(key.startswith('action.') for key in sample.keys()):
            # Extract from action.* keys
            gt_keys = sorted([k for k in sample.keys() if k.startswith('action.')])
            gt_values = []
            for key in gt_keys:
                val = sample[key]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                gt_values.append(val)
            gt_actions = np.concatenate(gt_values, axis=-1)
        else:
            print(f"   ⚠️  No ground truth available")
            continue
        
        print(f"   Predicted shape: {pred_actions.shape}")
        print(f"   Ground truth shape: {gt_actions.shape}")
        
        # Calculate MSE
        mse = np.mean((pred_actions - gt_actions) ** 2)
        mse_per_dim = np.mean((pred_actions - gt_actions) ** 2, axis=0)
        
        print(f"   Overall MSE: {mse:.6f}")
        print(f"   Per-dimension MSE:")
        for name, mse_val in zip(action_names, mse_per_dim):
            print(f"      {name}: {mse_val:.6f}")
        
        # Plot
        save_path = output_dir / f"sample_{sample_idx}_actions.png"
        fig = plot_actions(pred_actions, gt_actions, action_names, save_path)
        
        if args.show:
            plt.show()
        else:
            plt.close(fig)
    
    print("\n" + "=" * 80)
    print(f"✅ Visualization complete!")
    print(f"📁 Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
