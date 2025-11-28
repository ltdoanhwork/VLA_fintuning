#!/usr/bin/env python3
"""
Offline Evaluation Script: LIBERO + Isaac-GR00T
===============================================
Evaluates a fine-tuned GR00T model on the LIBERO dataset using offline metrics (MSE).
Generates plots comparing Ground Truth vs. Predicted actions.

Usage:
    python evaluate_groot_style.py --checkpoint output/libero_groot_training/checkpoint-20 --plot
"""

import sys
import os
import tyro
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

# Add Isaac-GR00T to path
sys.path.append('/home/serverai/ltdoanh/pi0_vggt/Isaac-GR00T')

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag

# Import config
from config import LiberoDataConfig

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    # Path to model checkpoint
    checkpoint: str = "output/libero_groot_training/checkpoint-20"
    
    # Path to dataset
    dataset_path: str = "./merged_libero_mask_depth_noops_lerobot_40"
    
    # Number of samples to evaluate (0 for all)
    num_samples: int = 10
    
    # Whether to generate plots
    plot: bool = False
    
    # Output directory for plots
    output_dir: str = "eval_results"
    
    # Denoising steps for inference
    denoising_steps: int = 8
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def plot_action_comparison(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    sample_idx: int,
    output_dir: str
):
    """
    Plot Ground Truth vs Predicted actions for a single sample.
    
    Args:
        gt_actions: Ground truth actions (T, D)
        pred_actions: Predicted actions (T, D)
        sample_idx: Index of the sample
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    action_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
    num_dims = gt_actions.shape[1]
    
    fig, axes = plt.subplots(num_dims, 1, figsize=(10, 2 * num_dims), sharex=True)
    if num_dims == 1:
        axes = [axes]
        
    time_steps = np.arange(gt_actions.shape[0])
    
    for i in range(num_dims):
        ax = axes[i]
        ax.plot(time_steps, gt_actions[:, i], label='Ground Truth', color='blue', linewidth=2)
        ax.plot(time_steps, pred_actions[:, i], label='Predicted', color='red', linestyle='--', linewidth=2)
        ax.set_ylabel(action_names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')
            
    axes[-1].set_xlabel('Time Step (Future Horizon)')
    plt.suptitle(f'Sample {sample_idx}: Action Trajectory Comparison')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"sample_{sample_idx}_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   📈 Plot saved to: {save_path}")


def main(cfg: EvalConfig):
    print("=" * 80)
    print("🎯 GR00T Offline Evaluation (MSE + Plots)")
    print("=" * 80)
    print(f"📁 Checkpoint: {cfg.checkpoint}")
    print(f"📦 Dataset: {cfg.dataset_path}")
    print(f"🔧 Device: {cfg.device}")
    
    # 1. Load Policy
    print(f"\n🤖 Loading policy...")
    data_config = LiberoDataConfig()
    
    try:
        policy = Gr00tPolicy(
            model_path=cfg.checkpoint,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            denoising_steps=cfg.denoising_steps,
            device=cfg.device,
        )
        print("✅ Policy loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        sys.exit(1)

    # 2. Load Dataset (Raw for GT)
    print(f"\n📦 Loading dataset (raw for Ground Truth)...")
    try:
        dataset_raw = LeRobotSingleDataset(
            dataset_path=cfg.dataset_path,
            modality_configs=data_config.modality_config(),
            video_backend="torchvision_av",
            video_backend_kwargs=None,
            transforms=None,  # No transforms to get raw values
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        )
        print(f"✅ Dataset loaded: {len(dataset_raw)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)

    # 3. Evaluation Loop
    num_samples = cfg.num_samples if cfg.num_samples > 0 else len(dataset_raw)
    num_samples = min(num_samples, len(dataset_raw))
    
    print(f"\n🚀 Starting evaluation on {num_samples} samples...")
    
    total_mse = 0.0
    mse_per_dim = np.zeros(7)
    
    # Select random indices if not evaluating all
    if num_samples < len(dataset_raw):
        indices = np.random.choice(len(dataset_raw), num_samples, replace=False)
        indices.sort()
    else:
        indices = range(len(dataset_raw))

    for i, idx in enumerate(indices):
        print(f"[{i+1}/{num_samples}] Processing sample {idx}...", end="\r")
        
        # Get sample
        raw_sample = dataset_raw[idx]
        
        # Run Inference
        with torch.no_grad():
            prediction = policy.get_action(raw_sample)
            
        # Extract Predictions
        pred_values = []
        pred_keys = sorted([k for k in prediction.keys() if k.startswith('action.')])
        for key in pred_keys:
            val = prediction[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            pred_values.append(val)
        pred_actions = np.concatenate(pred_values, axis=-1) # (T, D)
        
        # Extract Ground Truth
        gt_values = []
        gt_keys = sorted([k for k in raw_sample.keys() if k.startswith('action.')])
        for key in gt_keys:
            val = raw_sample[key]
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            gt_values.append(val)
        gt_actions = np.concatenate(gt_values, axis=-1) # (T, D)
        
        # Calculate MSE
        mse = np.mean((pred_actions - gt_actions) ** 2)
        total_mse += mse
        mse_per_dim += np.mean((pred_actions - gt_actions) ** 2, axis=0)
        
        # Plot if requested
        if cfg.plot and i < 5: # Limit plots to first 5 samples to avoid spam
            print(f"\n   Generating plot for sample {idx}...")
            plot_action_comparison(gt_actions, pred_actions, idx, cfg.output_dir)

    # 4. Summary
    avg_mse = total_mse / num_samples
    avg_mse_per_dim = mse_per_dim / num_samples
    
    print("\n" + "=" * 80)
    print("📊 Evaluation Summary")
    print("=" * 80)
    print(f"Samples Evaluated: {num_samples}")
    print(f"Average MSE:       {avg_mse:.6f}")
    
    print("\nPer-Dimension MSE:")
    action_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
    for name, val in zip(action_names, avg_mse_per_dim):
        status = "✅" if val < 0.01 else ("⚠️" if val < 0.1 else "❌")
        print(f"   {status} {name:<10}: {val:.6f}")
        
    print("\n" + "=" * 80)
    if cfg.plot:
        print(f"📈 Plots saved to: {os.path.abspath(cfg.output_dir)}")
    print("✅ Done!")

if __name__ == "__main__":
    cfg = tyro.cli(EvalConfig)
    main(cfg)
