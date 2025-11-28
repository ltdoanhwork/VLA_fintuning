#!/usr/bin/env python3
"""
Custom Training Script: LIBERO + Isaac-GR00T Integration
=========================================================
This script integrates LIBERO dataset with Isaac-GR00T for custom training.

Author: ltdoanh
Date: 2025-11-24
"""

import sys
import os
import torch
from pathlib import Path

# Add Isaac-GR00T to path
sys.path.append('/home/serverai/ltdoanh/pi0_vggt/Isaac-GR00T')

from gr00t.experiment.data_config import BaseDataConfig, ModalityConfig
from gr00t.data.transform import (
    VideoToTensor,
    VideoResize,
    VideoColorJitter,
    VideoToNumpy,
    StateActionToTensor,
    ComposedModalityTransform,
)
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionTransform
from gr00t.model.transforms import GR00TTransform
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.experiment.runner import TrainRunner
from transformers import TrainingArguments


class LiberoDataConfig(BaseDataConfig):
    """
    Data configuration for LIBERO dataset with multi-modal inputs.
    Supports RGB images, depth maps, and segmentation masks.
    """
    
    video_keys = [
        "video.image",
        "video.wrist_image",
        "video.image_depth",
        "video.wrist_depth",
        "video.image_mask",
        "video.wrist_mask",
        "video.object_of_interest_mask",
        "video.object_of_interest_wrist_mask",
    ]

    state_keys = [
        "state.x", "state.y", "state.z",
        "state.roll", "state.pitch", "state.yaw",
        "state.gripper",
    ]

    action_keys = [
        "action.x", "action.y", "action.z",
        "action.roll", "action.pitch", "action.yaw",
        "action.gripper",
    ]

    def modality_config(self):
        """Define modality configuration for video, state, and action."""
        return {
            "video": ModalityConfig(
                modality_keys=self.video_keys,
                delta_indices=[0],  # Current frame only
            ),
            "state": ModalityConfig(
                modality_keys=self.state_keys,
                delta_indices=[0],  # Current state only
            ),
            "action": ModalityConfig(
                modality_keys=self.action_keys,
                delta_indices=list(range(16)),  # 16 future actions
            ),
        }

    def transform(self) -> ComposedModalityTransform:
        """Define data transformations for augmentation and preprocessing."""
        transforms = [
            # Video transformations (RGB + depth + mask)
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=256, width=256),
            VideoColorJitter(
                apply_to=["video.image", "video.wrist_image"],  # Only augment RGB
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.1,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            
            # State transformations
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            
            # Action transformations
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            
            # Concat transforms - combine video/state/action into single tensors
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            
            # GR00T model-specific transform - creates eagle_* keys
            GR00TTransform(
                state_horizon=1,  # Current state only
                action_horizon=16,  # 16 future actions
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


def setup_device():
    """Setup CUDA device."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device


def load_dataset(dataset_path: str):
    """
    Load LIBERO dataset with custom configuration.
    
    Args:
        dataset_path: Path to the LIBERO dataset
        
    Returns:
        LeRobotSingleDataset instance
    """
    print(f"\n📦 Loading dataset from: {dataset_path}")
    
    data_config = LiberoDataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend="torchvision_av",
        video_backend_kwargs=None,
        transforms=modality_transform,  # ← Important: set transforms here!
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(dataset)}")
    
    return dataset


def load_model(base_model_path: str, device: str):
    """
    Load GR00T model with custom fine-tuning configuration.
    
    Args:
        base_model_path: Path or HuggingFace model ID
        device: Device to load model on
        
    Returns:
        GR00T_N1_5 model instance
    """
    print(f"\n🤖 Loading GR00T model from: {base_model_path}")
    
    # Fine-tuning configuration
    TUNE_LLM = False            # Freeze LLM backbone
    TUNE_VISUAL = False         # Freeze visual encoder
    TUNE_PROJECTOR = True       # Train projector
    TUNE_DIFFUSION_MODEL = True # Train diffusion model
    
    print(f"   Tune LLM: {TUNE_LLM}")
    print(f"   Tune Visual: {TUNE_VISUAL}")
    print(f"   Tune Projector: {TUNE_PROJECTOR}")
    print(f"   Tune Diffusion: {TUNE_DIFFUSION_MODEL}")
    
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        tune_llm=TUNE_LLM,
        tune_visual=TUNE_VISUAL,
        tune_projector=TUNE_PROJECTOR,
        tune_diffusion_model=TUNE_DIFFUSION_MODEL,
    )
    
    # Set compute dtype to bfloat16 for efficiency
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    model.to(device)
    
    print(f"✅ Model loaded successfully!")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def create_training_args(output_dir: str, max_steps: int = 20, batch_size: int = 8):
    """
    Create training arguments for HuggingFace Trainer.
    
    Args:
        output_dir: Directory to save model checkpoints
        max_steps: Maximum training steps
        batch_size: Per-device batch size
        
    Returns:
        TrainingArguments instance
    """
    print(f"\n⚙️  Creating training configuration")
    print(f"   Output directory: {output_dir}")
    print(f"   Max steps: {max_steps}")
    print(f"   Batch size: {batch_size}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=8,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=1e-4,
        weight_decay=1e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=max_steps,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=8,
        report_to="wandb",
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )
    
    return training_args


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("🚀 LIBERO + Isaac-GR00T Custom Training")
    print("=" * 80)
    
    # Configuration
    DATASET_PATH = "./merged_libero_mask_depth_noops_lerobot_40"
    BASE_MODEL_PATH = "nvidia/GR00T-N1.5-3B"
    OUTPUT_DIR = "output/libero_groot_training"
    MAX_STEPS = 20000
    BATCH_SIZE = 8
    
    # Setup
    device = setup_device()
    
    # Load dataset
    dataset = load_dataset(DATASET_PATH)
    
    # Load model
    model = load_model(BASE_MODEL_PATH, device)
    
    # Create training arguments
    training_args = create_training_args(OUTPUT_DIR, MAX_STEPS, BATCH_SIZE)
    
    # Create trainer
    print(f"\n🏋️  Initializing trainer...")
    experiment = TrainRunner(
        train_dataset=dataset,
        model=model,
        training_args=training_args,
    )
    
    # Start training
    print(f"\n🎯 Starting training...")
    print("=" * 80)
    
    try:
        experiment.train()
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ Training failed with error:")
        print(f"   {str(e)}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
