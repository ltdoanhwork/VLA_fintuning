#!/usr/bin/env python3
"""
Training Script: LIBERO + Isaac-GR00T Integration
=================================================
Main training script using centralized configuration.

Usage:
    python train.py
    python train.py --preset quick_test
    python train.py --preset full_training

Author: ltdoanh
Date: 2025-11-24
"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Add Isaac-GR00T to path
sys.path.append('/home/serverai/ltdoanh/pi0_vggt/Isaac-GR00T')

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.experiment.runner import TrainRunner
from transformers import TrainingArguments

# Import configurations
from config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    LiberoDataConfig,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    QUICK_TEST_CONFIG,
    FULL_TRAINING_CONFIG,
    LOW_MEMORY_CONFIG,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LIBERO + Isaac-GR00T")
    
    # Preset configurations
    parser.add_argument(
        "--preset",
        type=str,
        choices=["default", "quick_test", "full_training", "low_memory"],
        default="default",
        help="Use a preset configuration"
    )
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, help="Path to LIBERO dataset")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, help="Base model path or HuggingFace ID")
    parser.add_argument("--gpu_id", type=int, help="GPU ID to use")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--max_steps", type=int, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--save_steps", type=int, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=float, help="Log every N steps")
    parser.add_argument("--report_to", type=str, choices=["wandb", "tensorboard", "none"], help="Logging backend")
    
    return parser.parse_args()


def get_configs(args):
    """Get configurations based on arguments."""
    # Select preset
    if args.preset == "quick_test":
        training_config = QUICK_TEST_CONFIG
    elif args.preset == "full_training":
        training_config = FULL_TRAINING_CONFIG
    elif args.preset == "low_memory":
        training_config = LOW_MEMORY_CONFIG
    else:
        training_config = DEFAULT_TRAINING_CONFIG
    
    # Override with command line arguments
    if args.output_dir:
        training_config.output_dir = args.output_dir
    if args.max_steps:
        training_config.max_steps = args.max_steps
    if args.batch_size:
        training_config.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    if args.save_steps:
        training_config.save_steps = args.save_steps
    if args.logging_steps:
        training_config.logging_steps = args.logging_steps
    if args.report_to:
        training_config.report_to = args.report_to
    
    # Dataset config
    dataset_config = DEFAULT_DATASET_CONFIG
    if args.dataset_path:
        dataset_config.dataset_path = args.dataset_path
    
    # Model config
    model_config = DEFAULT_MODEL_CONFIG
    if args.model_path:
        model_config.base_model_path = args.model_path
    if args.gpu_id is not None:
        model_config.gpu_id = args.gpu_id
    
    return dataset_config, model_config, training_config


def setup_device(model_config: ModelConfig):
    """Setup CUDA device."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device


def load_dataset(dataset_config: DatasetConfig):
    """Load LIBERO dataset with custom configuration."""
    print(f"\n📦 Loading dataset from: {dataset_config.dataset_path}")
    
    # Use LiberoDataConfig from config.py
    data_config = LiberoDataConfig()
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_config.dataset_path,
        modality_configs=modality_config,
        video_backend=dataset_config.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(dataset)}")
    
    return dataset


def load_model(model_config: ModelConfig, device: str):
    """Load GR00T model with custom fine-tuning configuration."""
    print(f"\n🤖 Loading GR00T model from: {model_config.base_model_path}")
    print(f"   Tune LLM: {model_config.tune_llm}")
    print(f"   Tune Visual: {model_config.tune_visual}")
    print(f"   Tune Projector: {model_config.tune_projector}")
    print(f"   Tune Diffusion: {model_config.tune_diffusion_model}")
    
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=model_config.base_model_path,
        tune_llm=model_config.tune_llm,
        tune_visual=model_config.tune_visual,
        tune_projector=model_config.tune_projector,
        tune_diffusion_model=model_config.tune_diffusion_model,
        tune_visual_layers=model_config.tune_visual_layers,
        tune_llm_layers=model_config.tune_llm_layers,
        lora_config={
            "r": model_config.lora_rank,
            "lora_alpha": model_config.lora_alpha,
            "lora_dropout": model_config.lora_dropout,
            "target_modules": model_config.lora_target_modules,
        } if model_config.use_lora else None,
    )
    
    model.compute_dtype = model_config.compute_dtype
    model.config.compute_dtype = model_config.compute_dtype
    model.to(device)
    
    print(f"✅ Model loaded successfully!")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def create_training_args(training_config: TrainingConfig):
    """Create training arguments from config."""
    print(f"\n⚙️  Creating training configuration")
    print(f"   Output directory: {training_config.output_dir}")
    print(f"   Max steps: {training_config.max_steps}")
    print(f"   Batch size: {training_config.per_device_train_batch_size}")
    
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=training_config.gradient_checkpointing,
        bf16=training_config.bf16,
        tf32=training_config.tf32,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        dataloader_num_workers=training_config.dataloader_num_workers,
        dataloader_pin_memory=training_config.dataloader_pin_memory,
        dataloader_persistent_workers=training_config.dataloader_persistent_workers,
        optim="adamw_torch",
        adam_beta1=training_config.adam_beta1,
        adam_beta2=training_config.adam_beta2,
        adam_epsilon=training_config.adam_epsilon,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type=training_config.lr_scheduler_type,
        logging_steps=training_config.logging_steps,
        num_train_epochs=training_config.num_train_epochs,
        max_steps=training_config.max_steps,
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        report_to=training_config.report_to,
        seed=training_config.seed,
        do_eval=training_config.do_eval,
        ddp_find_unused_parameters=training_config.ddp_find_unused_parameters,
        ddp_bucket_cap_mb=training_config.ddp_bucket_cap_mb,
        torch_compile_mode=training_config.torch_compile_mode,
    )
    
    return training_args


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("🚀 LIBERO + Isaac-GR00T Training")
    print("=" * 80)
    
    # Parse arguments
    args = parse_args()
    print(f"\n📋 Using preset: {args.preset}")
    
    # Get configurations
    dataset_config, model_config, training_config = get_configs(args)
    
    # Setup
    device = setup_device(model_config)
    
    # Load dataset
    dataset = load_dataset(dataset_config)
    
    # Load model
    model = load_model(model_config, device)
    
    # Create training arguments
    training_args = create_training_args(training_config)
    
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
        print(f"   Checkpoints saved to: {training_config.output_dir}")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ Training failed with error:")
        print(f"   {str(e)}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
