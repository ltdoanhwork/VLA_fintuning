#!/usr/bin/env python3
"""
Test Script: Verify LIBERO + Isaac-GR00T Setup
===============================================
This script tests if the environment is properly configured.

Author: ltdoanh
Date: 2025-11-24
"""

import sys
import os

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_python_version():
    """Test Python version."""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("✅ Python version OK")
        return True
    else:
        print("❌ Python 3.10+ required")
        return False

def test_imports():
    """Test required imports."""
    print_section("Testing Imports")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("numpy", "NumPy"),
    ]
    
    all_ok = True
    for package, name in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: NOT FOUND")
            all_ok = False
    
    return all_ok

def test_cuda():
    """Test CUDA availability."""
    print_section("CUDA Configuration")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"      Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"      Compute Capability: {props.major}.{props.minor}")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def test_isaac_groot():
    """Test Isaac-GR00T import."""
    print_section("Isaac-GR00T")
    
    sys.path.append('/home/serverai/ltdoanh/pi0_vggt/Isaac-GR00T')
    
    try:
        from gr00t.model.gr00t_n1 import GR00T_N1_5
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.experiment.runner import TrainRunner
        print("✅ Isaac-GR00T imports OK")
        return True
    except ImportError as e:
        print(f"❌ Isaac-GR00T import failed: {e}")
        return False

def test_dataset():
    """Test dataset path."""
    print_section("Dataset")
    
    dataset_path = "./merged_libero_mask_depth_noops_lerobot_40"
    
    if os.path.exists(dataset_path):
        print(f"✅ Dataset found at: {dataset_path}")
        
        # Check for key files
        metadata_path = os.path.join(dataset_path, "meta_data")
        if os.path.exists(metadata_path):
            print(f"   ✅ Metadata directory exists")
        else:
            print(f"   ⚠️  Metadata directory not found")
        
        return True
    else:
        print(f"❌ Dataset not found at: {dataset_path}")
        print(f"   Please update the path in config_training.py")
        return False

def test_config_files():
    """Test if config files exist."""
    print_section("Configuration Files")
    
    files = [
        "config_training.py",
        "train_libero_groot.py",
        "train_libero_groot_v2.py",
        "run_training.sh",
    ]
    
    all_ok = True
    for file in files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} not found")
            all_ok = False
    
    return all_ok

def test_wandb():
    """Test WandB configuration."""
    print_section("Weights & Biases (WandB)")
    
    try:
        import wandb
        print(f"✅ WandB installed: {wandb.__version__}")
        
        # Check if logged in
        try:
            api = wandb.Api()
            print(f"✅ WandB logged in")
            return True
        except Exception:
            print(f"⚠️  WandB not logged in (run: wandb login)")
            return True  # Not critical
    except ImportError:
        print(f"⚠️  WandB not installed (optional)")
        return True  # Not critical

def main():
    """Run all tests."""
    print("=" * 80)
    print("  LIBERO + Isaac-GR00T Setup Verification")
    print("=" * 80)
    
    results = {
        "Python Version": test_python_version(),
        "Package Imports": test_imports(),
        "CUDA": test_cuda(),
        "Isaac-GR00T": test_isaac_groot(),
        "Dataset": test_dataset(),
        "Config Files": test_config_files(),
        "WandB": test_wandb(),
    }
    
    # Summary
    print_section("Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to train!")
        print("\nNext steps:")
        print("  1. Run: ./run_training.sh")
        print("  2. Or: python train_libero_groot_v2.py --preset quick_test")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
