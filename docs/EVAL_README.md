# Offline Evaluation for Custom GR00T Model

This guide explains how to use the `evaluate_groot_style.py` script to evaluate your fine-tuned GR00T model on the LIBERO dataset.

## Overview

The evaluation script performs **offline evaluation**, meaning it compares the model's predicted actions against the ground truth actions recorded in the dataset. It does not run a physics simulation.

**Metrics Calculated:**
- **MSE (Mean Squared Error)**: The average squared difference between predicted and ground truth actions.
- **Per-Dimension MSE**: MSE broken down by action component (X, Y, Z, Roll, Pitch, Yaw, Gripper).

**Visualizations:**
- **Action Trajectory Plots**: Graphs showing the predicted action trajectory vs. the ground truth trajectory over the prediction horizon (16 steps).

## Usage

### Basic Evaluation
Run the script with the path to your model checkpoint:

```bash
python evaluate_groot_style.py --checkpoint output/libero_groot_training/checkpoint-20
```

### Generate Plots
To save visualization plots of the action trajectories, add the `--plot` flag:

```bash
python evaluate_groot_style.py --checkpoint output/libero_groot_training/checkpoint-20 --plot
```
Plots will be saved to the `eval_results` directory by default.

### Evaluate Specific Number of Samples
To evaluate a specific number of random samples (e.g., for a quick check), use `--num-samples`:

```bash
python evaluate_groot_style.py --checkpoint output/libero_groot_training/checkpoint-20 --num-samples 50
```
Set `--num-samples 0` to evaluate the entire dataset.

### Custom Dataset Path
If your dataset is in a different location:

```bash
python evaluate_groot_style.py --dataset-path /path/to/your/dataset
```

## Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--checkpoint` | `output/libero_groot_training/checkpoint-20` | Path to the model checkpoint directory. |
| `--dataset-path` | `./merged_libero_mask_depth_noops_lerobot_40` | Path to the LIBERO dataset. |
| `--num-samples` | `10` | Number of samples to evaluate. Set to `0` for all. |
| `--plot` | `False` | Enable plotting of action trajectories. |
| `--output-dir` | `eval_results` | Directory to save generated plots. |
| `--denoising-steps` | `8` | Number of diffusion denoising steps (inference speed vs quality trade-off). |
| `--device` | `cuda` (if available) | Device to run evaluation on (`cuda` or `cpu`). |

## Output Interpretation

The script prints a summary table of MSE values.

- **MSE < 0.001**: Excellent. The model is predicting actions very close to the expert.
- **MSE < 0.01**: Good. The model captures the general trend well.
- **MSE > 0.1**: Needs Improvement. The model predictions are significantly deviating from the ground truth.

**Note on Gripper**: The gripper action is often binary (open/close) or has a specific range. High MSE in the gripper dimension might indicate the model is struggling to learn the precise timing of grasp/release.
