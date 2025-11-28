# Simulation Evaluation for Custom GR00T Model

This guide explains how to evaluate your fine-tuned GR00T model on the LIBERO simulation benchmark using the Isaac-GR00T client-server pattern.

## Overview

The evaluation consists of two parts running simultaneously:
1.  **Inference Server (`serve_custom_policy.py`)**: Hosts the model on the GPU and listens for action requests.
2.  **Simulation Client (`evaluate_libero_sim.py`)**: Runs the LIBERO simulation loop, sends observations to the server, and executes actions.

**Metrics:**
- **Success Rate**: Percentage of successful task completions.
- **Rollout Videos**: MP4 videos of each evaluation episode saved to `eval_sim_results/rollouts`.

## Prerequisites

- **LIBERO**: Ensure the `libero` package is installed.
- **Robosuite**: Ensure `robosuite` is installed.
- **Dataset**: The LIBERO dataset (for task definitions).

## Usage

### 1. Start the Inference Server
Open a terminal and run the server. Specify your model checkpoint path.

```bash
python serve_custom_policy.py --checkpoint output/libero_groot_training/checkpoint-20 --port 5555
```
Wait until you see `Server is ready and listening on ...`.

### 2. Run the Simulation Client
Open a **separate terminal** and run the client. Specify the task suite you want to evaluate.

```bash
python evaluate_libero_sim.py --task-suite-name libero_spatial --num-trials-per-task 5
```

### Arguments

#### Server (`serve_custom_policy.py`)
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--checkpoint` | `output/libero_groot_training/checkpoint-20` | Path to model checkpoint. |
| `--port` | `5555` | Port to listen on. |
| `--device` | `cuda` | Device to run model on. |

#### Client (`evaluate_libero_sim.py`)
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--task-suite-name` | `libero_spatial` | Task suite to evaluate (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`). |
| `--num-trials-per-task` | `5` | Number of evaluation episodes per task. |
| `--host` | `localhost` | Server hostname. |
| `--port` | `5555` | Server port. |
| `--output-dir` | `eval_sim_results` | Directory to save logs and videos. |
| `--headless` | `True` | Run without GUI (faster). Set to `False` to see the simulation window. |

## Output

Results are saved in `eval_sim_results/`:
- `eval_libero_spatial.log`: Detailed logs of success/failure.
- `rollouts/`: MP4 videos of every episode.

## Troubleshooting

- **Connection Refused**: Ensure the server is running and the port matches.
- **Missing Environment**: If you get `ModuleNotFoundError: No module named 'libero'`, ensure LIBERO is installed in your environment.
