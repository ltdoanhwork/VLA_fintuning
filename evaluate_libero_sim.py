#!/usr/bin/env python3
"""
Simulation Evaluation Client for LIBERO
=======================================
Connects to the GR00T inference server and runs evaluation on LIBERO tasks.
Calculates success rate and saves rollout videos.

Usage:
    python evaluate_libero_sim.py --task-suite-name libero_spatial --num-trials-per-task 5
"""

import sys
import os
import tyro
import tqdm
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Add Isaac-GR00T to path
sys.path.append('/home/serverai/ltdoanh/pi0_vggt/Isaac-GR00T')
sys.path.append('./LIBERO')
from libero.libero import benchmark
from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from gr00t.eval.service import ExternalRobotInferenceClient

@dataclass
class SimConfig:
    """Configuration for simulation evaluation."""
    # Task suite name (e.g., libero_spatial, libero_object, libero_goal, libero_10, libero_90)
    task_suite_name: str = "libero_spatial"
    
    # Number of trials per task
    num_trials_per_task: int = 5
    
    # Server host
    host: str = "localhost"
    
    # Server port
    port: int = 5555
    
    # Number of steps to wait for objects to stabilize
    num_steps_wait: int = 10
    
    # Output directory for logs and videos
    output_dir: str = "eval_sim_results"
    
    # Headless mode (no GUI)
    headless: bool = True


class GR00TClientPolicy:
    """Client-side policy wrapper that communicates with the inference server."""
    
    def __init__(self, host="localhost", port=5555):
        self.client = ExternalRobotInferenceClient(host=host, port=port)
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    def get_action(self, obs_dict, lang: str):
        """Get action from server given observation and language instruction."""
        # Process observation to match GR00T format
        processed_obs = self._process_observation(obs_dict, lang)
        
        # Query server
        try:
            action_chunk = self.client.get_action(processed_obs)
        except Exception as e:
            print(f"Error querying server: {e}")
            # Return no-op action on failure
            return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            
        # Convert to LIBERO format
        return self._convert_to_libero_action(action_chunk, 0)

    def _process_observation(self, obs, lang: str):
        """Convert Libero observation to GR00T format."""
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        return new_obs

    def _convert_to_libero_action(self, action_chunk: dict, idx: int = 0) -> np.ndarray:
        """Convert GR00T action chunk to Libero format."""
        # Check if action keys exist in response
        if "action.x" not in action_chunk:
             # Handle case where server might return 'actions' key or similar if structure differs
             # But based on service.py, it returns what the model returns.
             # If model returns dict with 'action.x', etc., we are good.
             pass

        action_components = []
        for key in self.action_keys:
            # Handle both scalar and array returns
            val = action_chunk.get(f"action.{key}")
            if val is None:
                raise ValueError(f"Missing key action.{key} in server response")
            
            # Extract specific time step
            if hasattr(val, "shape") and len(val.shape) > 0:
                val = val[idx]
            
            action_components.append(np.atleast_1d(val)[0])
            
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        return action_array


def main(cfg: SimConfig):
    print("=" * 80)
    print("🎯 LIBERO Simulation Evaluation")
    print("=" * 80)
    print(f"Task Suite: {cfg.task_suite_name}")
    print(f"Trials/Task: {cfg.num_trials_per_task}")
    print(f"Server: {cfg.host}:{cfg.port}")
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, f"eval_{cfg.task_suite_name}.log")
    log_file = open(log_path, "w")
    
    # Initialize Policy Client
    print("\n🔌 Connecting to inference server...")
    try:
        policy = GR00TClientPolicy(host=cfg.host, port=cfg.port)
        # Simple ping to check connection
        policy.client.ping()
        print("✅ Connected to server!")
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print("   Make sure 'serve_custom_policy.py' is running!")
        sys.exit(1)

    # Initialize LIBERO Benchmark
    print("\n📦 Initializing LIBERO benchmark...")
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        num_tasks = task_suite.n_tasks
        print(f"✅ Loaded {num_tasks} tasks from {cfg.task_suite_name}")
    except Exception as e:
        print(f"❌ Failed to load benchmark: {e}")
        sys.exit(1)

    # Evaluation Loop
    total_episodes = 0
    total_successes = 0
    
    print("\n🚀 Starting evaluation...")
    
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_desc = get_libero_env(task, resolution=256)
        
        print(f"\nTask {task_id+1}/{num_tasks}: {task_desc}")
        log_file.write(f"\nTask {task_id+1}: {task_desc}\n")
        
        task_successes = 0
        
        # Determine max steps based on suite
        if cfg.task_suite_name == "libero_spatial": max_steps = 220
        elif cfg.task_suite_name == "libero_object": max_steps = 280
        elif cfg.task_suite_name == "libero_goal": max_steps = 600
        elif cfg.task_suite_name == "libero_10": max_steps = 1000
        elif cfg.task_suite_name == "libero_90": max_steps = 400
        else: max_steps = 300
        
        for trial in range(cfg.num_trials_per_task):
            env.reset()
            # Set initial state
            init_state_idx = trial % len(initial_states)
            env.set_init_state(initial_states[init_state_idx])
            
            # Rollout
            t = 0
            done = False
            top_view_frames = []
            wrist_view_frames = []
            
            print(f"   Trial {trial+1}/{cfg.num_trials_per_task}...", end="", flush=True)
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # Wait for stabilization
                    if t < cfg.num_steps_wait:
                        obs, _, _, _ = env.step(get_libero_dummy_action())
                        t += 1
                        continue
                    
                    # Get images for video
                    img, wrist_img = get_libero_image(obs)
                    top_view_frames.append(img)
                    wrist_view_frames.append(wrist_img)
                    
                    # Get action
                    action = policy.get_action(obs, task.language)
                    
                    # Step environment
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        break
                    t += 1
                    
                except Exception as e:
                    print(f" Error: {e}")
                    break
            
            # Record result
            success = done
            if success:
                task_successes += 1
                total_successes += 1
                print(" ✅ Success")
            else:
                print(" ❌ Failure")
            
            total_episodes += 1
            
            # Save video
            video_path = save_rollout_video(
                top_view_frames,
                wrist_view_frames,
                idx=total_episodes,
                success=success,
                task_description=task_desc,
                log_file=log_file
            )
            
        # Task Summary
        task_rate = task_successes / cfg.num_trials_per_task
        print(f"   Task Success Rate: {task_rate:.2%}")
        log_file.write(f"   Task Success Rate: {task_rate:.2%}\n")
        log_file.flush()
        
        env.close()

    # Final Summary
    overall_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    print("\n" + "=" * 80)
    print("📊 Evaluation Summary")
    print("=" * 80)
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Successes: {total_successes}")
    print(f"Overall Success Rate: {overall_rate:.2%}")
    print(f"Logs saved to: {log_path}")
    print("=" * 80)
    
    log_file.write("\nFinal Summary\n")
    log_file.write(f"Total Episodes: {total_episodes}\n")
    log_file.write(f"Total Successes: {total_successes}\n")
    log_file.write(f"Overall Success Rate: {overall_rate:.2%}\n")
    log_file.close()

if __name__ == "__main__":
    cfg = tyro.cli(SimConfig)
    main(cfg)
