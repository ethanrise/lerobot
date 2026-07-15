import os
from pathlib import Path

# Must be set before MuJoCo creates any rendering context — there's no DISPLAY
# on a headless server, so the default GLFW backend fails.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from libero.libero import benchmark

from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action
from lerobot.utils.io_utils import write_video

# Pick any suite: libero_spatial, libero_object, libero_goal, libero_10, libero_90
task_suite_name = "libero_object"
task_id = 0
num_steps = 60
# 0.0 = no-op (camera stays put, just look at the static scene).
# e.g. 0.3 for a bit of jitter/motion from random actions.
random_action_scale = 0.0
out_dir = Path("outputs/debug/env_preview")


def obs_to_frame(observation: dict) -> np.ndarray:
    """Flip both cameras like LiberoEnv.render() does, then stack them side by side."""
    images = [img[::-1, ::-1] for img in observation["pixels"].values()]
    return np.concatenate(images, axis=1)


def print_robot_state(observation: dict, prefix: str) -> None:
    state = observation["robot_state"]
    eef, gripper, joints = state["eef"], state["gripper"], state["joints"]
    print(
        f"{prefix} eef_pos={np.round(eef['pos'], 3)} "
        f"gripper_qpos={np.round(gripper['qpos'], 3)} "
        f"joint_pos={np.round(joints['pos'], 3)}"
    )


bench = benchmark.get_benchmark_dict()
task_suite = bench[task_suite_name]()

print(f"=== {task_suite_name} ({task_suite.n_tasks} tasks) ===")
for i in range(task_suite.n_tasks):
    marker = ">" if i == task_id else " "
    print(f"{marker} [{i}] {task_suite.get_task(i).language}")

# "pixels_agent_pos" (not the default "pixels") is what actually carries
# eef/gripper/joint state alongside the two camera images.
env = LiberoEnv(
    task_suite=task_suite,
    task_id=task_id,
    task_suite_name=task_suite_name,
    obs_type="pixels_agent_pos",
)

print("\n=== TASK ===")
print("task_name:", env.task)
print("language instruction:", env.task_description)
print("bddl_file:", env._task_bddl_file)
print("max_episode_steps:", env._max_episode_steps)
print("action_space:", env.action_space)
print("observation_space:", env.observation_space)

observation, info = env.reset(seed=0)
print("\n=== STATE ===")
print_robot_state(observation, "step 000:")

frames = [obs_to_frame(observation)]
for step in range(num_steps):
    if random_action_scale > 0:
        action = (env.action_space.sample() * random_action_scale).astype(np.float32)
    else:
        action = np.array(get_libero_dummy_action(), dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(obs_to_frame(observation))
    if terminated or truncated:
        print(f"\nEpisode ended at step {step} (terminated={terminated}, truncated={truncated})")
        break

print_robot_state(observation, f"step {len(frames) - 1:03d}:")
print("is_success:", info["is_success"])

out_dir.mkdir(parents=True, exist_ok=True)
video_path = out_dir / f"{task_suite_name}_task{task_id}.mp4"
write_video(str(video_path), frames, fps=env.metadata["render_fps"])

print("\n=== saved ===")
print("frames:", len(frames))
print("video:", video_path, "(left=agentview / image, right=wrist / image2)")
