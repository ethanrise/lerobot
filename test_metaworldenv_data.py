import os
from pathlib import Path

# Must be set before MuJoCo creates any rendering context — there's no DISPLAY
# on a headless server, so the default GLFW backend fails.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from lerobot.envs.metaworld import TASK_DESCRIPTIONS, MetaworldEnv
from lerobot.utils.io_utils import write_video

# Full task list: lerobot/envs/metaworld_config.json (50 tasks, e.g. "assembly-v3", "basketball-v3", ...)
task_name = "pick-place-v3"
num_steps = 60
# 0.0 = no-op (action stays at zero). e.g. 0.3 for a bit of jitter/motion from random actions.
random_action_scale = 0.0
out_dir = Path("outputs/debug/env_preview")


def print_state(observation: dict, prefix: str) -> None:
    print(f"{prefix} agent_pos={np.round(observation['agent_pos'], 3)}")


print(f"=== task: {task_name} ===")
print("language instruction:", TASK_DESCRIPTIONS[task_name])

# "pixels_agent_pos" (not the default "pixels") is what actually carries
# the 4-dim agent_pos state alongside the camera image.
env = MetaworldEnv(task=task_name, obs_type="pixels_agent_pos")

print("\n=== TASK ===")
print("task:", env.task)
print("max_episode_steps:", env._max_episode_steps)
print("action_space:", env.action_space)
print("observation_space:", env.observation_space)

observation, info = env.reset(seed=0)
print("\n=== STATE ===")
print_state(observation, "step 000:")

frames = [env.render()]
for step in range(num_steps):
    if random_action_scale > 0:
        action = (env.action_space.sample() * random_action_scale).astype(np.float32)
    else:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        print(f"\nEpisode ended at step {step} (terminated={terminated}, truncated={truncated})")
        break

print_state(observation, f"step {len(frames) - 1:03d}:")
print("is_success:", info["is_success"])

out_dir.mkdir(parents=True, exist_ok=True)
video_path = out_dir / f"metaworld_{task_name}.mp4"
write_video(str(video_path), frames, fps=env.metadata["render_fps"]//10)

print("\n=== saved ===")
print("frames:", len(frames))
print("video:", video_path)

# Release the EGL context explicitly, before interpreter shutdown starts tearing
# down modules in an unpredictable order (otherwise OffScreenViewer/GLContext's
# __del__ can fire after mujoco.egl's own teardown, raising a harmless
# "TypeError: 'NoneType' object is not callable").
env.close()
