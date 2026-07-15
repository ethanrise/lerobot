import os
from pathlib import Path

# Must be set before mujoco/dm_control create any rendering context — there's
# no DISPLAY on a headless server, so the default GLFW backend fails.
os.environ.setdefault("MUJOCO_GL", "egl")

import gym_aloha  # noqa: F401  (registers gym_aloha/AlohaInsertion-v0, gym_aloha/AlohaTransferCube-v0)
import gymnasium as gym
import numpy as np

from lerobot.envs.configs import AlohaEnv
from lerobot.utils.io_utils import write_video

# AlohaInsertion-v0 (insert the peg into the socket) or AlohaTransferCube-v0 (hand the cube to the other arm)
task = "AlohaInsertion-v0"
num_steps = 60
# 0.0 = no-op (action stays at zero). e.g. 0.3 for a bit of jitter/motion from random actions.
random_action_scale = 0.0
out_dir = Path("outputs/debug/env_preview")

# Aloha has no language instruction either: each gym id IS a fixed single task
# (bimanual ALOHA, 14 DOF: 7 per arm), no instruction-following involved.
cfg = AlohaEnv(task=task)
print("=== TASK ===")
print("gym_id:", cfg.gym_id)

env = gym.make(cfg.gym_id, disable_env_checker=cfg.disable_env_checker, **cfg.gym_kwargs)

print("max_episode_steps:", cfg.episode_length)
print("action_space:", env.action_space)
print("observation_space:", env.observation_space)
print("camera(s):", list(env.observation_space["pixels"].spaces.keys()))

observation, info = env.reset(seed=0)
print("\n=== STATE ===")
print("step 000: agent_pos=", np.round(observation["agent_pos"], 3))

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

print("step", f"{len(frames) - 1:03d}: agent_pos=", np.round(observation["agent_pos"], 3))

out_dir.mkdir(parents=True, exist_ok=True)
video_path = out_dir / f"{task}.mp4"
write_video(str(video_path), frames, fps=env.unwrapped.metadata["render_fps"]//10)

print("\n=== saved ===")
print("frames:", len(frames))
print("video:", video_path)

# Release the EGL context explicitly, before interpreter shutdown starts tearing
# down modules in an unpredictable order (otherwise OffScreenViewer/GLContext's
# __del__ can fire after mujoco.egl's own teardown, raising a harmless
# "TypeError: 'NoneType' object is not callable").
env.close()
