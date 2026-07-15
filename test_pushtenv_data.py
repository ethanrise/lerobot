from pathlib import Path

import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import gymnasium as gym
import numpy as np

from lerobot.envs.configs import PushtEnv
from lerobot.utils.io_utils import write_video

num_steps = 60
# 0.0 = no-op (target stays at the pusher's current position).
# e.g. 30.0 for a bit of jitter/motion from random actions (in pixels).
random_action_scale = 0.0
out_dir = Path("outputs/debug/env_preview")

# PushT has no language instruction: it predates VLA-style instruction-following
# benchmarks, it's a single fixed task ("push the T block onto the target").
cfg = PushtEnv()
print("=== TASK ===")
print("gym_id:", cfg.gym_id)
print("task: push the T-shaped block onto the fixed target zone (no language instruction)")

env = gym.make(cfg.gym_id, disable_env_checker=cfg.disable_env_checker, **cfg.gym_kwargs)

print("max_episode_steps:", cfg.episode_length)
print("action_space:", env.action_space)
print("observation_space:", env.observation_space)

observation, info = env.reset(seed=0)
print("\n=== STATE ===")
print("step 000: agent_pos=", np.round(observation["agent_pos"], 1))

frames = [env.render()]
for step in range(num_steps):
    if random_action_scale > 0:
        # Action is an absolute pixel-space target for the pusher, not a delta —
        # jitter it around the current position instead of sampling the full space.
        jitter = np.random.uniform(-random_action_scale, random_action_scale, size=2)
        action = (observation["agent_pos"] + jitter).astype(np.float32)
    else:
        # True no-op for this action convention: target = current position.
        action = observation["agent_pos"].astype(np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        print(f"\nEpisode ended at step {step} (terminated={terminated}, truncated={truncated})")
        break

print("step", f"{len(frames) - 1:03d}: agent_pos=", np.round(observation["agent_pos"], 1))
print("is_success:", info.get("is_success", terminated))

out_dir.mkdir(parents=True, exist_ok=True)
video_path = out_dir / "pusht.mp4"
write_video(str(video_path), frames, fps=env.unwrapped.metadata["render_fps"])

print("\n=== saved ===")
print("frames:", len(frames))
print("video:", video_path)
