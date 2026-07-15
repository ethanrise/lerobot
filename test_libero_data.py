from pathlib import Path
from pprint import pprint

from PIL import Image
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

repo_id = ""
root = "datasets/libero"

# Set >1 to load a future action sequence per sample.
action_horizon = 16
state_history_offsets = [-0.2, -0.1, 0.0]
image_history_offsets = [-0.2, -0.1, 0.0]


def _tensor_to_pil_image(t: torch.Tensor) -> Image.Image:
    """Convert CHW tensor in [0,1] or uint8 to a PIL image."""
    x = t.detach().cpu()

    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor (C,H,W), got shape={tuple(x.shape)}")

    # CHW -> HWC
    if x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)

    if torch.is_floating_point(x):
        x = (x.clamp(0, 1) * 255.0).round().to(torch.uint8)
    else:
        x = x.to(torch.uint8)

    arr = x.numpy()
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


def save_sample_images(item: dict, camera_keys: list[str], out_dir: str | Path, prefix: str) -> list[Path]:
    """Save camera tensors from one sample to PNG files."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for cam in camera_keys:
        cam_tensor = item[cam]

        # With delta_timestamps, shape can be (T,C,H,W).
        if isinstance(cam_tensor, torch.Tensor) and cam_tensor.ndim == 4:
            for t_idx, frame in enumerate(cam_tensor):
                img = _tensor_to_pil_image(frame)
                save_path = out_path / f"{prefix}_{cam.replace('.', '_')}_t{t_idx:02d}.png"
                img.save(save_path)
                saved_paths.append(save_path)
        else:
            img = _tensor_to_pil_image(cam_tensor)
            save_path = out_path / f"{prefix}_{cam.replace('.', '_')}.png"
            img.save(save_path)
            saved_paths.append(save_path)

    return saved_paths


meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
print("=== META ===")
print("episodes:", meta.total_episodes)
print("frames:", meta.total_frames)
print("fps:", meta.fps)
print("robot_type:", meta.robot_type)
print("camera_keys:", meta.camera_keys)
print("video_keys:", meta.video_keys)

delta_timestamps = {
    "observation.state": state_history_offsets,
    # Predict a future action sequence instead of a single action.
    "action": [t / meta.fps for t in range(action_horizon)],
}
# Add image history for every camera key.
for cam_key in meta.camera_keys:
    delta_timestamps[cam_key] = image_history_offsets

ds = LeRobotDataset(
    repo_id=repo_id,
    root=root,
    episodes=[0],
    delta_timestamps=delta_timestamps,
)
print("\nlen(ds) =", len(ds))
print("action_horizon:", action_horizon)
print("state_history_offsets:", state_history_offsets)
print("image_history_offsets:", image_history_offsets)

item = ds[0]
print("\n=== item keys ===")
pprint(sorted(item.keys()))

for k in [
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "task",
    "observation.state",
    "action",
]:
    v = item[k]
    if isinstance(v, torch.Tensor):
        if v.ndim == 0:
            print(
                f"{k}: type={type(v).__name__}, shape={tuple(v.shape)}, dtype={v.dtype}, value={v.item()}"
            )
        else:
            values = v.detach().cpu().tolist()
            print(
                f"{k}: type={type(v).__name__}, shape={tuple(v.shape)}, dtype={v.dtype}, values={values}"
            )
    elif hasattr(v, "shape"):
        print(f"{k}: type={type(v).__name__}, shape={tuple(v.shape)}, dtype={getattr(v, 'dtype', None)}")
    else:
        print(f"{k}: type={type(v).__name__}, value={v}")

for cam in meta.camera_keys:
    v = item[cam]
    if isinstance(v, torch.Tensor):
        v_cpu = v.detach().cpu()
        print(
            f"{cam}: type={type(v).__name__}, shape={tuple(v.shape)}, dtype={v.dtype}, "
            f"min={float(v_cpu.min()):.4f}, max={float(v_cpu.max()):.4f}, mean={float(v_cpu.mean()):.4f}"
        )
    else:
        print(f"{cam}: type={type(v).__name__}, shape={tuple(v.shape)}, dtype={getattr(v, 'dtype', None)}")

sample_prefix = f"ep{int(item['episode_index'].item())}_idx{int(item['index'].item())}"
saved = save_sample_images(
    item=item,
    camera_keys=meta.camera_keys,
    out_dir="outputs/debug/libero_samples",
    prefix=sample_prefix,
)

print("\n=== saved images ===")
for p in saved:
    print(p)
