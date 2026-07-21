#!/usr/bin/env python

"""Load a policy checkpoint and compare one predicted action with dataset ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default="local/uqi_teleop_data_0514")
    parser.add_argument("--dataset.root", dest="dataset_root", required=True, type=Path)
    parser.add_argument("--dataset.video_backend", dest="video_backend", default="torchcodec")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def get_action_names(dataset: LeRobotDataset, dim: int) -> list[str]:
    names = dataset.meta.features.get(ACTION, {}).get("names")
    if isinstance(names, dict):
        names = names.get("axes")
    if not names or len(names) != dim:
        names = [f"dim_{i}" for i in range(dim)]
    return names


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(args.checkpoint, config=config)
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    sample = dataset[args.frame_index]
    for camera_key in dataset.meta.camera_keys:
        if camera_key in sample and sample[camera_key].dtype == torch.uint8:
            sample[camera_key] = sample[camera_key].to(dtype=torch.float32) / 255.0
    batch = preprocessor(sample)

    with torch.no_grad():
        raw_action = policy.select_action(batch)
        pred_action = postprocessor(raw_action)

    gt_action = sample[ACTION]
    if gt_action.ndim == 2:
        gt_action = gt_action[0]

    gt_np = to_numpy(gt_action).reshape(-1)
    dim = gt_np.shape[-1]
    raw_np = to_numpy(raw_action).reshape(-1, dim)
    pred_np = to_numpy(pred_action).reshape(-1, dim)
    action_names = get_action_names(dataset, dim)

    print("Loading weights from local directory")
    print("frame index:", args.frame_index)
    print("raw action shape:", tuple(raw_action.shape))
    print("pred action shape:", tuple(pred_action.shape))
    if pred_np.shape[0] > 1:
        print(f"(dataset only has 1 ground-truth frame; comparing against predicted step 0 of {pred_np.shape[0]})")

    diff = pred_np[0] - gt_np
    header = f"{'dim':>3}  {'name':<28}{'gt':>10}{'pred[0]':>10}{'diff':>10}"
    print()
    print(header)
    print("-" * len(header))
    for i, name in enumerate(action_names):
        print(f"{i:>3}  {name:<28}{gt_np[i]:>10.4f}{pred_np[0, i]:>10.4f}{diff[i]:>10.4f}")
    print("-" * len(header))
    print(f"{'mean abs diff':>41}: {np.abs(diff).mean():.4f}")
    print(f"{'max abs diff':>41}: {np.abs(diff).max():.4f}")

    if pred_np.shape[0] > 1:
        print(f"\nfull predicted action sequence ({pred_np.shape[0]} steps):")
        for step, row in enumerate(pred_np):
            print(f"  step {step}: " + " ".join(f"{v:>8.4f}" for v in row))


if __name__ == "__main__":
    main()
