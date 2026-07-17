#!/usr/bin/env python

"""Load an ACT checkpoint and compare one predicted action with dataset ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(args.checkpoint)
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

    print("Loading weights from local directory")
    print("frame index:", args.frame_index)
    print("raw action shape:", tuple(raw_action.shape))
    print("pred action shape:", tuple(pred_action.shape))
    print("raw action:", to_numpy(raw_action).reshape(-1))
    print("pred action:", to_numpy(pred_action).reshape(-1))
    print("gt action:", to_numpy(gt_action).reshape(-1))


if __name__ == "__main__":
    main()
