#!/usr/bin/env python

"""Offline ACT checkpoint comparison on LeRobotDataset frames."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--checkpoint-glob", type=str, default="")
    parser.add_argument("--model-name", default="act")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/uqi_act_offline"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-all", action="store_true")
    return parser.parse_args()


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoints = list(args.checkpoint)
    if args.checkpoint_glob:
        checkpoints.extend(Path().glob(args.checkpoint_glob))
    checkpoints = sorted({path.resolve() for path in checkpoints})
    if not checkpoints:
        raise ValueError("Pass at least one --checkpoint or --checkpoint-glob.")
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    return checkpoints


def prepare_sample(sample: dict, camera_keys: list[str]) -> dict:
    sample = dict(sample)
    for camera_key in camera_keys:
        if camera_key in sample and sample[camera_key].dtype == torch.uint8:
            sample[camera_key] = sample[camera_key].to(dtype=torch.float32) / 255.0
    return sample


def first_action(action: torch.Tensor) -> torch.Tensor:
    if action.ndim == 2:
        return action[0]
    return action


def infer_checkpoint_step(checkpoint: Path) -> str:
    if checkpoint.name == "pretrained_model":
        return checkpoint.parent.name
    return checkpoint.name


def infer_run_name(checkpoint: Path) -> str:
    parts = checkpoint.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx > 0:
            return parts[idx - 1]
    return checkpoint.parent.name


def short_name(checkpoint: Path, model_name: str, run_name: str | None) -> str:
    checkpoint_step = infer_checkpoint_step(checkpoint)
    if run_name:
        return f"{model_name}_{run_name}_{checkpoint_step}"
    return f"{model_name}_{checkpoint_step}"


def evaluate_checkpoint(
    checkpoint: Path,
    dataset: LeRobotDataset,
    frame_indices: list[int],
    device: torch.device,
    model_name: str,
    run_name: str | None,
) -> dict:
    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(checkpoint)
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    preds = []
    gts = []
    raw_preds = []

    with torch.no_grad():
        for frame_index in frame_indices:
            sample = prepare_sample(dataset[frame_index], dataset.meta.camera_keys)
            batch = preprocessor(sample)
            raw_chunk = policy.predict_action_chunk(batch)
            raw_action = raw_chunk[:, 0]
            pred_action = postprocessor(raw_action)
            gt_action = first_action(sample[ACTION])

            raw_preds.append(raw_action.detach().cpu().numpy().reshape(-1))
            preds.append(pred_action.detach().cpu().numpy().reshape(-1))
            gts.append(gt_action.detach().cpu().numpy().reshape(-1))

    pred = np.stack(preds)
    gt = np.stack(gts)
    raw_pred = np.stack(raw_preds)
    abs_err = np.abs(pred - gt)

    inferred_run_name = run_name or infer_run_name(checkpoint)
    checkpoint_step = infer_checkpoint_step(checkpoint)
    return {
        "checkpoint": str(checkpoint),
        "model": model_name,
        "run": inferred_run_name,
        "checkpoint_step": checkpoint_step,
        "name": short_name(checkpoint, model_name, inferred_run_name),
        "frame_indices": np.asarray(frame_indices),
        "pred": pred,
        "gt": gt,
        "raw_pred": raw_pred,
        "abs_err": abs_err,
        "overall_mae": float(abs_err.mean()),
        "arm_mae": float(abs_err[:, :7].mean()),
        "hand_mae": float(abs_err[:, 7:].mean()),
        "per_dim_mae": abs_err.mean(axis=0),
    }


def write_summary(results: list[dict], output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rank", "model", "run", "checkpoint_step", "name", "overall_mae", "arm_mae", "hand_mae", "checkpoint"]
        )
        for rank, result in enumerate(sorted(results, key=lambda item: item["overall_mae"]), start=1):
            writer.writerow(
                [
                    rank,
                    result["model"],
                    result["run"],
                    result["checkpoint_step"],
                    result["name"],
                    f"{result['overall_mae']:.8f}",
                    f"{result['arm_mae']:.8f}",
                    f"{result['hand_mae']:.8f}",
                    result["checkpoint"],
                ]
            )


def write_detail(result: dict, output_dir: Path) -> None:
    detail_dir = output_dir / result["name"]
    detail_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        detail_dir / "pred_gt.npz",
        frame_indices=result["frame_indices"],
        pred=result["pred"],
        gt=result["gt"],
        raw_pred=result["raw_pred"],
        abs_err=result["abs_err"],
        per_dim_mae=result["per_dim_mae"],
    )

    per_dim_path = detail_dir / "per_dim_mae.csv"
    with per_dim_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dim", "group", "mae"])
        for dim, mae in enumerate(result["per_dim_mae"]):
            writer.writerow([dim, "arm" if dim < 7 else "hand", f"{mae:.8f}"])


def plot_result(result: dict, output_dir: Path, plt) -> Path:
    x = result["frame_indices"]
    pred = result["pred"]
    gt = result["gt"]

    fig, axes = plt.subplots(13, 1, figsize=(14, 24), sharex=True)
    for dim, axis in enumerate(axes):
        axis.plot(x, gt[:, dim], label="gt", linewidth=1.2)
        axis.plot(x, pred[:, dim], label="pred", linewidth=1.0)
        axis.set_ylabel(f"a{dim:02d}")
        axis.grid(True, alpha=0.25)
        if dim == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("dataset frame index")
    fig.suptitle(f"Pred vs gt: {result['name']}", y=0.995)
    fig.tight_layout()
    output_path = output_dir / f"{result['name']}_pred_vs_gt.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def maybe_plot_results(results: list[dict], output_dir: Path, plot_all: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped plots. Install with: pip install matplotlib")
        return

    selected = sorted(results, key=lambda item: item["overall_mae"]) if plot_all else [
        min(results, key=lambda item: item["overall_mae"])
    ]
    for result in selected:
        print(f"plot: {plot_result(result, output_dir, plt)}")


def main() -> None:
    args = parse_args()
    checkpoints = resolve_checkpoints(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    first_metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)

    first_policy_class = get_policy_class("act")
    first_policy = first_policy_class.from_pretrained(checkpoints[0])
    delta_timestamps = resolve_delta_timestamps(first_policy.config, first_metadata)
    del first_policy

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )

    stop = min(len(dataset), args.start_frame + args.num_frames * args.stride)
    frame_indices = list(range(args.start_frame, stop, args.stride))[: args.num_frames]
    if not frame_indices:
        raise ValueError("No frame indices selected.")

    print(f"Evaluating model={args.model_name} with {len(checkpoints)} checkpoints on {len(frame_indices)} frames")
    print(f"frame range: {frame_indices[0]}..{frame_indices[-1]} stride={args.stride}")
    for checkpoint in checkpoints:
        run_name = args.run_name or infer_run_name(checkpoint)
        print(
            f"checkpoint: model={args.model_name} run={run_name} "
            f"step={infer_checkpoint_step(checkpoint)} -> {checkpoint}"
        )

    results = []
    for checkpoint in checkpoints:
        result = evaluate_checkpoint(checkpoint, dataset, frame_indices, device, args.model_name, args.run_name)
        results.append(result)
        write_detail(result, args.output_dir)
        print(
            f"{result['name']}: overall_mae={result['overall_mae']:.6f} "
            f"arm_mae={result['arm_mae']:.6f} hand_mae={result['hand_mae']:.6f}"
        )

    write_summary(results, args.output_dir)
    if args.plot or args.plot_all:
        maybe_plot_results(results, args.output_dir, plot_all=args.plot_all)

    print(f"summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
