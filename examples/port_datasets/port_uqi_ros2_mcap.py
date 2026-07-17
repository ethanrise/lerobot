#!/usr/bin/env python

"""Helpers for porting UQI ROS 2 MCAP recordings to LeRobotDataset."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import av
import numpy as np


STATE_TOPIC = "/teleop/sync_state"


@dataclass(frozen=True)
class TimedFrames:
    timestamps_ns: np.ndarray
    frames: list[np.ndarray]


def timestamp_ns(timestamp) -> int:
    return int(timestamp.sec) * 1_000_000_000 + int(timestamp.nanosec)


def discover_episodes(raw_dir: Path) -> list[Path]:
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    episodes = [path for path in raw_dir.iterdir() if path.is_dir() and any(path.glob("*.mcap"))]
    return sorted(episodes, key=lambda path: path.name)


def decode_h264(encoded_video: list[tuple[int, bytes]]) -> TimedFrames:
    if not encoded_video:
        raise ValueError("No H264 packets to decode")

    encoded_video = sorted(encoded_video, key=lambda item: item[0])
    packet_timestamps = np.asarray([timestamp for timestamp, _ in encoded_video], dtype=np.int64)
    raw_stream = b"".join(packet for _, packet in encoded_video)

    frames: list[np.ndarray] = []
    with av.open(BytesIO(raw_stream), format="h264") as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))

    if not frames:
        raise RuntimeError("H264 stream decoded to zero frames")

    if len(frames) == len(packet_timestamps):
        frame_timestamps = packet_timestamps
    else:
        frame_timestamps = np.linspace(
            int(packet_timestamps[0]),
            int(packet_timestamps[-1]),
            num=len(frames),
            dtype=np.int64,
        )

    return TimedFrames(frame_timestamps, frames)
