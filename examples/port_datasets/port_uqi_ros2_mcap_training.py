#!/usr/bin/env python

"""Build a lightweight 10 Hz right-arm LeRobotDataset from UQI ROS 2 MCAP bags."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from lerobot.datasets import LeRobotDataset

from port_uqi_ros2_mcap import (
    STATE_TOPIC,
    TimedFrames,
    decode_h264,
    discover_episodes,
    timestamp_ns,
)


FPS = 10
SOURCE_CAMERA_FPS = 30
PHASE_COUNT = 3
FRAME_PERIOD_NS = 1_000_000_000 // FPS
PHASE_PERIOD_NS = 1_000_000_000 // SOURCE_CAMERA_FPS
RIGHT_CAMERA_KEY = "observation.images.right"
RIGHT_CAMERA_TOPIC = "/robot/topic/rdap/h264_front_right_sensor_camera"
TASK = "用右手抓起满罐可口可乐并放下"
STATE_NAMES = [f"right_arm_joint_{i}.position" for i in range(7)] + [
    f"right_hand_joint_{i}.position" for i in range(6)
]


def resize_frames(timed_frames: TimedFrames, width: int, height: int) -> TimedFrames:
    frames = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in timed_frames.frames]
    return TimedFrames(timed_frames.timestamps_ns, frames)


def read_source_episode(mcap_path: Path, width: int, height: int) -> tuple[TimedFrames, np.ndarray, np.ndarray]:
    encoded_video: list[tuple[int, bytes]] = []
    timestamps: list[int] = []
    states: list[np.ndarray] = []

    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, _, message in reader.iter_decoded_messages(
            topics=[RIGHT_CAMERA_TOPIC, STATE_TOPIC]
        ):
            if channel.topic == RIGHT_CAMERA_TOPIC:
                encoded_video.append((timestamp_ns(message.timestamp), bytes(message.data)))
                continue

            upper = message.upper_states.states
            right_hand = message.right_hand_states.states
            if len(upper) < 7 or len(right_hand) != 6:
                raise ValueError(
                    f"Unexpected joint counts in {mcap_path}: upper={len(upper)}, right_hand={len(right_hand)}"
                )
            # The first seven upper-body motors are assumed to be the right arm.
            # Verify this ordering against the robot driver before deployment.
            state = [motor.q for motor in upper[:7]]
            state.extend(joint.cur_position for joint in right_hand)
            timestamps.append(timestamp_ns(message.header.stamp))
            states.append(np.asarray(state, dtype=np.float32))

    if not encoded_video or not states:
        raise RuntimeError(f"Missing right camera or state messages in {mcap_path}")

    camera = resize_frames(decode_h264(encoded_video), width, height)
    state_timestamps = np.asarray(timestamps, dtype=np.int64)
    state_values = np.stack(states)

    camera_order = np.argsort(camera.timestamps_ns)
    state_order = np.argsort(state_timestamps)
    return (
        TimedFrames(camera.timestamps_ns[camera_order], [camera.frames[i] for i in camera_order]),
        state_timestamps[state_order],
        state_values[state_order],
    )


def nearest_indices(sorted_timestamps: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(sorted_timestamps, query_timestamps)
    indices = np.clip(indices, 0, len(sorted_timestamps) - 1)
    previous = np.clip(indices - 1, 0, len(sorted_timestamps) - 1)
    choose_previous = np.abs(query_timestamps - sorted_timestamps[previous]) <= np.abs(
        sorted_timestamps[indices] - query_timestamps
    )
    return np.where(choose_previous, previous, indices)


def interpolate_states(
    state_timestamps: np.ndarray, state_values: np.ndarray, query_timestamps: np.ndarray
) -> np.ndarray:
    query = query_timestamps.astype(np.float64)
    source = state_timestamps.astype(np.float64)
    return np.stack(
        [np.interp(query, source, state_values[:, dim]) for dim in range(state_values.shape[1])],
        axis=1,
    ).astype(np.float32)


def make_phase(
    camera: TimedFrames,
    state_timestamps: np.ndarray,
    state_values: np.ndarray,
    phase_index: int,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    common_start = max(int(camera.timestamps_ns[0]), int(state_timestamps[0]))
    common_end = min(int(camera.timestamps_ns[-1]), int(state_timestamps[-1]))
    start = common_start + phase_index * PHASE_PERIOD_NS
    timeline = np.arange(start, common_end + 1, FRAME_PERIOD_NS, dtype=np.int64)
    if len(timeline) < 2:
        raise RuntimeError(f"Phase {phase_index} contains fewer than two frames")

    camera_indices = nearest_indices(camera.timestamps_ns, timeline)
    camera_errors_ms = np.abs(camera.timestamps_ns[camera_indices] - timeline) / 1e6
    max_camera_error_ms = float(camera_errors_ms.max())
    if max_camera_error_ms > 50:
        raise RuntimeError(
            f"Phase {phase_index} camera synchronization error is too large: {max_camera_error_ms:.1f} ms"
        )

    images = [camera.frames[index] for index in camera_indices]
    states = interpolate_states(state_timestamps, state_values, timeline)
    return images, states, max_camera_error_ms


def convert(
    raw_dir: Path,
    output_dir: Path,
    repo_id: str,
    width: int,
    height: int,
    max_source_episodes: int | None,
) -> None:
    source_episodes = discover_episodes(raw_dir)
    if max_source_episodes is not None:
        source_episodes = source_episodes[:max_source_episodes]
    if not source_episodes:
        raise FileNotFoundError(f"No MCAP recordings found under {raw_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Choose a new path or remove it explicitly."
        )

    features = {
        RIGHT_CAMERA_KEY: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "source_episode_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["source_episode_index"],
        },
        "phase_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["phase_index"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type="uqi_right_arm",
        fps=FPS,
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )

    saved_episodes = 0
    try:
        for source_index, episode_dir in enumerate(source_episodes):
            mcap_paths = sorted(episode_dir.glob("*.mcap"))
            if len(mcap_paths) != 1:
                raise ValueError(f"Expected one MCAP in {episode_dir}, found {len(mcap_paths)}")
            logging.info(
                "Source %d/%d: %s",
                source_index + 1,
                len(source_episodes),
                episode_dir.name,
            )
            camera, state_timestamps, state_values = read_source_episode(mcap_paths[0], width, height)
            task = TASK

            for phase_index in range(PHASE_COUNT):
                images, states, max_error_ms = make_phase(
                    camera, state_timestamps, state_values, phase_index
                )
                # The final state has no future target and is therefore omitted.
                for frame_index in range(len(states) - 1):
                    dataset.add_frame(
                        {
                            RIGHT_CAMERA_KEY: images[frame_index],
                            "observation.state": states[frame_index],
                            "action": states[frame_index + 1],
                            "source_episode_index": np.asarray([source_index], dtype=np.int64),
                            "phase_index": np.asarray([phase_index], dtype=np.int64),
                            "task": task,
                        }
                    )
                dataset.save_episode()
                saved_episodes += 1
                logging.info(
                    "Saved output episode %d: phase=%d frames=%d max_camera_error=%.1fms",
                    saved_episodes - 1,
                    phase_index,
                    len(states) - 1,
                    max_error_ms,
                )
    finally:
        dataset.finalize()

    logging.info(
        "Finished: %d source recordings -> %d LeRobot episodes",
        len(source_episodes),
        saved_episodes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/uqi_right_arm_10hz")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-source-episodes", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    convert(
        raw_dir=args.raw_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        repo_id=args.repo_id,
        width=args.width,
        height=args.height,
        max_source_episodes=args.max_source_episodes,
    )


if __name__ == "__main__":
    main()
