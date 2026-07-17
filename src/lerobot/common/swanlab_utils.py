#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import numbers
from pathlib import Path
from typing import TYPE_CHECKING

from termcolor import colored

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


class SwanLabLogger:
    """SwanLab experiment logger for scalar training metrics."""

    def __init__(self, cfg: "TrainPipelineConfig"):
        try:
            import swanlab
        except ImportError as e:
            raise ImportError(
                "SwanLab logging requires the 'swanlab' package. Install it with "
                "`pip install swanlab` in the active environment."
            ) from e

        log_dir = Path(cfg.swanlab.log_dir)
        if not log_dir.is_absolute():
            log_dir = cfg.output_dir / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        mode = "online" if cfg.swanlab.mode == "cloud" else cfg.swanlab.mode
        config = json.loads(json.dumps(cfg.to_dict(), default=str))
        self.log_dir = log_dir
        self._swanlab = swanlab
        self._run = swanlab.init(
            project=cfg.swanlab.project,
            workspace=cfg.swanlab.workspace,
            experiment_name=cfg.swanlab.experiment_name or cfg.job_name,
            description=cfg.swanlab.description,
            mode=mode,
            logdir=str(log_dir),
            config=config,
        )
        logging.info(
            colored(
                f"SwanLab logs will be saved to {log_dir} (mode={mode})",
                "blue",
                attrs=["bold"],
            )
        )

    def log_dict(self, metrics: dict, step: int, mode: str = "train") -> None:
        scalar_metrics = {}
        for key, value in metrics.items():
            scalar = self._to_scalar(value)
            if scalar is not None:
                scalar_metrics[f"{mode}/{key}"] = scalar
        if scalar_metrics:
            self._swanlab.log(scalar_metrics, step=step)

    @staticmethod
    def _to_scalar(value) -> float | int | bool | None:
        if isinstance(value, bool | numbers.Real):
            return value
        if hasattr(value, "numel") and value.numel() == 1:
            return value.item()
        if hasattr(value, "size") and value.size == 1:
            return value.item()
        return None

    def finish(self) -> None:
        self._swanlab.finish()
