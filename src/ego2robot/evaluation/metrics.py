"""Transparent task and trajectory metrics for one manipulation episode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    """Outcome and trajectory summary for a single seeded rollout."""

    success: bool
    episode_return: float
    final_object_target_distance_m: float
    object_trajectory_length_m: float
    steps: int

    def to_dict(self) -> dict[str, bool | float | int]:
        """Return CSV-compatible scalar fields."""
        return asdict(self)


def measure_episode(
    object_positions: Sequence[Sequence[float]],
    target_position: Sequence[float],
    episode_return: float,
    success: bool,
) -> EpisodeMetrics:
    """Calculate path length and final 3D distance from observed positions."""
    positions = np.asarray(object_positions, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("object_positions must be a non-empty sequence of 3D points")
    if target.shape != (3,):
        raise ValueError("target_position must be a 3D point")
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    final_distance = float(np.linalg.norm(positions[-1] - target))
    return EpisodeMetrics(
        success=bool(success),
        episode_return=float(episode_return),
        final_object_target_distance_m=final_distance,
        object_trajectory_length_m=path_length,
        steps=len(positions) - 1,
    )
