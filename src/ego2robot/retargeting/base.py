"""Interfaces and output types for retargeted robot task paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ego2robot.demos.schema import CanonicalDemo


@dataclass(frozen=True)
class RobotWaypoint:
    """A desired object location and optional action annotation in robot space."""

    timestamp_s: float
    object_position_xyz: tuple[float, float, float]
    task_phase: str | None = None
    gripper_state: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class RobotDemo:
    """Canonical demonstration mapped to robot-world object waypoints."""

    demo_id: str
    task: str
    source: str
    coordinate_frame: str
    waypoints: tuple[RobotWaypoint, ...]


class Retargeter(ABC):
    """Map a canonical human demonstration into a robot task representation."""

    @abstractmethod
    def retarget(
        self,
        demo: CanonicalDemo,
        robot_start_position: tuple[float, float, float],
        robot_goal_position: tuple[float, float, float],
    ) -> RobotDemo:
        """Return object waypoints anchored to the current robot task endpoints."""
