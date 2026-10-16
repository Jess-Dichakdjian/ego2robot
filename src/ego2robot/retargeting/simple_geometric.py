"""Endpoint-normalized 2D image-path to 3D robot-path retargeting."""

from __future__ import annotations

import math

from ego2robot.demos.schema import CanonicalDemo
from ego2robot.retargeting.base import Retargeter, RobotDemo, RobotWaypoint


class SimpleGeometricRetargeter(Retargeter):
    """Map a clicked object path between known robot start and goal positions.

    The transform preserves progress along the human start-to-goal direction
    and scales lateral deviations to the robot route length. It maps object
    locations, not human hand poses or robot end-effector poses.
    """

    def __init__(self, lift_height: float = 0.12) -> None:
        if not math.isfinite(lift_height) or lift_height <= 0.0:
            raise ValueError("lift_height must be finite and positive")
        self.lift_height = float(lift_height)

    def retarget(
        self,
        demo: CanonicalDemo,
        robot_start_position: tuple[float, float, float],
        robot_goal_position: tuple[float, float, float],
    ) -> RobotDemo:
        """Retarget 2D image keyframes using this episode's robot endpoints."""
        if demo.coordinate_frame != "image_normalized":
            raise ValueError("SimpleGeometricRetargeter expects image_normalized input")
        start = self._position3("robot_start_position", robot_start_position)
        goal = self._position3("robot_goal_position", robot_goal_position)

        source_frames = [
            frame for frame in demo.frames if frame.object_position is not None
        ]
        if len(source_frames) < 2:
            raise ValueError("at least two object-position keyframes are required")
        first = source_frames[0].object_position
        last = source_frames[-1].object_position
        assert first is not None and last is not None
        human_dx = last[0] - first[0]
        human_dy = last[1] - first[1]
        human_length_sq = human_dx * human_dx + human_dy * human_dy
        if human_length_sq <= 1e-12:
            raise ValueError("human start and goal positions must differ")

        robot_delta = tuple(goal[i] - start[i] for i in range(3))
        robot_length = math.sqrt(sum(value * value for value in robot_delta))
        horizontal_length = math.hypot(robot_delta[0], robot_delta[1])
        if horizontal_length > 1e-12:
            lateral_axis = (
                -robot_delta[1] / horizontal_length,
                robot_delta[0] / horizontal_length,
                0.0,
            )
        else:
            lateral_axis = (1.0, 0.0, 0.0)

        waypoints: list[RobotWaypoint] = []
        for frame in source_frames:
            point = frame.object_position
            assert point is not None
            relative_x = point[0] - first[0]
            relative_y = point[1] - first[1]
            along = (relative_x * human_dx + relative_y * human_dy) / human_length_sq
            # Image y increases downward; robot-world y increases upward.
            lateral = (human_dy * relative_x - human_dx * relative_y) / human_length_sq
            position = tuple(
                start[axis]
                + along * robot_delta[axis]
                + lateral * robot_length * lateral_axis[axis]
                for axis in range(3)
            )
            # The phone annotations contain image-plane coordinates only. The
            # basket fixture's z is not the desired can-center height, so use
            # the observed robot can height on the table and add an explicit
            # lift for the transport phase instead of interpolating fixture z.
            if frame.task_phase == "transport":
                position = (position[0], position[1], start[2] + self.lift_height)
            elif frame.task_phase in {"start", "approach", "grasp", "release"}:
                position = (position[0], position[1], start[2])
            waypoints.append(
                RobotWaypoint(
                    timestamp_s=frame.timestamp_s,
                    object_position_xyz=position,
                    task_phase=frame.task_phase,
                    gripper_state=frame.gripper_state,
                    confidence=frame.confidence,
                )
            )

        return RobotDemo(
            demo_id=demo.demo_id,
            task=demo.task,
            source=demo.source,
            coordinate_frame="robot_world_m",
            waypoints=tuple(waypoints),
        )

    @staticmethod
    def _position3(name: str, position: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError(f"{name} must contain three finite values")
        return tuple(float(value) for value in position)
