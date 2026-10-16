"""Closed-loop pick-and-place policies for LIBERO's object suite."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ego2robot.retargeting.base import RobotDemo, RobotWaypoint


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for scalar-first (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = left
    w0, x0, y0, z0 = right
    return np.asarray(
        (
            w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
            w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
            w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
        ),
        dtype=np.float64,
    )


def _rotation_action(current: Sequence[float], target: np.ndarray) -> np.ndarray:
    """Return OSC rotation delta; LIBERO observations store quaternions as xyzw."""
    current_xyzw = np.asarray(current, dtype=np.float64)
    current_q = current_xyzw[[3, 0, 1, 2]]
    current_q /= np.linalg.norm(current_q)
    target_q = target / np.linalg.norm(target)
    error = _quat_multiply(target_q, current_q * np.asarray((1.0, -1.0, -1.0, -1.0)))
    if error[0] < 0:
        error = -error
    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(error[0], -1.0, 1.0))
    rotation_vector = error[1:] * (angle / vector_norm)
    # Robosuite OSC_POSE maps each normalized rotational action unit to 0.5 rad.
    return np.clip(rotation_vector / 0.5, -1.0, 1.0).astype(np.float32)


class PickPlacePolicy:
    """Task-state baseline using object poses and an OSC Cartesian controller."""

    phases = (
        "raise", "orient", "approach", "descend", "pregrasp", "grasp", "lift", "transfer",
        "lower", "release", "retreat", "done",
    )

    def __init__(
        self,
        max_action: float = 1.0,
        tolerance: float = 0.025,
        object_key: str = "alphabet_soup_1_pos",
    ) -> None:
        self.max_action = float(max_action)
        self.tolerance = float(tolerance)
        self.object_key = object_key
        self.reset()

    def reset(self) -> None:
        self.phase_index = 0
        self.phase_steps = 0
        self.start_position: np.ndarray | None = None
        self.pick_position: np.ndarray | None = None
        self.transit_height: float | None = None
        self.target_quaternion: np.ndarray | None = None
        self.orientation_hold_position: np.ndarray | None = None

    @property
    def phase(self) -> str:
        return self.phases[self.phase_index]

    @property
    def finished(self) -> bool:
        """Whether the scripted baseline has completed its release/retreat."""
        return self.phase == "done"

    def act(
        self, observation: dict[str, Any], optional_context: Any | None = None
    ) -> np.ndarray:
        eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
        obj = np.asarray(observation[self.object_key], dtype=np.float32)
        basket = np.asarray(observation["basket_1_pos"], dtype=np.float32)
        if self.start_position is None:
            self.start_position = eef.copy()
            self.pick_position = obj.copy()
            self.transit_height = float(max(eef[2] + 0.10, basket[2] + 0.18))
            initial_xyzw = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
            initial_quaternion = initial_xyzw[[3, 0, 1, 2]]
            initial_quaternion /= np.linalg.norm(initial_quaternion)
            x_quarter_turn = np.asarray((np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0))
            self.target_quaternion = _quat_multiply(x_quarter_turn, initial_quaternion)
        if self.phase == "orient" and self.orientation_hold_position is None:
            self.orientation_hold_position = eef.copy()
        orientation_position = (
            self.orientation_hold_position
            if self.orientation_hold_position is not None
            else eef
        )
        targets = (
            np.asarray((self.start_position[0], self.start_position[1], self.transit_height)),
            orientation_position,
            np.asarray((self.pick_position[0] + 0.12, self.pick_position[1], self.transit_height)),
            np.asarray((self.pick_position[0] + 0.12, self.pick_position[1], self.pick_position[2] + 0.06)),
            self.pick_position + (0.0, 0.0, 0.06),
            self.pick_position + (0.0, 0.0, 0.06),
            obj + (0.0, 0.0, 0.16),
            np.asarray((basket[0], basket[1], self.transit_height)),
            basket + (0.0, 0.0, 0.12),
            basket + (0.0, 0.0, 0.12),
            basket + (0.0, 0.0, 0.20),
            eef,
        )
        gripper = (-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0)
        if self.phase == "done":
            return self._action(np.zeros(3, dtype=np.float32), -1.0)

        target = np.asarray(targets[self.phase_index], dtype=np.float32)
        phase_tolerance = (
            min(self.tolerance, 0.02)
            if self.phase in ("descend", "pregrasp", "grasp", "lower", "release")
            else self.tolerance
        )
        delta = target - eef
        rotation = (
            _rotation_action(observation["robot0_eef_quat"], self.target_quaternion)
            if self.phase == "orient"
            else np.zeros(3, dtype=np.float32)
        )
        at_target = np.linalg.norm(delta) <= phase_tolerance
        if self.phase == "orient":
            at_target = np.linalg.norm(rotation) < 0.05
        if at_target:
            self.phase_steps += 1
            hold_steps = (
                8 if self.phase in ("grasp", "release")
                else 5 if self.phase in ("descend", "pregrasp", "lower")
                else 1
            )
            if self.phase_steps >= hold_steps:
                self.phase_index = min(self.phase_index + 1, len(self.phases) - 1)
                self.phase_steps = 0
                return self.act(observation)
        else:
            self.phase_steps = 0
            delta = np.clip(delta / 0.05, -self.max_action, self.max_action)
        return self._action(delta, gripper[self.phase_index], rotation)

    @staticmethod
    def _action(
        delta: np.ndarray, gripper: float, rotation: np.ndarray | None = None
    ) -> np.ndarray:
        rotation = np.zeros(3, dtype=np.float32) if rotation is None else rotation
        return np.asarray((*delta.tolist(), *rotation.tolist(), gripper), dtype=np.float32)


class DemoConditionedPolicy:
    """Follow retargeted object waypoints, conditioned on a human demo."""

    def __init__(
        self,
        demo: RobotDemo,
        max_action: float = 1.0,
        tolerance: float = 0.025,
        grasp_offset_z: float = 0.02,
        unphased_offset_z: float = 0.005,
        hover_offset_z: float = 0.12,
        carry_offset_z: float = 0.12,
        release_approach_offset_z: float = 0.24,
        release_offset_z: float = 0.08,
        object_key: str = "alphabet_soup_1_pos",
        target_key: str = "basket_1_pos",
        approach_distance: float = 0.08,
        grasp_close_steps: int = 20,
    ) -> None:
        if demo.coordinate_frame != "robot_world_m":
            raise ValueError("demo waypoints must use robot_world_m")
        if not demo.waypoints:
            raise ValueError("demo must contain at least one waypoint")
        self.demo = demo
        self.max_action = float(max_action)
        self.tolerance = float(tolerance)
        self.grasp_offset_z = float(grasp_offset_z)
        self.unphased_offset_z = float(unphased_offset_z)
        self.hover_offset_z = float(hover_offset_z)
        self.carry_offset_z = float(carry_offset_z)
        self.release_approach_offset_z = float(release_approach_offset_z)
        self.release_offset_z = float(release_offset_z)
        self.object_key = object_key
        self.target_key = target_key
        self.approach_distance = float(approach_distance)
        self.grasp_close_steps = int(grasp_close_steps)
        if self.grasp_close_steps < 1:
            raise ValueError("grasp_close_steps must be positive")
        start = np.asarray(self.demo.waypoints[0].object_position_xyz[:2])
        grasp = next(
            (point for point in self.demo.waypoints if point.task_phase == "grasp"),
            self.demo.waypoints[min(1, len(self.demo.waypoints) - 1)],
        )
        approach_vector = np.asarray(grasp.object_position_xyz[:2]) - start
        if np.linalg.norm(approach_vector) < 1e-6:
            transport = next(
                (point for point in self.demo.waypoints if point.task_phase == "transport"),
                self.demo.waypoints[-1],
            )
            approach_vector = np.asarray(transport.object_position_xyz[:2]) - start
        norm = float(np.linalg.norm(approach_vector))
        self.approach_direction_xy = (
            approach_vector / norm if norm > 1e-6 else np.asarray((1.0, 0.0))
        )
        self.reset()

    def reset(self) -> None:
        self.waypoint_index = 0
        self._finished = False
        self.target_quaternion: np.ndarray | None = None
        self.orientation_ready = False
        self.last_task_phase: str | None = None
        self.last_waypoint_index: int | None = None
        self.last_object_waypoint_xyz: tuple[float, float, float] | None = None
        self.last_requested_eef_target: tuple[float, float, float] | None = None
        self.initial_object_position: np.ndarray | None = None
        self.initial_target_position: np.ndarray | None = None
        self.grasp_object_to_eef_offset_xy: np.ndarray | None = None
        self.release_lowered = False
        self.grasp_hold_count = 0

    @property
    def finished(self) -> bool:
        """Whether the final demo waypoint has been reached."""
        return self._finished

    def act(
        self, observation: dict[str, Any], optional_context: Any | None = None
    ) -> np.ndarray:
        if self._finished:
            last = self.demo.waypoints[-1]
            return PickPlacePolicy._action(
                np.zeros(3, dtype=np.float32), self._gripper_action(last)
            )
        eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
        if self.initial_object_position is None:
            self.initial_object_position = np.asarray(
                observation[self.object_key], dtype=np.float32
            ).copy()
        if self.initial_target_position is None:
            self.initial_target_position = np.asarray(
                observation[self.target_key], dtype=np.float32
            ).copy()
        if self.target_quaternion is None:
            initial_xyzw = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
            initial = initial_xyzw[[3, 0, 1, 2]]
            initial /= np.linalg.norm(initial)
            # Retain LIBERO's valid initial top-down orientation. The prior
            # extra quarter-turn caused palm-first contact with the can.
            self.target_quaternion = initial.copy()
            self.orientation_ready = True
        rotation = _rotation_action(observation["robot0_eef_quat"], self.target_quaternion)
        if not self.orientation_ready:
            if np.linalg.norm(rotation) < 0.05:
                self.orientation_ready = True
            else:
                self.last_task_phase = "orientation"
                return PickPlacePolicy._action(np.zeros(3, dtype=np.float32), -1.0, rotation)
        point = self.demo.waypoints[self.waypoint_index]
        if point.task_phase == "transport":
            object_now = np.asarray(observation[self.object_key], dtype=np.float32)
            self.grasp_object_to_eef_offset_xy = object_now[:2] - eef[:2]
        target = self._eef_target(point, self.waypoint_index, observation)
        if point.task_phase == "release" and not self.release_lowered:
            if np.linalg.norm(target[:2] - eef[:2]) <= 0.025:
                self.release_lowered = True
                target = self._eef_target(point, self.waypoint_index, observation)
        delta = target - eef
        distance = float(np.linalg.norm(delta))
        gripper_command = self._gripper_action(point)
        if point.task_phase == "release":
            # Keep the grasp while traveling from the transport waypoint to
            # the metric-anchored container pose. Open only on the terminal
            # release waypoint after it is reached.
            gripper_command = 1.0
        # The measured Panda endpoint settles about 14 mm from this can
        # target at the table edge; a 20 mm reach tolerance allows closure
        # without asking the arm to push farther into the table.
        phase_tolerance = min(self.tolerance, 0.02) if point.task_phase == "grasp" else self.tolerance
        advance = distance <= phase_tolerance
        if point.task_phase == "release" and not self.release_lowered:
            advance = False
        if point.task_phase == "grasp":
            if advance:
                self.grasp_hold_count += 1
                gripper_command = 1.0
                advance = self.grasp_hold_count >= self.grasp_close_steps
            else:
                self.grasp_hold_count = 0
                gripper_command = -1.0
        if advance:
            if self.waypoint_index == len(self.demo.waypoints) - 1:
                self._finished = True
                delta = np.zeros(3, dtype=np.float32)
                if point.task_phase == "release":
                    gripper_command = -1.0
            else:
                # Advance one annotated phase at a time. Start and approach
                # can be nearly coincident, but both phases remain observable.
                self.waypoint_index += 1
                point = self.demo.waypoints[self.waypoint_index]
                target = self._eef_target(point, self.waypoint_index, observation)
                delta = target - eef
                gripper_command = self._gripper_action(point)
                if point.task_phase == "grasp":
                    self.grasp_hold_count = 0
                    gripper_command = -1.0
        if np.linalg.norm(delta) > self.tolerance:
            delta = np.clip(delta / 0.05, -self.max_action, self.max_action)
        else:
            delta = np.zeros(3, dtype=np.float32)

        # Canonical annotations use 0=open and 1=closed; Robosuite uses
        # -1=open and +1=closed; infer missing labels from task phase.
        return PickPlacePolicy._action(delta, gripper_command)

    def _eef_target(
        self,
        point: RobotWaypoint,
        waypoint_index: int,
        observation: dict[str, Any],
    ) -> np.ndarray:
        """Use live object/target poses for metric anchors and demo XY as prior."""
        self.last_task_phase = point.task_phase
        self.last_waypoint_index = waypoint_index
        object_now = np.asarray(observation[self.object_key], dtype=np.float32)
        object_target = np.asarray(point.object_position_xyz, dtype=np.float32).copy()
        assert self.initial_object_position is not None
        if point.task_phase in {"start", "approach"}:
            object_target[2] = self.initial_object_position[2]
            if point.task_phase == "approach":
                object_target[:2] = (
                    self.initial_object_position[:2]
                    - self.approach_direction_xy * self.approach_distance
                )
        elif point.task_phase == "grasp":
            # Track LIBERO's measured can pose, so any small nudge during
            # approach does not leave the gripper targeting stale 2D-mapped XYZ.
            object_target = object_now.copy()
        elif point.task_phase == "release":
            assert self.initial_target_position is not None
            object_target[:2] = self.initial_target_position[:2]
            if self.grasp_object_to_eef_offset_xy is not None:
                # Position the held can over the basket, accounting for its
                # measured offset from the gripper EEF during transport.
                object_target[:2] -= self.grasp_object_to_eef_offset_xy
            object_target[2] = self.initial_object_position[2]
        self.last_object_waypoint_xyz = tuple(float(value) for value in object_target)
        target = object_target.copy()
        if point.task_phase in {"start", "approach"}:
            target[2] += self.hover_offset_z
        elif point.task_phase == "grasp":
            target[2] += self.grasp_offset_z
        elif point.task_phase == "transport":
            target[2] += self.carry_offset_z
        elif point.task_phase == "release":
            target[2] += (
                self.release_offset_z
                if self.release_lowered
                else self.release_approach_offset_z
            )
        else:
            target[2] += self.unphased_offset_z
        self.last_requested_eef_target = tuple(float(value) for value in target)
        return target

    @staticmethod
    def _gripper_action(point: RobotWaypoint) -> float:
        """Use explicit gripper labels, or infer them from the task phase."""
        if point.gripper_state is not None:
            state = point.gripper_state
        else:
            state = 1.0 if point.task_phase in {"grasp", "transport"} else 0.0
        return float(2.0 * state - 1.0)



