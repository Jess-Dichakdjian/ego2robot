"""Versioned canonical representation for human or robot demonstrations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _validate_vector(
    name: str, value: tuple[float, ...] | None, lengths: tuple[int, ...]
) -> None:
    if value is None:
        return
    if not isinstance(value, (tuple, list)) or len(value) not in lengths:
        choices = " or ".join(str(length) for length in lengths)
        raise ValueError(f"{name} must contain {choices} values")
    normalized = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in normalized):
        raise ValueError(f"{name} values must be finite")


@dataclass(frozen=True)
class DemoFrame:
    """One timestamped observation or annotation in a demonstration.

    Image-space positions use normalized ``(x, y)`` coordinates in [0, 1].
    ``gripper_state`` is optional and ranges from 0 (open) to 1 (closed).
    """

    timestamp_s: float
    object_position: tuple[float, ...] | None = None
    ee_position: tuple[float, ...] | None = None
    ee_orientation_xyzw: tuple[float, ...] | None = None
    gripper_state: float | None = None
    target_position: tuple[float, ...] | None = None
    task_phase: str | None = None
    source: str = "phone_video_manual"
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        for name, lengths in (
            ("object_position", (2, 3)),
            ("ee_position", (3,)),
            ("ee_orientation_xyzw", (4,)),
            ("target_position", (2, 3)),
        ):
            value = getattr(self, name)
            _validate_vector(name, value, lengths)
            if value is not None:
                object.__setattr__(self, name, tuple(float(x) for x in value))
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.gripper_state is not None and (
            not math.isfinite(self.gripper_state)
            or not 0.0 <= self.gripper_state <= 1.0
        ):
            raise ValueError("gripper_state must be between 0 and 1")
        if self.confidence is not None and (
            not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class CanonicalDemo:
    """A sequence of timestamped annotations in one declared coordinate frame."""

    demo_id: str
    task: str
    coordinate_frame: str
    source: str
    frames: tuple[DemoFrame, ...]
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("demo_id", "task", "coordinate_frame", "source"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("a demo must contain at least one frame")
        if not all(isinstance(frame, DemoFrame) for frame in frames):
            raise TypeError("frames must contain DemoFrame instances")
        timestamps = [frame.timestamp_s for frame in frames]
        if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
            raise ValueError("frame timestamps must be strictly increasing")
        if self.coordinate_frame == "image_normalized":
            for frame in frames:
                for name in ("object_position", "target_position"):
                    position = getattr(frame, name)
                    if position is not None and (
                        len(position) != 2
                        or any(component < 0.0 or component > 1.0 for component in position)
                    ):
                        raise ValueError(
                            f"{name} must be normalized 2D coordinates in [0, 1]"
                        )
        object.__setattr__(self, "frames", frames)
