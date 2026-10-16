"""Typed access to supplementary human hand/object interaction annotations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InteractionEvent:
    """One sparse human-labelled hand/object observation in image pixels."""

    demo_id: str
    timestamp_s: float
    event_type: str
    object_center_px: tuple[float, float]
    hand_point_px: tuple[float, float] | None
    hand_point_kind: str | None
    hand_visibility: str
    confidence: float
    image_width_px: int
    image_height_px: int

    @property
    def object_center_norm(self) -> tuple[float, float]:
        return (
            self.object_center_px[0] / self.image_width_px,
            self.object_center_px[1] / self.image_height_px,
        )

    @property
    def hand_point_norm(self) -> tuple[float, float] | None:
        if self.hand_point_px is None:
            return None
        return (
            self.hand_point_px[0] / self.image_width_px,
            self.hand_point_px[1] / self.image_height_px,
        )

    @property
    def hand_to_object_px(self) -> tuple[float, float] | None:
        """Return the labelled anatomical point relative to the can centre."""
        if self.hand_point_px is None:
            return None
        return (
            self.hand_point_px[0] - self.object_center_px[0],
            self.hand_point_px[1] - self.object_center_px[1],
        )


def _point(value: Any, name: str, *, optional: bool = False) -> tuple[float, float] | None:
    if value is None and optional:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value list" if not optional else f"{name} must be null or a two-value list")
    point = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in point):
        raise ValueError(f"{name} must contain finite values")
    return point


def load_interaction(path: Path) -> tuple[dict[str, Any], tuple[InteractionEvent, ...]]:
    """Load and validate one interaction JSON without filling missing points."""
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "annotation_type": "ego2robot_human_object_interaction",
        "schema_version": 1,
        "coordinate_frame": "image_pixels_top_left",
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise ValueError(f"{path.name}: expected {key}={expected!r}")
    width, height = data.get("image_width_px"), data.get("image_height_px")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(f"{path.name}: invalid image dimensions")
    demo_id = data.get("demo_id")
    if not isinstance(demo_id, str) or not demo_id:
        raise ValueError(f"{path.name}: missing demo_id")
    events: list[InteractionEvent] = []
    prior_time = -math.inf
    for index, raw in enumerate(data.get("events", [])):
        timestamp = float(raw["timestamp_s"])
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= prior_time:
            raise ValueError(f"{path.name}: event timestamps must be finite and strictly increasing")
        prior_time = timestamp
        object_px = _point(raw.get("object_center_px"), "object_center_px")
        hand_px = _point(raw.get("hand_point_px"), "hand_point_px", optional=True)
        object_norm = _point(raw.get("object_center_norm"), "object_center_norm")
        hand_norm = _point(raw.get("hand_point_norm"), "hand_point_norm", optional=True)
        confidence = float(raw["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"{path.name}: invalid confidence at event {index}")
        for name, point in (("object_center_px", object_px), ("hand_point_px", hand_px)):
            if point is not None and not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError(f"{path.name}: {name} outside frame at event {index}")
        expected_object_norm = (object_px[0] / width, object_px[1] / height)
        if math.dist(object_norm, expected_object_norm) > 0.002:
            raise ValueError(f"{path.name}: object pixel/normalized mismatch at event {index}")
        if hand_px is None:
            if hand_norm is not None or raw.get("hand_point_kind") is not None:
                raise ValueError(f"{path.name}: missing hand point must not carry coordinates or landmark kind")
        else:
            if hand_norm is None or not raw.get("hand_point_kind"):
                raise ValueError(f"{path.name}: hand point must retain normalized point and anatomical kind")
            if math.dist(hand_norm, (hand_px[0] / width, hand_px[1] / height)) > 0.002:
                raise ValueError(f"{path.name}: hand pixel/normalized mismatch at event {index}")
        events.append(InteractionEvent(
            demo_id=demo_id,
            timestamp_s=timestamp,
            event_type=str(raw.get("event_type", "uncertain_event")),
            object_center_px=object_px,
            hand_point_px=hand_px,
            hand_point_kind=raw.get("hand_point_kind"),
            hand_visibility=str(raw.get("hand_visibility", "uncertain")),
            confidence=confidence,
            image_width_px=width,
            image_height_px=height,
        ))
    if not events:
        raise ValueError(f"{path.name}: no interaction events")
    return data, tuple(events)
