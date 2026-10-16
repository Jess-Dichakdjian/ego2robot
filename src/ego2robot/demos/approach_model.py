"""Tiny ridge model for human hand approach side, in task-aligned 2D coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ego2robot.demos.interaction import load_interaction


@dataclass(frozen=True)
class ApproachExample:
    demo_id: str
    features: tuple[float, float]
    target_direction: tuple[float, float]
    event_timestamp_s: float


@dataclass(frozen=True)
class RidgeApproachModel:
    feature_mean: tuple[float, float]
    feature_scale: tuple[float, float]
    target_mean: tuple[float, float]
    weights: tuple[tuple[float, float], tuple[float, float]]
    alpha: float = 1.0

    def predict(self, features: Iterable[float]) -> tuple[float, float]:
        x = np.asarray(tuple(features), dtype=np.float64)
        if x.shape != (2,) or not np.isfinite(x).all():
            raise ValueError("approach model expects two finite features")
        standardized = (x - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        result = np.asarray(self.target_mean) + standardized @ np.asarray(self.weights)
        norm = float(np.linalg.norm(result))
        if norm < 1e-8:
            raise ValueError("model predicted a degenerate approach direction")
        result /= norm
        return float(result[0]), float(result[1])

    def to_dict(self) -> dict:
        return {
            "model": "ridge_approach_direction_v1",
            "features": ["object_progress_along_start_to_release", "object_lateral_offset_normalized_by_route_length"],
            "target": "unit_hand_point_minus_can_center_in_task_aligned_basis",
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "target_mean": list(self.target_mean),
            "weights": [list(row) for row in self.weights],
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "RidgeApproachModel":
        return cls(
            feature_mean=tuple(value["feature_mean"]),
            feature_scale=tuple(value["feature_scale"]),
            target_mean=tuple(value["target_mean"]),
            weights=tuple(tuple(row) for row in value["weights"]),
            alpha=float(value.get("alpha", 1.0)),
        )


def _object_map(canonical: dict, timestamp_s: float) -> tuple[float, float]:
    frame = min(canonical["frames"], key=lambda f: abs(float(f["timestamp_s"]) - timestamp_s))
    if abs(float(frame["timestamp_s"]) - timestamp_s) > 0.002:
        raise ValueError(f"no canonical object point at {timestamp_s:.3f}s")
    point = frame.get("object_position")
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError("canonical event lacks a 2D object point")
    return float(point[0]), float(point[1])


def build_approach_examples(interaction_dir: Path, canonical_dir: Path) -> tuple[ApproachExample, ...]:
    """Build only explicit approach examples with a visible human landmark."""
    examples: list[ApproachExample] = []
    for path in sorted(interaction_dir.glob("phone-demo-*-interaction.json")):
        raw, events = load_interaction(path)
        canonical = json.loads((canonical_dir / f"{raw['demo_id']}.json").read_text(encoding="utf-8"))
        valid_frames = [f for f in canonical["frames"] if f.get("object_position") is not None]
        # A failed first approach is a correction target, not a positive grasp
        # approach example. Keep those clips out of this predictor's training.
        if any(
            (f.get("task_phase") or f.get("event_type")) == "failed_contact/push"
            for f in valid_frames
        ):
            continue
        starts = [f for f in valid_frames if f.get("task_phase") == "start"]
        releases = [f for f in valid_frames if f.get("task_phase") == "release"]
        if not starts or not releases:
            continue
        start = np.asarray(starts[0]["object_position"], dtype=np.float64)
        goal = np.asarray(releases[-1]["object_position"], dtype=np.float64)
        route = goal - start
        route_length = float(np.linalg.norm(route))
        if route_length <= 1e-6:
            continue
        route_axis = route / route_length
        # Image-space positive y points down. This right-handed local axis
        # matches SimpleGeometricRetargeter's lateral sign convention.
        lateral_axis = np.asarray((route_axis[1], -route_axis[0]))
        for event in events:
            if event.event_type != "approach" or event.hand_point_px is None:
                continue
            obj = np.asarray(_object_map(canonical, event.timestamp_s))
            rel_obj = obj - start
            features = (
                float(rel_obj @ route_axis / route_length),
                float(rel_obj @ lateral_axis / route_length),
            )
            displacement = np.asarray(event.hand_to_object_px, dtype=np.float64)
            magnitude = float(np.linalg.norm(displacement))
            if magnitude <= 1e-6:
                continue
            target = (
                float(displacement @ route_axis / magnitude),
                float(displacement @ lateral_axis / magnitude),
            )
            examples.append(ApproachExample(raw["demo_id"], features, target, event.timestamp_s))
    return tuple(examples)


def fit_ridge(examples: Iterable[ApproachExample], alpha: float = 1.0) -> RidgeApproachModel:
    rows = tuple(examples)
    if len(rows) < 3 or len({row.demo_id for row in rows}) < 3:
        raise ValueError("at least three approach examples from distinct trials are required")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    x = np.asarray([row.features for row in rows], dtype=np.float64)
    y = np.asarray([row.target_direction for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    target_mean = y.mean(axis=0)
    weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - target_mean))
    return RidgeApproachModel(tuple(mean), tuple(scale), tuple(target_mean), tuple(map(tuple, weights)), alpha)


def save_model(model: RidgeApproachModel, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
