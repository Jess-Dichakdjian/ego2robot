"""Evaluation-only bridge from learned image-plane approach direction to LIBERO."""

from __future__ import annotations

from typing import Any

import numpy as np

from ego2robot.demos.approach_model import RidgeApproachModel


def apply_interaction_approach_prior(
    policy: Any,
    model: RidgeApproachModel,
    object_start_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
) -> dict[str, list[float]]:
    """Change only the approach-side direction; controller mechanics stay fixed.

    The input progress/lateral features are measured from the live simulator's
    initial object location relative to its task target. This is (0, 0) by
    construction at episode start, matching the annotated pre-contact state.
    Human image-plane task coordinates are mapped into LIBERO's endpoint basis,
    the same basis convention as the geometric retargeter.
    """
    start = np.asarray(object_start_xyz, dtype=np.float64)
    goal = np.asarray(target_xyz, dtype=np.float64)
    route = goal[:2] - start[:2]
    route_norm = float(np.linalg.norm(route))
    if route_norm <= 1e-8:
        raise ValueError("object and target must have distinct XY positions")
    route_axis = route / route_norm
    lateral_axis = np.asarray((-route_axis[1], route_axis[0]))
    features = (0.0, 0.0)
    predicted_hand_side_local = np.asarray(model.predict(features), dtype=np.float64)
    predicted_hand_side_world = (
        predicted_hand_side_local[0] * route_axis
        + predicted_hand_side_local[1] * lateral_axis
    )
    # DemoConditionedPolicy places the pregrasp at object - direction * d.
    commanded_direction = -predicted_hand_side_world
    policy.approach_direction_xy = commanded_direction
    return {
        "model_features": list(features),
        "predicted_human_hand_side_task_basis": predicted_hand_side_local.tolist(),
        "mapped_human_hand_side_robot_xy": predicted_hand_side_world.tolist(),
        "commanded_policy_approach_direction_xy": commanded_direction.tolist(),
    }
