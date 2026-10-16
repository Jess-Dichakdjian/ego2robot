"""Train and trial-held-out evaluate a tiny human approach-direction regressor."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ego2robot.demos.approach_model import build_approach_examples, fit_ridge, save_model


def angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left = left / max(float(np.linalg.norm(left)), 1e-12)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    return math.degrees(math.acos(float(np.clip(left @ right, -1.0, 1.0))))


def evaluate_leave_one_trial_out(examples, approach_distance_m: float = 0.08) -> list[dict]:
    trial_ids = sorted({row.demo_id for row in examples})
    predictions = []
    for heldout_id in trial_ids:
        train = [row for row in examples if row.demo_id != heldout_id]
        test = [row for row in examples if row.demo_id == heldout_id]
        model = fit_ridge(train, alpha=1.0)
        train_targets = np.asarray([row.target_direction for row in train])
        constant = train_targets.mean(axis=0)
        constant /= max(float(np.linalg.norm(constant)), 1e-12)
        geometry = np.asarray((-1.0, 0.0))
        for row in test:
            model_prediction = np.asarray(model.predict(row.features))
            for name, pred in (
                ("ridge", model_prediction),
                ("constant_train_mean", constant),
                ("straight_from_start_side", geometry),
            ):
                error = angle_degrees(pred, np.asarray(row.target_direction))
                predictions.append({
                    "heldout_demo_id": heldout_id,
                    "timestamp_s": row.event_timestamp_s,
                    "model": name,
                    "target_along": row.target_direction[0],
                    "target_lateral": row.target_direction[1],
                    "predicted_along": float(pred[0]),
                    "predicted_lateral": float(pred[1]),
                    "angular_error_deg": error,
                    "waypoint_error_m_at_fixed_approach_distance": 2.0 * approach_distance_m * math.sin(math.radians(error) / 2.0),
                    "train_trial_ids": ";".join(sorted({item.demo_id for item in train})),
                })
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction-dir", type=Path, default=ROOT / "data/interaction_annotations")
    parser.add_argument("--canonical-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/interaction_learning")
    args = parser.parse_args()
    examples = build_approach_examples(args.interaction_dir, args.canonical_dir)
    ids = sorted({row.demo_id for row in examples})
    if len(ids) < 4:
        raise SystemExit("Need at least four distinct trial groups for a held-out approach study")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "approach_training_inputs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("demo_id", "timestamp_s", "progress", "lateral", "target_along", "target_lateral"))
        writer.writeheader()
        for row in examples:
            writer.writerow({"demo_id": row.demo_id, "timestamp_s": row.event_timestamp_s, "progress": row.features[0], "lateral": row.features[1], "target_along": row.target_direction[0], "target_lateral": row.target_direction[1]})
    predictions = evaluate_leave_one_trial_out(examples)
    with (args.output_dir / "approach_heldout_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    model = fit_ridge(examples, alpha=1.0)
    save_model(model, args.output_dir / "approach_model.json")
    metrics = {}
    for name in ("ridge", "constant_train_mean", "straight_from_start_side"):
        rows = [row for row in predictions if row["model"] == name]
        angles = [row["angular_error_deg"] for row in rows]
        distances = [row["waypoint_error_m_at_fixed_approach_distance"] for row in rows]
        metrics[name] = {
            "samples": len(rows),
            "mean_angular_error_deg": float(np.mean(angles)),
            "median_angular_error_deg": float(np.median(angles)),
            "mean_fixed_distance_waypoint_error_m": float(np.mean(distances)),
        }
    report = {
        "model": "ridge_approach_direction_v1",
        "alpha": 1.0,
        "split": "leave_one_demo_id_out; feature scaling and model fit use training trials only",
        "training_samples": len(examples),
        "unique_trial_groups": len(ids),
        "trial_ids": ids,
        "features": ["object_progress_along_start_to_release", "object_lateral_offset_normalized_by_route_length"],
        "target": "unit_hand_point_minus_can_center_in_task_aligned_basis",
        "positive_example_filter": "approach hand points from trials without explicit failed_contact/push labels; failed first approaches are reserved for correction analysis",
        "baselines": {
            "constant_train_mean": "mean unit direction of training-trial approach labels",
            "straight_from_start_side": "fixed hand-side vector opposite the can start-to-release axis",
        },
        "metrics": metrics,
        "interpretation_limit": "Offline human-data prediction only; not yet evidence of robot action or manipulation benefit.",
    }
    (args.output_dir / "approach_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
