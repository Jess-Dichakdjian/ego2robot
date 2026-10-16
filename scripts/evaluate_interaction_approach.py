"""Matched three-route LIBERO check using the unchanged demo controller."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ego2robot.demos.io import read_demo, write_demo
from evaluate_route_ablation import route_variant


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def eef_lengths(path: Path) -> tuple[float, float]:
    rows = read_csv(path)
    points = np.asarray([[float(row["eef_x_m"]), float(row["eef_y_m"]), float(row["eef_z_m"])] for row in rows])
    phases = [row["phase"] for row in rows]
    total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    approach = sum(
        float(np.linalg.norm(points[index] - points[index - 1]))
        for index in range(1, len(points))
        if phases[index] == "approach" and phases[index - 1] == "approach"
    )
    return total, approach


def run_case(condition: str, demo_path: Path, model_path: Path | None, state_id: int, seed: int, out_dir: Path, force: bool) -> dict:
    stem = f"{condition}_seed{seed}_state{state_id}"
    metrics_path = out_dir / f"{stem}.csv"
    trajectory_path = out_dir / f"{stem}_trajectory.csv"
    log_path = out_dir / f"{stem}.log"
    if force or not (metrics_path.exists() and trajectory_path.exists() and log_path.exists()):
        command = [
            sys.executable, str(ROOT / "scripts/evaluate.py"),
            "--policy", "demo", "--demo-json", str(demo_path),
            "--episodes", "1", "--seed", str(seed),
            "--initial-state-id", str(state_id), "--max-steps", "500",
            "--settle-steps", "10", "--post-release-steps", "20",
            "--output", str(metrics_path), "--trajectory-output", str(trajectory_path),
            "--trace-gripper",
        ]
        if model_path is not None:
            command.extend(("--interaction-approach-model", str(model_path)))
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    metric = read_csv(metrics_path)[0]
    total_eef, approach_eef = eef_lengths(trajectory_path)
    trajectory_rows = read_csv(trajectory_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    initial_row = trajectory_rows[0]
    initial = np.asarray([float(initial_row["object_x_m"]), float(initial_row["object_y_m"]), float(initial_row["object_z_m"])])
    contact_re = re.compile(r"phase=grasp .*?actual_object_xyz=\[([^\]]+)\].*?object_gripper_contacts=\[([^\]]*)\]")
    contact_matches = list(contact_re.finditer(log_text))
    grasp_contact_steps = sum(bool(match.group(2).strip()) for match in contact_matches)
    first_contact_xyz = next((
        np.asarray([float(value.strip()) for value in match.group(1).split(",")])
        for match in contact_matches if match.group(2).strip()
    ), None)
    object_displacement_by_first_contact = (
        float(np.linalg.norm(first_contact_xyz - initial)) if first_contact_xyz is not None else 0.0
    )
    prediction = None
    match = re.search(r"interaction_approach_prior (\{[^\n]+\})", log_text)
    if match:
        import json
        prediction = json.loads(match.group(1))
    return {
        "condition": condition,
        "seed": int(metric["seed"]),
        "initial_state_id": int(metric["initial_state_id"]),
        "initial_object_xyz_m": ",".join(initial_row[key] for key in ("object_x_m", "object_y_m", "object_z_m")),
        "initial_target_xyz_m": ",".join(initial_row[key] for key in ("target_x_m", "target_y_m", "target_z_m")),
        "success": metric["success"],
        "final_distance_m": float(metric["final_object_target_distance_m"]),
        "object_path_m": float(metric["object_trajectory_length_m"]),
        "eef_path_m": total_eef,
        "approach_eef_path_m": approach_eef,
        "steps": int(metric["steps"]),
        "grasp_contact_steps": grasp_contact_steps,
        "grasp_contact_observed": grasp_contact_steps > 0,
        "object_displacement_by_first_grasp_contact_m": object_displacement_by_first_contact,
        "predicted_approach": prediction,
        "metrics_file": str(metrics_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-json", type=Path, default=ROOT / "data/processed/phone-demo-018.json")
    parser.add_argument("--interaction-model", type=Path, default=ROOT / "results/interaction_learning/approach_model.json")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--initial-state-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/interaction_learning/matched")
    parser.add_argument("--force", action="store_true", help="Rerun existing matched episodes")
    args = parser.parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    demo_path = args.demo_json if args.demo_json.is_absolute() else ROOT / args.demo_json
    model_path = args.interaction_model if args.interaction_model.is_absolute() else ROOT / args.interaction_model
    if not model_path.is_file():
        raise SystemExit(f"approach model not found: {model_path}")
    straight_path = ROOT / "work/interaction_learning/phone-demo-018_straight.json"
    straight_path.parent.mkdir(parents=True, exist_ok=True)
    write_demo(route_variant(read_demo(demo_path), "straight"), straight_path)
    rows: list[dict] = []
    for state_id in args.initial_state_ids:
        rows.append(run_case("straight_geometric", straight_path, None, state_id, args.seed, args.output_dir, args.force))
        rows.append(run_case("human_geometric", demo_path, None, state_id, args.seed, args.output_dir, args.force))
        rows.append(run_case("learned_approach", demo_path, model_path, state_id, args.seed, args.output_dir, args.force))
    output = args.output_dir / "matched_results.csv"
    fields = [key for key in rows[0] if key != "predicted_approach"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    print(f"Saved {len(rows)} matched episodes to {output}")
    for row in rows:
        print(
            f"state={row['initial_state_id']} {row['condition']}: success={row['success']} "
            f"distance={row['final_distance_m']:.4f}m object_path={row['object_path_m']:.3f}m "
            f"eef_path={row['eef_path_m']:.3f}m steps={row['steps']} grasp_contacts={row['grasp_contact_steps']}"
        )


if __name__ == "__main__":
    main()
