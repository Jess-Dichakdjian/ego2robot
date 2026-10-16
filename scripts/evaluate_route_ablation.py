"""Compare human, straight, and shuffled routes with one fixed LIBERO controller.

The phase labels, phase order, timestamps, task, controller, and evaluator stay
fixed. ``straight`` projects every interior XY point onto the endpoint line;
``shuffled`` deterministically permutes the three interior phase coordinates.
The evaluator matches each route across the same explicit LIBERO initial states.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ego2robot.demos.io import read_demo, write_demo


def route_variant(demo, name: str, shuffle_seed: int = 23):
    if name == "human":
        return demo
    phases = [frame.task_phase for frame in demo.frames]
    phase_index = {phase: phases.index(phase) for phase in ("start", "approach", "grasp", "transport", "release")}
    if any(phases.count(phase) != 1 for phase in phase_index):
        raise ValueError("route ablation requires exactly one waypoint for all five phases")
    if any(demo.frames[phase_index[p]].object_position is None for p in phase_index):
        raise ValueError("all five phases need object-center coordinates")

    start = np.asarray(demo.frames[phase_index["start"]].object_position, dtype=np.float64)
    release = np.asarray(demo.frames[phase_index["release"]].object_position, dtype=np.float64)
    delta = release - start
    length_sq = float(delta @ delta)
    if length_sq <= 1e-12:
        raise ValueError("start and release image positions must differ")

    frames = list(demo.frames)
    interior_phases = ("approach", "grasp", "transport")
    points = [np.asarray(demo.frames[phase_index[p]].object_position, dtype=np.float64) for p in interior_phases]
    if name == "straight":
        for phase, point in zip(interior_phases, points):
            alpha = float((point - start) @ delta / length_sq)
            projected = start + np.clip(alpha, 0.0, 1.0) * delta
            frames[phase_index[phase]] = replace(
                frames[phase_index[phase]],
                object_position=tuple(float(value) for value in projected),
            )
    elif name == "shuffled":
        permutation = np.random.default_rng(shuffle_seed).permutation(len(points))
        if np.array_equal(permutation, np.arange(len(points))):
            permutation = np.roll(permutation, 1)
        shuffled = [points[index] for index in permutation]
        for phase, point in zip(interior_phases, shuffled):
            if np.any(point < 0.0) or np.any(point > 1.0):
                raise ValueError("shuffled point leaves normalized image bounds")
            frames[phase_index[phase]] = replace(
                frames[phase_index[phase]],
                object_position=tuple(float(value) for value in point),
            )
    else:
        raise ValueError(f"unknown route variant: {name}")
    metadata = dict(demo.metadata)
    metadata["route_ablation"] = name
    return replace(demo, frames=tuple(frames), metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-json", type=Path, default=Path("data/processed/phone-demo-001.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--initial-states", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shuffle-seed", type=int, default=23)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("results/route_ablation.csv"))
    args = parser.parse_args()

    source_path = args.demo_json if args.demo_json.is_absolute() else REPO_ROOT / args.demo_json
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    source_demo = read_demo(source_path)
    scratch = REPO_ROOT / "work" / "route_ablation"
    scratch.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    trajectory_rows: list[dict[str, object]] = []
    for variant in ("human", "straight", "shuffled"):
        variant_path = scratch / f"{source_demo.demo_id}_{variant}.json"
        write_demo(route_variant(source_demo, variant, args.shuffle_seed), variant_path)
        for seed in args.seeds:
            for initial_state_id in args.initial_states:
                metrics_path = scratch / f"{source_demo.demo_id}_{variant}_seed{seed}_state{initial_state_id}.csv"
                trajectory_path = scratch / f"{source_demo.demo_id}_{variant}_seed{seed}_state{initial_state_id}_trajectory.csv"
                command = [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "evaluate.py"),
                    "--policy", "demo",
                    "--demo-json", str(variant_path),
                    "--episodes", "1",
                    "--seed", str(seed),
                    "--initial-state-id", str(initial_state_id),
                    "--max-steps", str(args.max_steps),
                    "--settle-steps", "10",
                    "--post-release-steps", "20",
                    "--output", str(metrics_path),
                    "--trajectory-output", str(trajectory_path),
                ]
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                with metrics_path.open(newline="", encoding="utf-8") as stream:
                    metric = next(csv.DictReader(stream))
                rows.append({
                    "demo_id": source_demo.demo_id,
                    "route_variant": variant,
                    "seed": seed,
                    "initial_state_id": initial_state_id,
                    **metric,
                })
                with trajectory_path.open(newline="", encoding="utf-8") as stream:
                    for sample in csv.DictReader(stream):
                        trajectory_rows.append({"route_variant": variant, **sample})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "demo_id", "route_variant", "seed", "initial_state_id", "policy",
        "success", "episode_return", "final_object_target_distance_m",
        "object_trajectory_length_m", "steps",
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    trajectory_output = REPO_ROOT / "results" / "route_ablation_trajectories.csv"
    trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=trajectory_rows[0].keys())
        writer.writeheader()
        writer.writerows(trajectory_rows)
    print(f"Saved matched route ablation: {output_path}")
    print(f"Saved object trajectories: {trajectory_output}")


if __name__ == "__main__":
    main()
