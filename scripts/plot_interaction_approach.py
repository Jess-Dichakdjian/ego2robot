"""Plot human approach predictions, robot actions, and matched outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/interaction_learning/matched_success_approaches")
    parser.add_argument("--output", type=Path, default=ROOT / "results/interaction_learning/approach_to_robot_action.svg")
    args = parser.parse_args()
    result_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    metrics = rows(result_dir / "matched_results.csv")
    predictions = rows(ROOT / "results/interaction_learning/approach_heldout_predictions.csv")
    colors = {"straight_geometric": "#9b9b9b", "human_geometric": "#2878b5", "learned_approach": "#d14b40"}
    labels = {"straight_geometric": "Straight", "human_geometric": "Human route", "learned_approach": "Learned approach"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle("Human hand approach prior: prediction to Panda execution", fontsize=16, fontweight="bold")

    # Panel A: held-out directions are in task-aligned image coordinates.
    ax = axes[0, 0]
    theta = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(theta), np.sin(theta), color="#dddddd", linewidth=1)
    subset = [row for row in predictions if row["model"] in ("ridge", "constant_train_mean")]
    for row in subset:
        model = row["model"]
        x, y = float(row["predicted_along"]), float(row["predicted_lateral"])
        trial = row["heldout_demo_id"][-3:]
        ax.scatter([x], [y], s=22, color="#d14b40" if model == "ridge" else "#2878b5", marker="o" if model == "ridge" else "x", alpha=.75)
        if model == "ridge":
            tx, ty = float(row["target_along"]), float(row["target_lateral"])
            ax.annotate(trial, (tx, ty), xytext=(3, 2), textcoords="offset points", fontsize=8)
            ax.plot([x, tx], [y, ty], color="#d14b40", alpha=.3, linewidth=.8)
    ax.scatter([], [], color="#d14b40", marker="o", label="Ridge held-out prediction")
    ax.scatter([], [], color="#2878b5", marker="x", label="Constant train mean")
    ax.scatter([], [], color="#236b38", marker="^", label="Human hand-point target")
    for row in predictions:
        if row["model"] == "ridge":
            ax.scatter([float(row["target_along"])], [float(row["target_lateral"])], color="#236b38", marker="^", s=24)
    ax.axhline(0, color="#eeeeee", linewidth=.8); ax.axvline(0, color="#eeeeee", linewidth=.8)
    ax.set_aspect("equal"); ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
    ax.set_xlabel("Along task route"); ax.set_ylabel("Lateral to task route")
    ax.set_title("A  Leave-one-trial-out approach directions")
    ax.legend(fontsize=8, loc="lower left")

    # Panel B: executed EEF traces and commanded approach waypoint on the action-influence state.
    ax = axes[0, 1]
    state0 = [row for row in metrics if int(row["initial_state_id"]) == 0]
    for row in state0:
        path = result_dir / f"{row['condition']}_seed{row['seed']}_state0_trajectory.csv"
        samples = rows(path)
        eef = np.asarray([[float(r["eef_x_m"]), float(r["eef_y_m"])] for r in samples])
        ax.plot(eef[:, 0], eef[:, 1], color=colors[row["condition"]], label=labels[row["condition"]], linewidth=1.6)
    if state0:
        first = rows(result_dir / f"straight_geometric_seed{state0[0]['seed']}_state0_trajectory.csv")[0]
        object_xy = np.asarray([float(first["object_x_m"]), float(first["object_y_m"])])
        target_xy = np.asarray([float(first["target_x_m"]), float(first["target_y_m"])])
        ax.scatter(*object_xy, color="#111111", marker="o", s=45, label="Can start")
        ax.scatter(*target_xy, color="#7a4f9a", marker="s", s=45, label="Basket")
        learned_log = (result_dir / "learned_approach_seed41_state0.log").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"interaction_approach_prior (\{[^\n]+\})", learned_log)
        if match:
            command = json.loads(match.group(1))["commanded_policy_approach_direction_xy"]
            approach_target = object_xy - np.asarray(command, dtype=float) * .08
            ax.annotate("Model commanded 8 cm stand-off", xy=approach_target, xytext=object_xy,
                        arrowprops={"arrowstyle": "->", "color": "#d14b40", "lw": 1.6}, fontsize=8, color="#d14b40")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Robot X (m)"); ax.set_ylabel("Robot Y (m)")
    ax.set_title("B  Executed Panda EEF paths · seed 41, state 0")
    ax.legend(fontsize=8, loc="best")

    # Panel C: actual final distances, with each simulator state as one paired observation.
    ax = axes[1, 0]
    states = sorted({int(row["initial_state_id"]) for row in metrics})
    x = np.arange(len(states)); width = .23
    for offset, condition in zip((-width, 0, width), colors):
        values = [next(float(r["final_distance_m"]) for r in metrics if int(r["initial_state_id"]) == s and r["condition"] == condition) for s in states]
        ax.bar(x + offset, values, width, color=colors[condition], label=labels[condition])
    ax.axhline(.06, color="#666666", linestyle="--", linewidth=.8, label="Existing success boundary")
    ax.set_xticks(x, [f"State {s}" for s in states]); ax.set_ylabel("Final can–basket distance (m)")
    ax.set_title("C  Matched placement error")
    ax.legend(fontsize=8)

    # Panel D: episode steps and outcomes.
    ax = axes[1, 1]
    for condition in colors:
        values = [next(int(r["steps"]) for r in metrics if int(r["initial_state_id"]) == s and r["condition"] == condition) for s in states]
        ax.plot(x, values, marker="o", color=colors[condition], label=labels[condition])
    ax.set_xticks(x, [f"State {s}" for s in states]); ax.set_ylabel("Simulation steps (max 500)")
    ax.set_title("D  Matched task completion cost")
    ax.legend(fontsize=8)
    fig.text(.5, -.015, "Five successful human approaches train the ridge prior; no recovery model. All methods use the same fixed DemoConditionedPolicy.", ha="center", fontsize=9)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
