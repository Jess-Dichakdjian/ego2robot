"""Plot matched route-ablation metrics and per-state object-path overlays."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
VARIANTS = ("human", "straight", "shuffled")
LABELS = {"human": "Human route", "straight": "Straight route", "shuffled": "Shuffled route"}
COLORS = {"human": "#3569A8", "straight": "#D9772E", "shuffled": "#2B8A6E"}
TEXT = "#25313C"
GRID = "#D9DEE5"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    svg_path = OUT / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    results = read_rows(ROOT / "results/route_ablation.csv")
    trajectories = read_rows(ROOT / "results/route_ablation_trajectories.csv")
    groups = {name: [row for row in results if row["route_variant"] == name] for name in VARIANTS}
    if any(not groups[name] for name in VARIANTS):
        raise ValueError("route_ablation.csv needs human, straight, and shuffled runs")

    state_ids = sorted({int(row["initial_state_id"]) for row in results})
    if len(state_ids) < 2:
        raise ValueError("run at least two distinct LIBERO initial states for the presentation plots")

    summary: list[dict[str, str]] = []
    for name in VARIANTS:
        group = groups[name]
        distances = [float(row["final_object_target_distance_m"]) for row in group]
        paths = [float(row["object_trajectory_length_m"]) for row in group]
        steps = [int(row["steps"]) for row in group]
        successes = sum(row["success"].lower() == "true" for row in group)
        summary.append({
            "route": LABELS[name],
            "episodes": str(len(group)),
            "success": f"{successes}/{len(group)} ({successes / len(group):.0%})",
            "final_distance_m_median_range": f"{statistics.median(distances):.4f} ({min(distances):.4f}–{max(distances):.4f})",
            "object_path_m_median_range": f"{statistics.median(paths):.4f} ({min(paths):.4f}–{max(paths):.4f})",
            "steps_median_range": f"{statistics.median(steps):g} ({min(steps)}–{max(steps)})",
        })

    fields = list(summary[0])
    with (OUT / "results_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    with (OUT / "results_table.md").open("w", encoding="utf-8") as stream:
        stream.write("# Matched route ablation\n\n")
        stream.write(
            f"Same controller and task conditions across {len(state_ids)} distinct LIBERO initial states. "
            "Values are medians over matched conditions; see `../route_ablation.csv` for episode-level rows.\n\n"
        )
        stream.write("| Route prior | Episodes | Success | Final distance m: median (range) | Object path m: median (range) | Steps: median (range) |\n")
        stream.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary:
            stream.write("| " + " | ".join(row[field] for field in fields) + " |\n")

    success = [
        sum(row["success"].lower() == "true" for row in groups[name]) / len(groups[name])
        for name in VARIANTS
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    style_axis(ax)
    bars = ax.bar(range(len(VARIANTS)), success, width=0.56, color=[COLORS[n] for n in VARIANTS])
    ax.set_xticks(range(len(VARIANTS)), [f"{LABELS[n]}\n(n={len(groups[n])})" for n in VARIANTS])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Success rate")
    ax.set_title("Matched route ablation: success", loc="left", color=TEXT, fontsize=13, weight="bold")
    for bar, value in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.0%}", ha="center", color=TEXT, fontsize=10)
    save(fig, "success_rate")

    specs = [
        ("final_object_target_distance", "Final object-to-target distance", "Distance (m)", "final_object_target_distance_m"),
        ("trajectory_length", "Object trajectory length", "Path length (m)", "object_trajectory_length_m"),
        ("episode_steps", "Episode length", "Steps", "steps"),
    ]
    for stem, title, ylabel, key in specs:
        fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
        style_axis(ax)
        values_by_variant = [[float(row[key]) for row in groups[name]] for name in VARIANTS]
        for index, (name, values) in enumerate(zip(VARIANTS, values_by_variant)):
            jitter = [(i - (len(values) - 1) / 2) * 0.055 for i in range(len(values))]
            ax.scatter([index + offset for offset in jitter], values, s=52, color=COLORS[name], edgecolor="white", linewidth=0.7, zorder=3)
            ax.hlines(statistics.median(values), index - 0.18, index + 0.18, color=COLORS[name], linewidth=2, zorder=2)
        all_values = [value for values in values_by_variant for value in values]
        low, high = min(all_values), max(all_values)
        padding = max((high - low) * 0.2, max(abs(low), abs(high), 1) * 0.025)
        ax.set_ylim(max(0, low - padding), high + padding)
        ax.set_xticks(range(len(VARIANTS)), [LABELS[n] for n in VARIANTS])
        ax.set_xlim(-0.5, len(VARIANTS) - 0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=TEXT, fontsize=13, weight="bold")
        save(fig, stem)

    fig, axes = plt.subplots(1, len(state_ids), figsize=(5.2 * len(state_ids), 4.8), layout="constrained", squeeze=False)
    for ax, state_id in zip(axes[0], state_ids):
        style_axis(ax)
        state_rows = [row for row in trajectories if int(row["initial_state_id"]) == state_id]
        for name in VARIANTS:
            path = sorted(
                (row for row in state_rows if row["route_variant"] == name),
                key=lambda row: int(row["step"]),
            )
            if not path:
                raise ValueError(f"no trajectory for route={name}, initial_state_id={state_id}")
            x = [float(row["object_x_m"]) for row in path]
            y = [float(row["object_y_m"]) for row in path]
            ax.plot(x, y, color=COLORS[name], linewidth=1.8, label=LABELS[name])
            ax.scatter(x[0], y[0], color=COLORS[name], marker="o", s=24, zorder=4)
        target = state_rows[0]
        ax.scatter(float(target["target_x_m"]), float(target["target_y_m"]), marker="*", s=110, color="#C43C39", label="Container target", zorder=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_title(f"LIBERO initial state {state_id}", loc="left", color=TEXT, fontsize=12, weight="bold")
        ax.grid(color=GRID, linewidth=0.7, alpha=0.8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False)
    save(fig, "trajectory_overlay")


if __name__ == "__main__":
    main()
