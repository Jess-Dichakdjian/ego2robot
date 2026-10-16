"""Plot recorded LIBERO Panda end-effector XY paths for the matched route ablation."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
VARIANTS = ("human", "straight", "shuffled")
LABELS = {"human": "Human route", "straight": "Straight route", "shuffled": "Shuffled route"}
COLORS = {"human": "#3569A8", "straight": "#D9772E", "shuffled": "#2B8A6E"}
TEXT = "#25313C"


def main() -> None:
    data_path = ROOT / "results/route_ablation_trajectories.csv"
    with data_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"eef_x_m", "eef_y_m", "initial_state_id", "route_variant", "step"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{data_path} does not contain recorded end-effector trajectories")
    states = sorted({int(row["initial_state_id"]) for row in rows})
    if len(states) != 3:
        raise ValueError(f"expected the three matched initial states, found {states}")

    width, height = 1440, 520
    panel_width = width / len(states)
    plot_x0, plot_y, plot_w, plot_h = 75, 112, 320, 320
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Executed Panda end-effector XY trajectories</title>',
        '<desc id="desc">Recorded end-effector paths for human, straight, and shuffled demonstration routes in three matched LIBERO initial states. Circles mark starts and crosses mark ends.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#25313c}.grid{stroke:#d9dee5;stroke-width:1}.axis{stroke:#65717d;stroke-width:1.2}.tick{font-size:12px}.panel-title{font-size:16px;font-weight:700}.axis-label{font-size:13px}.route{fill:none;stroke-width:3;stroke-linejoin:round;stroke-linecap:round}</style>',
        '<text x="36" y="38" font-size="23" font-weight="700">Executed Panda end-effector XY trajectories</text>',
        '<text x="36" y="64" font-size="13" fill="#52606d">Recorded robot0_eef_pos from the same nine matched episodes; each panel uses its own metric scale.</text>',
    ]

    for panel_index, state in enumerate(states):
        panel_x = panel_index * panel_width
        state_rows = [row for row in rows if int(row["initial_state_id"]) == state]
        routes: dict[str, list[dict[str, str]]] = {}
        for variant in VARIANTS:
            route = sorted(
                (row for row in state_rows if row["route_variant"] == variant),
                key=lambda row: int(row["step"]),
            )
            if not route:
                raise ValueError(f"missing {variant} EEF trajectory for initial state {state}")
            routes[variant] = route

        all_x = [float(row["eef_x_m"]) for route in routes.values() for row in route]
        all_y = [float(row["eef_y_m"]) for route in routes.values() for row in route]
        x_pad = max((max(all_x) - min(all_x)) * 0.08, 0.008)
        y_pad = max((max(all_y) - min(all_y)) * 0.08, 0.008)
        xmin, xmax = min(all_x) - x_pad, max(all_x) + x_pad
        ymin, ymax = min(all_y) - y_pad, max(all_y) + y_pad
        scale = min(plot_w / (xmax - xmin), plot_h / (ymax - ymin))
        used_w, used_h = (xmax - xmin) * scale, (ymax - ymin) * scale
        left = panel_x + plot_x0 + (plot_w - used_w) / 2
        top = plot_y + (plot_h - used_h) / 2
        svg.append(f'<text class="panel-title" x="{panel_x + 38:.1f}" y="94">LIBERO initial state {state}</text>')
        for tick in range(5):
            fraction = tick / 4
            x_value = xmin + (xmax - xmin) * fraction
            x_pixel = left + used_w * fraction
            svg.append(f'<line class="grid" x1="{x_pixel:.1f}" y1="{top:.1f}" x2="{x_pixel:.1f}" y2="{top + used_h:.1f}"/>')
            svg.append(f'<text class="tick" text-anchor="middle" x="{x_pixel:.1f}" y="{top + used_h + 18:.1f}">{x_value:.3f}</text>')
            y_value = ymin + (ymax - ymin) * fraction
            y_pixel = top + used_h * (1 - fraction)
            svg.append(f'<line class="grid" x1="{left:.1f}" y1="{y_pixel:.1f}" x2="{left + used_w:.1f}" y2="{y_pixel:.1f}"/>')
            svg.append(f'<text class="tick" text-anchor="end" x="{left - 8:.1f}" y="{y_pixel + 4:.1f}">{y_value:.3f}</text>')
        svg.append(f'<line class="axis" x1="{left:.1f}" y1="{top + used_h:.1f}" x2="{left + used_w:.1f}" y2="{top + used_h:.1f}"/>')
        svg.append(f'<line class="axis" x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{top + used_h:.1f}"/>')
        svg.append(f'<text class="axis-label" text-anchor="middle" x="{left + used_w / 2:.1f}" y="{top + used_h + 42:.1f}">World X (m)</text>')
        svg.append(f'<text class="axis-label" text-anchor="middle" transform="translate({left - 52:.1f} {top + used_h / 2:.1f}) rotate(-90)">World Y (m)</text>')
        for variant, route in routes.items():
            points = [
                (
                    left + (float(row["eef_x_m"]) - xmin) * scale,
                    top + used_h - (float(row["eef_y_m"]) - ymin) * scale,
                )
                for row in route
            ]
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            svg.append(f'<polyline class="route" stroke="{COLORS[variant]}" points="{coordinates}"/>')
            start_x, start_y = points[0]
            end_x, end_y = points[-1]
            svg.append(f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="5" fill="{COLORS[variant]}" stroke="white" stroke-width="1.5"/>')
            svg.append(f'<path d="M {end_x - 5:.2f} {end_y - 5:.2f} L {end_x + 5:.2f} {end_y + 5:.2f} M {end_x + 5:.2f} {end_y - 5:.2f} L {end_x - 5:.2f} {end_y + 5:.2f}" stroke="{COLORS[variant]}" stroke-width="2.5"/>')

    legend_y, legend_x = 492, 335
    for index, variant in enumerate(VARIANTS):
        x = legend_x + index * 205
        svg.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 30}" y2="{legend_y}" stroke="{COLORS[variant]}" stroke-width="3"/>')
        svg.append(f'<text x="{x + 38}" y="{legend_y + 5}" font-size="13">{html.escape(LABELS[variant])}</text>')
    svg.extend([
        f'<circle cx="{legend_x + 650}" cy="{legend_y}" r="5" fill="{TEXT}"/>',
        f'<text x="{legend_x + 662}" y="{legend_y + 5}" font-size="13">Start</text>',
        f'<path d="M {legend_x + 730} {legend_y - 5} l 10 10 m 0 -10 l -10 10" stroke="{TEXT}" stroke-width="2.5"/>',
        f'<text x="{legend_x + 748}" y="{legend_y + 5}" font-size="13">End</text>',
        '</svg>',
    ])
    output_path = OUT / "eef_trajectory_overlay.svg"
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"Saved {output_path} from {len(states)} matched initial states.")


if __name__ == "__main__":
    main()
