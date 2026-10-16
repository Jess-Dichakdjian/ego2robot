"""Create a dependency-free SVG of human and robot object trajectories."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego2robot.demos.io import read_demo
from ego2robot.retargeting import SimpleGeometricRetargeter


def _map_points(
    points: list[tuple[float, float]],
    box: tuple[float, float, float, float],
    *,
    image_coordinates: bool,
) -> list[tuple[float, float]]:
    """Scale points into a plot box while preserving equal axis scale."""
    left, top, width, height = box
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    x_span = max(max_x - min_x, 0.05)
    y_span = max(max_y - min_y, 0.05)
    scale = min(width / x_span, height / y_span)
    drawn_width = x_span * scale
    drawn_height = y_span * scale
    offset_x = left + (width - drawn_width) / 2.0
    offset_y = top + (height - drawn_height) / 2.0
    mapped = []
    for x_value, y_value in points:
        x = offset_x + (x_value - min_x) * scale
        y_delta = y_value - min_y if image_coordinates else max_y - y_value
        y = offset_y + y_delta * scale
        mapped.append((x, y))
    return mapped


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>'


def create_svg(
    demo_path: Path,
    robot_start: tuple[float, float, float],
    robot_goal: tuple[float, float, float],
) -> str:
    """Build a two-panel SVG comparing normalized video and robot paths."""
    demo = read_demo(demo_path)
    robot = SimpleGeometricRetargeter().retarget(demo, robot_start, robot_goal)
    human_xy = [
        frame.object_position
        for frame in demo.frames
        if frame.object_position is not None
    ]
    assert all(point is not None for point in human_xy)
    human_points = [(point[0], point[1]) for point in human_xy if point is not None]
    robot_points = [
        (waypoint.object_position_xyz[0], waypoint.object_position_xyz[1])
        for waypoint in robot.waypoints
    ]
    human_plot = _map_points(
        human_points, (110, 100, 320, 320), image_coordinates=True
    )
    robot_plot = _map_points(
        robot_points, (570, 100, 320, 320), image_coordinates=False
    )
    human_labels = [frame.task_phase or "" for frame in demo.frames if frame.object_position]

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" viewBox="0 0 960 520">',
        '<rect width="960" height="520" fill="#ffffff"/>',
        '<text x="32" y="42" font-family="sans-serif" font-size="24" font-weight="bold">Human object path → robot object waypoints</text>',
        '<text x="110" y="78" font-family="sans-serif" font-size="18">Phone image (normalized)</text>',
        '<text x="570" y="78" font-family="sans-serif" font-size="18">Robot workspace (m)</text>',
        '<rect x="100" y="90" width="340" height="340" fill="#f7f9fc" stroke="#9aa4b2"/>',
        '<rect x="560" y="90" width="340" height="340" fill="#f7f9fc" stroke="#9aa4b2"/>',
        _polyline(human_plot, "#1565c0"),
        _polyline(robot_plot, "#2e7d32"),
        '<text x="270" y="465" text-anchor="middle" font-family="sans-serif" font-size="15">image x →</text>',
        '<text x="730" y="465" text-anchor="middle" font-family="sans-serif" font-size="15">robot x (m) →</text>',
        '<text x="32" y="260" transform="rotate(-90 32 260)" text-anchor="middle" font-family="sans-serif" font-size="15">image y ↓</text>',
        '<text x="500" y="260" transform="rotate(-90 500 260)" text-anchor="middle" font-family="sans-serif" font-size="15">robot y (m) ↑</text>',
    ]
    for index, ((x, y), label) in enumerate(zip(human_plot, human_labels)):
        color = "#2e7d32" if index == len(human_plot) - 1 else "#1565c0"
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        if label:
            elements.append(
                f'<text x="{x + 10:.1f}" y="{y - 8:.1f}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
            )
    for index, (x, y) in enumerate(robot_plot):
        color = "#2e7d32" if index == len(robot_plot) - 1 else "#1565c0"
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
    elements.append(
        f'<text x="32" y="505" font-family="sans-serif" font-size="13">{html.escape(demo.demo_id)} · {html.escape(demo.task)} · object path only (not an end-effector pose)</text>'
    )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_json", type=Path)
    parser.add_argument("--robot-start", type=float, nargs=3, required=True)
    parser.add_argument("--robot-goal", type=float, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    svg = create_svg(
        args.demo_json,
        tuple(args.robot_start),
        tuple(args.robot_goal),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Saved trajectory comparison: {args.output}")


if __name__ == "__main__":
    main()
