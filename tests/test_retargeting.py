"""Coordinate transform checks for the simple geometric retargeter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego2robot.demos.io import read_demo
from ego2robot.demos.schema import CanonicalDemo, DemoFrame
from ego2robot.retargeting import SimpleGeometricRetargeter


class SimpleGeometricRetargeterTests(unittest.TestCase):
    def test_endpoints_map_to_robot_start_and_goal(self) -> None:
        demo = read_demo(
            Path(__file__).resolve().parents[1]
            / "data/processed/example_demo.json"
        )
        robot = SimpleGeometricRetargeter().retarget(
            demo,
            robot_start_position=(0.4, 0.1, 0.03),
            robot_goal_position=(0.7, -0.2, 0.03),
        )
        self.assertEqual(robot.coordinate_frame, "robot_world_m")
        for actual, expected in zip(
            robot.waypoints[0].object_position_xyz, (0.4, 0.1, 0.03)
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            robot.waypoints[-1].object_position_xyz, (0.7, -0.2, 0.03)
        ):
            self.assertAlmostEqual(actual, expected)

    def test_lateral_deviation_is_preserved(self) -> None:
        demo = CanonicalDemo(
            demo_id="lateral",
            task="pick and place",
            coordinate_frame="image_normalized",
            source="test",
            frames=(
                DemoFrame(timestamp_s=0.0, object_position=(0.0, 0.0)),
                DemoFrame(timestamp_s=1.0, object_position=(0.5, 0.75)),
                DemoFrame(timestamp_s=2.0, object_position=(1.0, 1.0)),
            ),
        )
        robot = SimpleGeometricRetargeter().retarget(
            demo,
            robot_start_position=(0.0, 0.0, 0.0),
            robot_goal_position=(0.4, 0.0, 0.0),
        )
        middle = robot.waypoints[1].object_position_xyz
        self.assertAlmostEqual(middle[0], 0.25)
        self.assertAlmostEqual(middle[1], -0.05)

    def test_phase_heights_do_not_follow_basket_fixture_z(self) -> None:
        demo = CanonicalDemo(
            demo_id="phase-height",
            task="pick and place",
            coordinate_frame="image_normalized",
            source="test",
            frames=(
                DemoFrame(0.0, object_position=(0.0, 0.0), task_phase="start"),
                DemoFrame(1.0, object_position=(0.5, 0.5), task_phase="transport"),
                DemoFrame(2.0, object_position=(1.0, 1.0), task_phase="release"),
            ),
        )
        robot = SimpleGeometricRetargeter(lift_height=0.12).retarget(
            demo,
            robot_start_position=(0.0, 0.0, 0.038),
            robot_goal_position=(0.4, 0.0, -0.005),
        )

        self.assertAlmostEqual(robot.waypoints[0].object_position_xyz[2], 0.038)
        self.assertAlmostEqual(robot.waypoints[1].object_position_xyz[2], 0.158)
        self.assertAlmostEqual(robot.waypoints[2].object_position_xyz[2], 0.038)

    def test_missing_coordinates_are_ignored_but_two_are_required(self) -> None:
        demo = CanonicalDemo(
            demo_id="insufficient",
            task="pick and place",
            coordinate_frame="image_normalized",
            source="test",
            frames=(DemoFrame(timestamp_s=0.0), DemoFrame(timestamp_s=1.0)),
        )
        with self.assertRaisesRegex(ValueError, "two object-position"):
            SimpleGeometricRetargeter().retarget(
                demo, (0.0, 0.0, 0.0), (0.1, 0.0, 0.0)
            )


if __name__ == "__main__":
    unittest.main()
