from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from ego2robot.demos.io import read_demo
from evaluate_route_ablation import route_variant


class RouteAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = read_demo(ROOT / "data/processed/phone-demo-001.json")

    def _by_phase(self, demo):
        return {frame.task_phase: frame for frame in demo.frames}

    def test_straight_projects_all_interior_points_and_preserves_endpoints(self):
        straight = route_variant(self.demo, "straight")
        original = self._by_phase(self.demo)
        result = self._by_phase(straight)
        start = original["start"].object_position
        end = original["release"].object_position
        delta = (end[0] - start[0], end[1] - start[1])
        for phase in ("approach", "grasp", "transport"):
            point = result[phase].object_position
            cross = delta[0] * (point[1] - start[1]) - delta[1] * (point[0] - start[0])
            self.assertAlmostEqual(cross, 0.0, places=10)
        self.assertEqual(result["start"].object_position, start)
        self.assertEqual(result["release"].object_position, end)
        self.assertEqual(
            [frame.timestamp_s for frame in straight.frames],
            [frame.timestamp_s for frame in self.demo.frames],
        )

    def test_shuffled_permutation_is_deterministic_and_preserves_phases(self):
        shuffled = route_variant(self.demo, "shuffled", shuffle_seed=23)
        repeated = route_variant(self.demo, "shuffled", shuffle_seed=23)
        original = self._by_phase(self.demo)
        result = self._by_phase(shuffled)
        self.assertEqual(
            [frame.object_position for frame in shuffled.frames],
            [frame.object_position for frame in repeated.frames],
        )
        original_points = sorted(original[p].object_position for p in ("approach", "grasp", "transport"))
        shuffled_points = sorted(result[p].object_position for p in ("approach", "grasp", "transport"))
        self.assertEqual(original_points, shuffled_points)
        self.assertNotEqual(
            [original[p].object_position for p in ("approach", "grasp", "transport")],
            [result[p].object_position for p in ("approach", "grasp", "transport")],
        )
        self.assertEqual(
            [frame.task_phase for frame in shuffled.frames],
            [frame.task_phase for frame in self.demo.frames],
        )
        self.assertEqual(
            [frame.timestamp_s for frame in shuffled.frames],
            [frame.timestamp_s for frame in self.demo.frames],
        )


if __name__ == "__main__":
    unittest.main()
