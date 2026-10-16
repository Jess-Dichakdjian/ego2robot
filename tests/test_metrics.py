"""Tests for episode-level manipulation metrics."""

import unittest

from ego2robot.evaluation.metrics import measure_episode


class EpisodeMetricTests(unittest.TestCase):
    def test_calculates_distance_path_length_and_steps(self):
        metrics = measure_episode(
            [(0, 0, 0), (0.3, 0.0, 0.0), (0.3, 0.4, 0.0)],
            (0.3, 0.4, 0.5),
            episode_return=2.0,
            success=False,
        )
        self.assertEqual(metrics.steps, 2)
        self.assertAlmostEqual(metrics.object_trajectory_length_m, 0.7)
        self.assertAlmostEqual(metrics.final_object_target_distance_m, 0.5)
        self.assertEqual(metrics.episode_return, 2.0)
        self.assertFalse(metrics.success)

    def test_rejects_empty_trajectory(self):
        with self.assertRaises(ValueError):
            measure_episode([], (0.0, 0.0, 0.0), 0.0, False)


if __name__ == "__main__":
    unittest.main()
