"""Tests for bounded Cartesian policy outputs and demo conditioning."""

import unittest

import numpy as np

from ego2robot.policies import DemoConditionedPolicy, PickPlacePolicy
from ego2robot.retargeting.base import RobotDemo, RobotWaypoint


def observation(eef=(0.0, 0.0, 0.2), quat=(0.0, 0.0, 0.0, 1.0)):
    return {
        "robot0_eef_pos": np.asarray(eef, dtype=np.float32),
        "robot0_eef_quat": np.asarray(quat, dtype=np.float32),
        "alphabet_soup_1_pos": np.asarray((-0.1, -0.2, 0.02), dtype=np.float32),
        "basket_1_pos": np.asarray((0.1, 0.2, 0.03), dtype=np.float32),
    }


class PolicyTests(unittest.TestCase):
    def test_baseline_outputs_bounded_seven_dimensional_action(self):
        policy = PickPlacePolicy(max_action=0.03)
        action = policy.act(observation())
        self.assertEqual(action.shape, (7,))
        self.assertTrue(np.isfinite(action).all())
        self.assertLessEqual(float(np.max(np.abs(action[:3]))), 0.03)
        self.assertEqual(float(action[6]), -1.0)

    def test_baseline_rotation_uses_xyzw_and_holds_position(self):
        policy = PickPlacePolicy()
        policy.phase_index = policy.phases.index("orient")
        first = observation()
        action = policy.act(first)
        self.assertGreater(float(action[3]), 0.9)
        moved = observation(eef=(0.05, 0.0, 0.2))
        correction = policy.act(moved)
        self.assertLess(float(correction[0]), 0.0)

    def test_demo_policy_uses_retargeted_waypoint_and_gripper_annotation(self):
        demo = RobotDemo(
            demo_id="demo",
            task="pick and place",
            source="test",
            coordinate_frame="robot_world_m",
            waypoints=(
                RobotWaypoint(0.0, (0.0, 0.0, 0.2), gripper_state=1.0),
            ),
        )
        policy = DemoConditionedPolicy(demo, tolerance=0.001)
        rotation_action = policy.act(observation())
        self.assertAlmostEqual(float(rotation_action[3]), 0.0)
        target_xyzw = policy.target_quaternion[[1, 2, 3, 0]].astype(np.float32)
        action = policy.act(
            observation(quat=target_xyzw)
        )
        self.assertEqual(action.shape, (7,))
        self.assertEqual(float(action[6]), 1.0)
        np.testing.assert_allclose(action[:3], (0.0, 0.0, 0.1), atol=1e-7)

    def test_demo_policy_infers_close_at_grasp_without_gripper_annotation(self):
        demo = RobotDemo(
            demo_id="demo",
            task="pick and place",
            source="test",
            coordinate_frame="robot_world_m",
            waypoints=(
                RobotWaypoint(1.0, (0.0, 0.0, 0.2), task_phase="grasp"),
            ),
        )
        policy = DemoConditionedPolicy(demo)
        policy.orientation_ready = True
        obs = observation(eef=(0.0, 0.0, 0.16))
        obs["alphabet_soup_1_pos"] = np.asarray((0.0, 0.0, 0.14), dtype=np.float32)

        action = policy.act(obs)

        self.assertEqual(float(action[6]), 1.0)

    def test_demo_policy_holds_gripper_closed_until_release_target(self):
        demo = RobotDemo(
            demo_id="demo",
            task="pick and place",
            source="test",
            coordinate_frame="robot_world_m",
            waypoints=(RobotWaypoint(1.0, (0.0, 0.0, 0.02), task_phase="release"),),
        )
        policy = DemoConditionedPolicy(demo)
        policy.orientation_ready = True

        action = policy.act(observation())

        self.assertEqual(float(action[6]), 1.0)

    def test_demo_policy_anchors_release_to_initial_target_pose(self):
        demo = RobotDemo(
            demo_id="demo",
            task="pick and place",
            source="test",
            coordinate_frame="robot_world_m",
            waypoints=(RobotWaypoint(1.0, (0.0, 0.0, 0.02), task_phase="release"),),
        )
        policy = DemoConditionedPolicy(demo)
        policy.orientation_ready = True
        obs = observation()

        policy.act(obs)
        obs["basket_1_pos"] = np.asarray((0.6, 0.7, 0.03), dtype=np.float32)
        policy.act(obs)

        np.testing.assert_allclose(
            policy.last_requested_eef_target, (0.1, 0.2, 0.26), atol=1e-7
        )

    def test_demo_policy_compensates_release_for_carried_object_offset(self):
        demo = RobotDemo(
            demo_id="demo",
            task="pick and place",
            source="test",
            coordinate_frame="robot_world_m",
            waypoints=(RobotWaypoint(1.0, (0.0, 0.0, 0.02), task_phase="release"),),
        )
        policy = DemoConditionedPolicy(demo)
        policy.initial_object_position = np.asarray((0.0, 0.0, 0.02), dtype=np.float32)
        policy.initial_target_position = np.asarray((0.1, 0.2, 0.03), dtype=np.float32)
        policy.grasp_object_to_eef_offset_xy = np.asarray((0.01, -0.02), dtype=np.float32)
        policy.release_lowered = True

        policy._eef_target(demo.waypoints[0], 0, observation())

        np.testing.assert_allclose(
            policy.last_requested_eef_target, (0.09, 0.22, 0.10), atol=1e-7
        )

    def test_demo_policy_release_lowers_only_after_xy_alignment(self):
        demo = RobotDemo(
            demo_id="demo",
            task="pick and place",
            source="test",
            coordinate_frame="robot_world_m",
            waypoints=(RobotWaypoint(1.0, (0.0, 0.0, 0.02), task_phase="release"),),
        )
        policy = DemoConditionedPolicy(demo)
        policy.orientation_ready = True
        policy.initial_object_position = np.asarray((0.0, 0.0, 0.02), dtype=np.float32)
        policy.initial_target_position = np.asarray((0.1, 0.2, 0.03), dtype=np.float32)

        policy._eef_target(demo.waypoints[0], 0, observation())

        self.assertAlmostEqual(policy.last_requested_eef_target[2], 0.26, places=6)
        self.assertFalse(policy.release_lowered)

    def test_demo_policy_uses_hover_target_for_start_and_approach(self):
        demo = RobotDemo(
            demo_id="demo",
            task="pick and place",
            source="test",
            coordinate_frame="robot_world_m",
            waypoints=(
                RobotWaypoint(0.0, (0.0, 0.0, 0.04), task_phase="start"),
                RobotWaypoint(1.0, (0.1, 0.0, 0.04), task_phase="approach"),
            ),
        )
        policy = DemoConditionedPolicy(demo)
        policy.orientation_ready = True
        obs = observation()
        obs["alphabet_soup_1_pos"] = np.asarray((0.0, 0.0, 0.04), dtype=np.float32)

        policy.act(obs)

        self.assertAlmostEqual(policy.last_requested_eef_target[2], 0.16)

    def test_demo_policy_does_not_skip_nearby_approach_phase(self):
        demo = RobotDemo(
            demo_id="demo",
            task="pick and place",
            source="test",
            coordinate_frame="robot_world_m",
            waypoints=(
                RobotWaypoint(0.0, (0.0, 0.0, 0.04), task_phase="start"),
                RobotWaypoint(1.0, (0.0, 0.0, 0.04), task_phase="approach"),
                RobotWaypoint(2.0, (0.0, 0.0, 0.04), task_phase="grasp"),
            ),
        )
        policy = DemoConditionedPolicy(demo)
        policy.orientation_ready = True
        obs = observation(eef=(0.0, 0.0, 0.16))
        obs["alphabet_soup_1_pos"] = np.asarray((0.0, 0.0, 0.04), dtype=np.float32)

        policy.act(obs)
        self.assertEqual(policy.last_task_phase, "approach")
        approach_obs = observation(eef=(-0.08, 0.0, 0.16))
        approach_obs["alphabet_soup_1_pos"] = obs["alphabet_soup_1_pos"]
        policy.act(approach_obs)
        self.assertEqual(policy.last_task_phase, "grasp")

    def test_demo_policy_rejects_wrong_coordinate_frame(self):
        demo = RobotDemo("demo", "task", "test", "image_normalized", (
            RobotWaypoint(0.0, (0.0, 0.0, 0.0)),
        ))
        with self.assertRaises(ValueError):
            DemoConditionedPolicy(demo)


if __name__ == "__main__":
    unittest.main()
