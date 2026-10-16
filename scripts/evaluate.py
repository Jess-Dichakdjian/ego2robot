"""Run seeded baseline or demo-conditioned LIBERO episodes and save metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ego2robot.demos.io import read_demo
from ego2robot.evaluation.metrics import measure_episode
from ego2robot.policies import DemoConditionedPolicy, PickPlacePolicy
from ego2robot.retargeting import SimpleGeometricRetargeter
from ego2robot.simulation import LiberoAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("baseline", "demo"), default="baseline")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demo-json", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--post-release-steps",
        type=int,
        default=20,
        help="Maximum open-gripper steps to let a placed object settle before ending",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initial-state-id",
        type=int,
        default=0,
        help="LIBERO task initial-state index; with multiple episodes, increment from this index",
    )
    parser.add_argument("--output", type=Path, default=Path("results/runs/metrics.csv"))
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        help="Optional CSV of object and target positions at every evaluated step",
    )
    parser.add_argument(
        "--video-camera",
        choices=("agentview_image", "robot0_eye_in_hand_image"),
        default="agentview_image",
        help="Observation camera used only for optional video output",
    )
    parser.add_argument("--video", type=Path, help="Optional video path for the first episode")
    parser.add_argument("--debug-video", type=Path, help="Optional annotated debug video path for the first episode")
    parser.add_argument("--trace-contacts", action="store_true", help="Print MuJoCo contacts at trace steps")
    parser.add_argument("--trace-gripper", action="store_true", help="Log requested and actuated gripper commands and finger positions")
    args = parser.parse_args()
    if (
        args.episodes < 1
        or args.max_steps < 1
        or args.initial_state_id < 0
        or args.settle_steps < 0
        or args.post_release_steps < 0
    ):
        parser.error(
            "--episodes and --max-steps must be positive; --initial-state-id, "
            "--settle-steps and --post-release-steps cannot be negative"
        )
    if args.policy == "demo" and args.demo_json is None:
        parser.error("--demo-json is required for --policy demo")

    canonical_demo = read_demo(args.demo_json) if args.demo_json else None
    if args.task_id < 0 or args.task_id >= 10:
        parser.error("--task-id must be in the range 0..9 for libero_object")
    adapter = LiberoAdapter(task_id=args.task_id, camera_size=256 if args.video or args.debug_video else 64)
    object_key = adapter.object_key
    records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []
    video_writer = None
    debug_video_writer = None
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            initial_state_id = (
                args.initial_state_id + episode
            ) % len(adapter.initial_states)
            observation = adapter.reset(
                seed=seed, initial_state_id=initial_state_id
            )
            settle_action = np.asarray((0, 0, 0, 0, 0, 0, -1), dtype=np.float32)
            for _ in range(args.settle_steps):
                observation = adapter.step(settle_action).observation
            start = np.asarray(observation[object_key], dtype=float)
            target = np.asarray(observation["basket_1_pos"], dtype=float)
            if canonical_demo is None:
                policy = PickPlacePolicy(object_key=object_key)
            else:
                robot_demo = SimpleGeometricRetargeter().retarget(
                    canonical_demo,
                    tuple(start.tolist()),
                    tuple(target.tolist()),
                )
                if args.trace_gripper:
                    annotated_frames = [
                        frame for frame in canonical_demo.frames
                        if frame.object_position is not None
                    ]
                    print(
                        "retarget_model="
                        "endpoint_normalized_2d_to_sim_object_path; "
                        "camera_to_world_calibration=none; "
                        "image_depth=not_measured; "
                        f"sim_object_start_xyz={start.tolist()}; "
                        f"sim_basket_goal_xyz={target.tolist()}",
                        flush=True,
                    )
                    for index, (frame, waypoint) in enumerate(
                        zip(annotated_frames, robot_demo.waypoints)
                    ):
                        print(
                            f"waypoint_map index={index} phase={frame.task_phase} "
                            f"annotated_object_center_xy={frame.object_position} "
                            f"reconstructed_object_center_xyz={waypoint.object_position_xyz} "
                            f"transformed_sim_object_waypoint_xyz={waypoint.object_position_xyz}",
                            flush=True,
                        )
                policy = DemoConditionedPolicy(
                    robot_demo, object_key=object_key, target_key="basket_1_pos"
                )
            policy.reset()
            if args.trace_gripper:
                sim = adapter.env.env.sim
                gripper = adapter.env.env.robots[0].gripper
                gripper_actuator_ids = tuple(
                    sim.model.actuator_name2id(name) for name in gripper.actuators
                )
            positions = [start]
            if args.trajectory_output:
                initial_obj = np.asarray(observation[object_key], dtype=float)
                initial_target = np.asarray(observation["basket_1_pos"], dtype=float)
                initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=float)
                trajectory_records.append({
                    "policy": args.policy,
                    "episode": episode,
                    "seed": seed,
                    "initial_state_id": initial_state_id,
                    "step": 0,
                    "phase": "initial",
                    "object_x_m": initial_obj[0],
                    "object_y_m": initial_obj[1],
                    "object_z_m": initial_obj[2],
                    "target_x_m": initial_target[0],
                    "target_y_m": initial_target[1],
                    "target_z_m": initial_target[2],
                    "eef_x_m": initial_eef[0],
                    "eef_y_m": initial_eef[1],
                    "eef_z_m": initial_eef[2],
                })
            total_return = 0.0
            finished_steps = 0
            for step in range(args.max_steps):
                action = policy.act(observation)
                result = adapter.step(action)
                observation = result.observation
                if args.trace_gripper:
                    phase = getattr(policy, "last_task_phase", None) or getattr(policy, "phase", "unknown")
                    actuator_values = [
                        float(sim.data.ctrl[index]) for index in gripper_actuator_ids
                    ]
                    object_xyz = np.asarray(observation[object_key], dtype=float)
                    target_xyz = np.asarray(observation["basket_1_pos"], dtype=float)
                    eef_xyz = np.asarray(observation["robot0_eef_pos"], dtype=float)
                    object_contacts = sorted(
                        {
                            "{}<->{}".format(
                                sim.model.geom_id2name(sim.data.contact[i].geom1),
                                sim.model.geom_id2name(sim.data.contact[i].geom2),
                            )
                            for i in range(sim.data.ncon)
                            if object_key.removesuffix("_pos")
                            in (
                                sim.model.geom_id2name(sim.data.contact[i].geom1) or ""
                            ) + (
                                sim.model.geom_id2name(sim.data.contact[i].geom2) or ""
                            )
                            and "gripper0" in (
                                sim.model.geom_id2name(sim.data.contact[i].geom1) or ""
                            ) + (
                                sim.model.geom_id2name(sim.data.contact[i].geom2) or ""
                            )
                        }
                    )
                    print(
                        f"gripper_trace episode={episode} step={step} phase={phase} "
                        f"waypoint_index={getattr(policy, 'last_waypoint_index', None)} "
                        f"reconstructed_object_center_xyz={getattr(policy, 'last_object_waypoint_xyz', None)} "
                        f"requested_eef_target_xyz={getattr(policy, 'last_requested_eef_target', None)} "
                        f"actual_eef_xyz={np.round(observation['robot0_eef_pos'], 5).tolist()} "
                        f"actual_eef_quat_xyzw={np.round(observation['robot0_eef_quat'], 5).tolist()} "
                        f"actual_object_xyz={np.round(object_xyz, 5).tolist()} "
                        f"actual_target_xyz={np.round(target_xyz, 5).tolist()} "
                        f"object_eef_distance={float(np.linalg.norm(object_xyz - eef_xyz)):.5f} "
                        f"requested={float(action[-1]):+.3f} "
                        f"actuator_ctrl={actuator_values} "
                        f"qpos={np.round(observation['robot0_gripper_qpos'], 5).tolist()} "
                        f"object_gripper_contacts={object_contacts}",
                        flush=True,
                    )
                positions.append(np.asarray(observation[object_key], dtype=float))
                if args.trajectory_output:
                    object_position = np.asarray(observation[object_key], dtype=float)
                    target_position = np.asarray(observation["basket_1_pos"], dtype=float)
                    eef_position = np.asarray(observation["robot0_eef_pos"], dtype=float)
                    trajectory_records.append({
                        "policy": args.policy,
                        "episode": episode,
                        "seed": seed,
                        "initial_state_id": initial_state_id,
                        "step": step + 1,
                        "phase": (
                            getattr(policy, "last_task_phase", None)
                            or getattr(policy, "phase", "unknown")
                        ),
                        "object_x_m": object_position[0],
                        "object_y_m": object_position[1],
                        "object_z_m": object_position[2],
                        "target_x_m": target_position[0],
                        "target_y_m": target_position[1],
                        "target_z_m": target_position[2],
                        "eef_x_m": eef_position[0],
                        "eef_y_m": eef_position[1],
                        "eef_z_m": eef_position[2],
                    })
                total_return += result.reward
                if policy.finished:
                    finished_steps += 1
                if step % 50 == 0 or (
                    getattr(policy, "phase", None) == "grasp" and step % 10 == 0
                ):
                    contact_text = ""
                    if args.trace_contacts:
                        sim = adapter.env.env.sim
                        contacts = sorted(
                            {
                                "{}<->{}".format(
                                    sim.model.geom_id2name(sim.data.contact[i].geom1),
                                    sim.model.geom_id2name(sim.data.contact[i].geom2),
                                )
                                for i in range(sim.data.ncon)
                            }
                        )
                        contact_text = f" contacts={contacts}"
                    print(
                        f"episode={episode} step={step} phase={getattr(policy, 'phase', 'demo')} "
                        f"eef={np.round(observation['robot0_eef_pos'], 3).tolist()} "
                        f"object={np.round(observation[object_key], 3).tolist()} "
                        f"gripper={np.round(observation['robot0_gripper_qpos'], 3).tolist()} "
                        f"quat={np.round(observation['robot0_eef_quat'], 3).tolist()}"
                        f"{contact_text}",
                        flush=True,
                    )
                if (args.video or args.debug_video) and episode == 0:
                    import cv2

                    frame = observation[args.video_camera]
                    height, width = frame.shape[:2]
                    if args.video and video_writer is None:
                        video_path = args.video if args.video.is_absolute() else REPO_ROOT / args.video
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        video_writer = cv2.VideoWriter(
                            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError(f"Could not open video writer: {video_path}")
                    if video_writer is not None:
                        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if args.debug_video:
                        debug_path = args.debug_video if args.debug_video.is_absolute() else REPO_ROOT / args.debug_video
                        debug_path.parent.mkdir(parents=True, exist_ok=True)
                        if debug_video_writer is None:
                            debug_video_writer = cv2.VideoWriter(
                                str(debug_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
                            )
                            if not debug_video_writer.isOpened():
                                raise RuntimeError(f"Could not open debug video writer: {debug_path}")
                        debug_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        phase = getattr(policy, "last_task_phase", None) or getattr(policy, "phase", "unknown")
                        object_now = np.asarray(observation[object_key], dtype=float)
                        target_now = np.asarray(observation["basket_1_pos"], dtype=float)
                        debug_lines = (
                            f"phone_demo_001 | step {step} | phase {phase}",
                            f"object-target distance: {np.linalg.norm(object_now - target_now):.3f} m",
                            f"camera: {args.video_camera}",
                        )
                        for line_index, line in enumerate(debug_lines):
                            cv2.putText(debug_frame, line, (5, 15 + 14 * line_index),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1,
                                        cv2.LINE_AA)
                        debug_video_writer.write(debug_frame)
                # LIBERO's containment predicate can become true while the
                # object is still held above the basket. Only accept success
                # after the policy has opened at release and the settling
                # window has elapsed.
                if policy.finished and finished_steps >= args.post_release_steps:
                    break
            metrics = measure_episode(
                positions, target, total_return, adapter.check_success()
            )
            records.append({
                "policy": args.policy,
                "seed": seed,
                "initial_state_id": initial_state_id,
                **metrics.to_dict(),
            })
            print(records[-1])
    finally:
        if video_writer is not None:
            video_writer.release()
        if debug_video_writer is not None:
            debug_video_writer.release()
        adapter.close()

    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved metrics: {output}")
    if args.trajectory_output:
        trajectory_output = (
            args.trajectory_output
            if args.trajectory_output.is_absolute()
            else REPO_ROOT / args.trajectory_output
        )
        trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        with trajectory_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=trajectory_records[0].keys())
            writer.writeheader()
            writer.writerows(trajectory_records)
        print(f"Saved trajectory: {trajectory_output}")


if __name__ == "__main__":
    main()
