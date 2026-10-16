"""Run the official LIBERO task-load, reset, and step smoke test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml


def configure_libero(repo_root: Path) -> None:
    """Point LIBERO at the checked-out benchmark assets without prompting."""
    package_root = repo_root / ".libero-src" / "libero" / "libero"
    config_dir = repo_root / ".libero-config"
    config_dir.mkdir(exist_ok=True)
    config = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(package_root.parent / "datasets"),
        "assets": str(package_root / "assets"),
    }
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    repo_root = Path(__file__).resolve().parents[1]
    configure_libero(repo_root)
    sys.path.insert(0, str(repo_root / ".libero-src"))

    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    video_writer = None

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    bddl_file = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    print(f"Task: {task.name} — {task.language}")
    print(f"BDDL: {bddl_file}")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file), camera_heights=128, camera_widths=128
    )
    try:
        env.seed(args.seed)
        observation = env.reset()
        initial_states = suite.get_task_init_states(args.task_id)
        env.set_init_state(initial_states[0])
        print("Observation:")
        for key, value in observation.items():
            shape = getattr(value, "shape", None)
            print(f"  {key}: {shape if shape is not None else type(value).__name__}")

        rng = np.random.default_rng(args.seed)
        action_low, action_high = env.env.action_spec
        for step in range(args.steps):
            if args.random_actions:
                action = rng.uniform(action_low, action_high).astype(np.float32)
            else:
                action = np.zeros(7)
            observation, reward, done, _info = env.step(action)
            if args.video:
                import cv2

                frame = observation["agentview_image"]
                if video_writer is None:
                    video_path = args.video
                    if not video_path.is_absolute():
                        video_path = repo_root / video_path
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0,
                        (width, height),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Could not open video writer: {video_path}")
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if done:
                break
        print(f"Stepped: {step + 1}, reward: {reward}, done: {done}")
        if args.video:
            print(f"Video: {video_path} ({step + 1} frames at 20 fps)")
    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()


if __name__ == "__main__":
    main()
