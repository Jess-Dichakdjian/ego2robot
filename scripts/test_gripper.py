"""Exercise open/close gripper commands and record simulator actuation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ego2robot.simulation import LiberoAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-command", type=int, default=30)
    parser.add_argument(
        "--output", type=Path, default=Path("results/runs/gripper_diagnostic.csv")
    )
    args = parser.parse_args()
    if args.steps_per_command < 1:
        parser.error("--steps-per-command must be positive")

    adapter = LiberoAdapter(task_id=args.task_id)
    try:
        observation = adapter.reset(seed=args.seed, initial_state_id=0)
        sim = adapter.env.env.sim
        robot = adapter.env.env.robots[0]
        actuator_names = tuple(robot.gripper.actuators)
        actuator_ids = tuple(sim.model.actuator_name2id(name) for name in actuator_names)
        commands = (("open", -1.0), ("close", 1.0), ("open_again", -1.0))
        output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "stage", "step", "requested_gripper_command",
            "executed_actuator_command", "actual_gripper_qpos",
        )
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for stage, command in commands:
                for step in range(args.steps_per_command):
                    action = np.asarray((0, 0, 0, 0, 0, 0, command), dtype=np.float32)
                    observation = adapter.step(action).observation
                    actuator_values = [float(sim.data.ctrl[index]) for index in actuator_ids]
                    gripper_positions = [
                        float(value) for value in observation["robot0_gripper_qpos"]
                    ]
                    writer.writerow({
                        "stage": stage,
                        "step": step,
                        "requested_gripper_command": float(action[-1]),
                        "executed_actuator_command": actuator_values,
                        "actual_gripper_qpos": gripper_positions,
                    })
                print(
                    f"stage={stage} requested={command:+.1f} "
                    f"actuator_ctrl={[float(sim.data.ctrl[index]) for index in actuator_ids]} "
                    f"gripper_qpos={[float(value) for value in observation['robot0_gripper_qpos']]}",
                    flush=True,
                )
        print(f"Saved gripper trace: {output_path}")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
