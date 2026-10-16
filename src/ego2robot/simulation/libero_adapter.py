"""LIBERO task adapter. All benchmark-specific imports stay in this module."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import yaml

from ego2robot.simulation.base import StepResult


class LiberoAdapter:
    """Load one LIBERO task and expose reset, step, and action limits."""

    def __init__(
        self,
        suite_name: str = "libero_object",
        task_id: int = 0,
        repo_root: Path | None = None,
        camera_size: int = 128,
    ) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]
        self.source_root = self.repo_root / ".libero-src"
        self._configure_paths()

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suites = benchmark.get_benchmark_dict()
        key = suite_name.lower()
        if key not in suites:
            raise ValueError(f"unknown LIBERO suite {suite_name!r}")
        self.suite = suites[key]()
        if task_id < 0 or task_id >= self.suite.get_num_tasks():
            raise ValueError(f"task_id {task_id} is out of range for {suite_name}")
        self.suite_name = key
        self.task_id = task_id
        self.task = self.suite.get_task(task_id)
        self.task_description = self.task.language
        bddl_path = (
            Path(get_libero_path("bddl_files"))
            / self.task.problem_folder
            / self.task.bddl_file
        )
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=camera_size,
            camera_widths=camera_size,
        )
        objects_of_interest = self.env.obj_of_interest
        object_name = (
            objects_of_interest[0]
            if isinstance(objects_of_interest, (list, tuple))
            else objects_of_interest
        )
        self.object_key = f"{object_name}_pos"
        self.initial_states = self.suite.get_task_init_states(task_id)
        low, high = self.env.env.action_spec
        self.action_low = tuple(float(value) for value in low)
        self.action_high = tuple(float(value) for value in high)

    def _configure_paths(self) -> None:
        """Set local source and asset paths without LIBERO's interactive prompt."""
        if not self.source_root.exists():
            raise FileNotFoundError(
                f"LIBERO source not found at {self.source_root}; see docs/setup.md"
            )
        source_root = str(self.source_root)
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        os.environ.setdefault("MUJOCO_GL", "egl")

        package_root = self.source_root / "libero" / "libero"
        config_dir = self.repo_root / ".libero-config"
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

    def reset(self, seed: int = 0, initial_state_id: int = 0) -> dict[str, Any]:
        """Reset LIBERO and apply the requested benchmark initial state."""
        if initial_state_id < 0 or initial_state_id >= len(self.initial_states):
            raise ValueError(
                f"initial_state_id {initial_state_id} is out of range "
                f"(0..{len(self.initial_states) - 1})"
            )
        self.env.seed(seed)
        self.env.reset()
        return self.env.set_init_state(self.initial_states[initial_state_id])

    def step(self, action: Sequence[float]) -> StepResult:
        """Advance LIBERO with one controller-space action."""
        observation, reward, done, info = self.env.step(
            np.asarray(action, dtype=np.float32)
        )
        return StepResult(
            observation=observation,
            reward=float(reward),
            done=bool(done),
            info=dict(info),
        )

    def check_success(self) -> bool:
        """Return LIBERO's task success predicate for the current state."""
        return bool(self.env.check_success())

    def close(self) -> None:
        """Close the underlying robosuite environment."""
        self.env.close()
