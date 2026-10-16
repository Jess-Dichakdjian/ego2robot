"""Simulation adapter contract shared by interchangeable backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class StepResult:
    """One simulator transition with raw observations kept intact."""

    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class SimulationAdapter(Protocol):
    """Minimal interface consumed by policies and evaluators."""

    def reset(self, seed: int = 0, initial_state_id: int = 0) -> dict[str, Any]:
        """Reset the task and return the first observation."""

    def step(self, action: Sequence[float]) -> StepResult:
        """Advance one control step."""

    def close(self) -> None:
        """Release simulator resources."""
