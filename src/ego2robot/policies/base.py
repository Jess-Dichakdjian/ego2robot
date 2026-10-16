"""Small policy interface shared by scripted and demo-conditioned control."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    """A policy maps observations to normalized LIBERO controller actions."""

    def reset(self) -> None:
        """Reset per-episode state."""

    def act(
        self, observation: dict[str, Any], optional_context: Any | None = None
    ) -> np.ndarray:
        """Return a seven-dimensional OSC pose plus gripper action."""
