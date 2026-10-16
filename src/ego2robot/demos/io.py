"""JSON serialization for canonical demonstrations."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import CanonicalDemo, DemoFrame


def write_demo(demo: CanonicalDemo, path: Path) -> None:
    """Write a demo as readable, deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(demo), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(payload + "\n", encoding="utf-8")


def read_demo(path: Path) -> CanonicalDemo:
    """Load and validate a canonical demo from JSON."""
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    frames = tuple(DemoFrame(**frame) for frame in raw.pop("frames"))
    return CanonicalDemo(frames=frames, **raw)
