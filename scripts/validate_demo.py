"""Validate a serialized canonical demo and print a short summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego2robot.demos.io import read_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_json", type=Path)
    path = parser.parse_args().demo_json
    demo = read_demo(path)
    print(
        f"Valid demo {demo.demo_id!r}: {len(demo.frames)} frames, "
        f"task={demo.task!r}, coordinate_frame={demo.coordinate_frame!r}"
    )


if __name__ == "__main__":
    main()
