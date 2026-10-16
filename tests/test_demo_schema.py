"""Serialization and validation checks for the canonical demo schema."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ego2robot.demos.io import read_demo, write_demo
from ego2robot.demos.schema import CanonicalDemo, DemoFrame


class DemoSchemaTests(unittest.TestCase):
    def sample_demo(self) -> CanonicalDemo:
        return CanonicalDemo(
            demo_id="schema-example-001",
            task="move a block from start to target",
            coordinate_frame="image_normalized",
            source="synthetic_example",
            metadata={"is_synthetic": True},
            frames=(
                DemoFrame(
                    timestamp_s=0.0,
                    object_position=(0.25, 0.70),
                    task_phase="start",
                    source="synthetic_keyframe",
                    confidence=1.0,
                ),
                DemoFrame(
                    timestamp_s=1.0,
                    object_position=(0.72, 0.34),
                    gripper_state=0.0,
                    task_phase="release",
                    source="synthetic_keyframe",
                    confidence=1.0,
                ),
            ),
        )

    def test_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.json"
            demo = self.sample_demo()
            write_demo(demo, path)
            self.assertEqual(read_demo(path), demo)

    def test_committed_example_loads(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data/processed/example_demo.json"
        demo = read_demo(path)
        self.assertEqual(demo.source, "synthetic_example")
        self.assertEqual(len(demo.frames), 2)

    def test_timestamps_must_increase(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            CanonicalDemo(
                demo_id="bad-time",
                task="pick and place",
                coordinate_frame="image_normalized",
                source="test",
                frames=(DemoFrame(timestamp_s=1.0), DemoFrame(timestamp_s=1.0)),
            )

    def test_image_coordinates_must_be_normalized(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalized 2D"):
            CanonicalDemo(
                demo_id="bad-coordinate",
                task="pick and place",
                coordinate_frame="image_normalized",
                source="test",
                frames=(
                    DemoFrame(timestamp_s=0.0, object_position=(1.2, 0.5)),
                ),
            )


if __name__ == "__main__":
    unittest.main()
