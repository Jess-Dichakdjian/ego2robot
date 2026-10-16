import json
import tempfile
import unittest
from pathlib import Path

from ego2robot.demos.interaction import load_interaction
from ego2robot.demos.approach_model import ApproachExample, RidgeApproachModel, fit_ridge
from ego2robot.policies.interaction_approach import apply_interaction_approach_prior
from scripts.audit_interaction_dataset import audit


def interaction_document(hand_point=(120.0, 80.0), kind="index_fingertip"):
    width, height = 1920, 1080
    return {
        "annotation_type": "ego2robot_human_object_interaction",
        "schema_version": 1,
        "demo_id": "phone-demo-010",
        "video_filename": "trial_10.mp4",
        "coordinate_frame": "image_pixels_top_left",
        "image_width_px": width,
        "image_height_px": height,
        "events": [{
            "timestamp_s": 0.4,
            "event_type": "approach",
            "object_center_px": [100.0, 100.0],
            "object_center_norm": [100 / width, 100 / height],
            "hand_point_px": list(hand_point) if hand_point is not None else None,
            "hand_point_norm": [hand_point[0] / width, hand_point[1] / height] if hand_point is not None else None,
            "hand_point_kind": kind if hand_point is not None else None,
            "hand_visibility": "visible" if hand_point is not None else "uncertain",
            "confidence": 0.8,
        }],
    }


class InteractionDatasetTests(unittest.TestCase):
    def test_imported_dataset_matches_canonical_points_without_filling_missing_hands(self):
        root = Path(__file__).resolve().parents[1]
        result = audit(root / "data/interaction_annotations", root / "data/processed")
        self.assertEqual(result["interaction_file_count"], 12)
        self.assertEqual(result["valid_trial_count"], 12)
        self.assertEqual(result["event_count"], 70)
        self.assertEqual(result["canonical_event_matches"], 70)
        self.assertEqual(result["valid_hand_to_can_displacement_count"], 28)
        self.assertEqual(result["exact_duplicate_hand_object_event_count"], 0)
        self.assertEqual(result["issues"], [])

    def test_keeps_missing_points_and_anatomical_kind(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.json"
            doc = interaction_document((120.0, 80.0), "other_fingertip")
            path.write_text(json.dumps(doc), encoding="utf-8")
            _, events = load_interaction(path)
            self.assertEqual(events[0].hand_point_kind, "other_fingertip")
            self.assertEqual(events[0].hand_to_object_px, (20.0, -20.0))
            doc["events"][0].update(hand_point_px=None, hand_point_norm=None, hand_point_kind=None, hand_visibility="occluded")
            path.write_text(json.dumps(doc), encoding="utf-8")
            _, missing = load_interaction(path)
            self.assertIsNone(missing[0].hand_point_norm)
            self.assertIsNone(missing[0].hand_to_object_px)

    def test_rejects_normalized_coordinates_that_disagree_with_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.json"
            doc = interaction_document()
            doc["events"][0]["hand_point_norm"] = [0.5, 0.5]
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pixel/normalized mismatch"):
                load_interaction(path)

    def test_rejects_hand_point_without_landmark_kind(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.json"
            doc = interaction_document(kind=None)
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "anatomical kind"):
                load_interaction(path)

    def test_rejects_duplicate_or_nonchronological_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.json"
            doc = interaction_document()
            doc["events"].append({**doc["events"][0]})
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_interaction(path)

    def test_ridge_approach_model_predicts_finite_unit_direction(self):
        rows = tuple(
            ApproachExample(f"demo-{i}", (i / 10, -i / 20), (0.2 + i / 100, 0.98), float(i))
            for i in range(5)
        )
        model = fit_ridge(rows, alpha=1.0)
        prediction = model.predict((0.0, 0.0))
        self.assertAlmostEqual(sum(value * value for value in prediction), 1.0, places=6)
        restored = RidgeApproachModel.from_dict(model.to_dict())
        self.assertAlmostEqual(restored.predict((0.0, 0.0))[0], prediction[0], places=8)
        with self.assertRaisesRegex(ValueError, "three approach examples"):
            fit_ridge(rows[:2])

    def test_approach_prior_maps_human_side_to_policy_direction(self):
        rows = tuple(
            ApproachExample(f"demo-{i}", (float(i), 0.0), (0.0, 1.0), float(i))
            for i in range(4)
        )
        model = fit_ridge(rows)

        class Policy:
            approach_direction_xy = (1.0, 0.0)

        policy = Policy()
        info = apply_interaction_approach_prior(policy, model, (0, 0, 0), (1, 0, 0))
        self.assertAlmostEqual(policy.approach_direction_xy[0], 0.0, places=6)
        self.assertAlmostEqual(policy.approach_direction_xy[1], -1.0, places=6)
        self.assertEqual(info["model_features"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
