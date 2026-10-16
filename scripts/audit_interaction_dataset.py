"""Audit supplementary hand/object annotations against canonical labels and clips."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from ego2robot.demos.interaction import load_interaction


def audit(interaction_dir: Path, canonical_dir: Path, video_dir: Path | None = None) -> dict:
    files = sorted(interaction_dir.glob("phone-demo-*-interaction.json"))
    trials: list[dict] = []
    issues: list[dict[str, str]] = []
    events_total = points_total = exact_matches = 0
    phase_counts: Counter[str] = Counter()
    hand_phase_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    visibility_counts: Counter[str] = Counter()
    annotation_source_counts: Counter[str] = Counter()
    confidence_values: list[float] = []
    all_video_dims: set[tuple[int, int]] = set()
    hand_object_signatures: list[tuple] = []
    stale_hand_source_count = 0
    session_note = "All provided clips belong to the same original concatenated recording session."

    for path in files:
        try:
            raw, events = load_interaction(path)
            canonical_path = canonical_dir / f"{raw['demo_id']}.json"
            canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
            if raw.get("demo_id") != canonical.get("demo_id"):
                raise ValueError("interaction demo_id does not match canonical demo")
            if raw.get("video_filename") != f"trial_{int(raw['demo_id'][-3:]):02d}.mp4":
                raise ValueError("interaction filename does not follow trial ID")
            if canonical.get("coordinate_frame") != "image_normalized":
                raise ValueError("canonical demonstration is not image_normalized")
            canonical_frames = canonical.get("frames", [])
            canonical_by_time = {round(float(f["timestamp_s"]), 3): f for f in canonical_frames}
            exact_trial_matches = 0
            event_labels: list[str] = []
            hand_events = 0
            vectors: list[dict] = []
            for event in events:
                key = round(event.timestamp_s, 3)
                source_frame = canonical_by_time.get(key)
                if source_frame is None:
                    raise ValueError(f"timestamp {event.timestamp_s:.3f}s has no canonical match")
                can = source_frame.get("object_position")
                if not can or math.dist(can, event.object_center_norm) > 0.002:
                    raise ValueError(f"can centre differs from canonical at {event.timestamp_s:.3f}s")
                exact_trial_matches += 1
                canonical_label = str(source_frame.get("task_phase") or source_frame.get("event_type") or "uncertain_event")
                label = event.event_type
                # Earlier annotator versions could not import the combined canonical failure label.
                if label == "uncertain_event" and canonical_label == "failed_contact/push":
                    label = canonical_label
                event_labels.append(label)
                phase_counts[label] += 1
                visibility_counts[event.hand_visibility] += 1
                confidence_values.append(event.confidence)
                source_value = str(next(
                    item.get("annotation_source", "unspecified")
                    for item in raw["events"]
                    if round(float(item["timestamp_s"]), 3) == key
                ))
                annotation_source_counts[source_value] += 1
                if event.hand_point_px is not None:
                    hand_events += 1
                    if source_value == "imported_canonical_object_only":
                        stale_hand_source_count += 1
                    hand_object_signatures.append((
                        event.event_type,
                        tuple(round(value, 1) for value in event.object_center_px),
                        tuple(round(value, 1) for value in event.hand_point_px),
                        event.hand_point_kind,
                    ))
                    kind = str(event.hand_point_kind)
                    kind_counts[kind] += 1
                    hand_phase_counts[label] += 1
                    displacement = event.hand_to_object_px
                    assert displacement is not None
                    vectors.append({
                        "timestamp_s": event.timestamp_s,
                        "event_type": label,
                        "hand_point_kind": kind,
                        "hand_to_object_px": [round(displacement[0], 2), round(displacement[1], 2)],
                        "hand_to_object_norm": [round(displacement[0] / event.image_width_px, 6), round(displacement[1] / event.image_height_px, 6)],
                        "confidence": event.confidence,
                    })
            video_found = video_dir is not None and (video_dir / raw["video_filename"]).is_file()
            video_metadata = None
            if video_found:
                try:
                    import cv2  # Optional dependency, only required for frame-level clip metadata.
                    capture = cv2.VideoCapture(str(video_dir / raw["video_filename"]))
                    if not capture.isOpened():
                        raise ValueError("matching video exists but cannot be opened")
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = float(capture.get(cv2.CAP_PROP_FPS))
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    capture.release()
                    video_metadata = {"width_px": width, "height_px": height, "fps": fps, "frame_count": frame_count}
                    if (width, height) != (event.image_width_px, event.image_height_px):
                        raise ValueError("interaction dimensions do not match the named video")
                    all_video_dims.add((width, height))
                except ImportError:
                    video_metadata = {"error": "opencv_unavailable; only filename was checked"}

            events_total += len(events)
            points_total += hand_events
            exact_matches += exact_trial_matches
            trials.append({
                "demo_id": raw["demo_id"],
                "video_filename": raw["video_filename"],
                "video_exists": bool(video_found),
                "video_metadata": video_metadata,
                "event_count": len(events),
                "hand_point_count": hand_events,
                "missing_hand_count": len(events) - hand_events,
                "canonical_timestamp_and_can_matches": exact_trial_matches,
                "canonical_event_labels": event_labels,
                "event_type_counts": dict(Counter(event_labels)),
                "hand_landmark_kinds": dict(Counter(e.hand_point_kind for e in events if e.hand_point_kind)),
                "hand_to_can_vectors": vectors,
            })
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append({"file": path.name, "issue": str(exc)})

    recovery_trials = []
    for trial in trials:
        labels = trial["canonical_event_labels"]
        if "failed_contact/push" in labels and "realign" in labels:
            recovery_trials.append(trial["demo_id"])
    return {
        "interaction_file_count": len(files),
        "valid_trial_count": len(trials),
        "event_count": events_total,
        "hand_landmark_count": points_total,
        "valid_hand_to_can_displacement_count": points_total,
        "canonical_event_matches": exact_matches,
        "visibility_counts": dict(visibility_counts),
        "landmark_kind_counts": dict(kind_counts),
        "annotation_source_counts": dict(annotation_source_counts),
        "hand_point_source_metadata_stale_count": stale_hand_source_count,
        "exact_duplicate_hand_object_event_count": len(hand_object_signatures) - len(set(hand_object_signatures)),
        "event_type_counts": dict(phase_counts),
        "hand_point_event_type_counts": dict(hand_phase_counts),
        "confidence_range": [min(confidence_values), max(confidence_values)] if confidence_values else None,
        "video_dimensions_seen": [list(dim) for dim in sorted(all_video_dims)],
        "recovery_trials_with_failure_and_realign_labels": recovery_trials,
        "split_unit": "demo_id/trial; never split events from the same clip across train and test",
        "recording_session_note": session_note,
        "trials": trials,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction-dir", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.interaction_dir, args.canonical_dir, args.video_dir)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
