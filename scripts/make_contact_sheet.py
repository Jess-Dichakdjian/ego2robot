"""Build the labeled seven-frame evidence sheet from an evaluation video/log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


TRACE = re.compile(
    r"gripper_trace episode=0 step=(\d+) phase=(\w+).*?"
    r"actual_object_xyz=\[([^]]+)\].*?object_gripper_contacts=(\[[^]]*\])"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    steps: list[dict[str, object]] = []
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRACE.search(line)
        if not match:
            continue
        xyz = [float(value.strip()) for value in match.group(3).split(",")]
        steps.append({"step": int(match.group(1)), "phase": match.group(2),
                      "xyz": xyz, "contact": match.group(4) != "[]"})
    if not steps:
        raise SystemExit(f"No step traces found in {args.log}")

    def first(predicate):
        return next(item for item in steps if predicate(item))

    grasp_phase = next(i for i, item in enumerate(steps) if item["phase"] == "grasp")
    release_phase = next(i for i, item in enumerate(steps) if item["phase"] == "release")
    transport_indices = [i for i, item in enumerate(steps)
                         if item["phase"] == "transport" and i < release_phase]
    contact_step = first(lambda item: item["contact"])
    base_z = float(steps[0]["xyz"][2])
    lift_step = first(lambda item: int(item["step"]) >= int(contact_step["step"])
                      and float(item["xyz"][2]) > base_z + 0.035)
    choices = [
        ("start", steps[0]),
        ("approach", first(lambda item: item["phase"] == "approach")),
        ("grasp", contact_step),
        ("lift", lift_step),
        ("transport", steps[transport_indices[len(transport_indices) // 2]]),
        ("release", steps[release_phase]),
        ("success", steps[-1]),
    ]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    frames = []
    for label, item in choices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(item["step"]))
        ok, bgr = cap.read()
        if not ok:
            raise SystemExit(f"Could not read frame {item['step']} from {args.video}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(rgb)
        frame = frame.resize((320, 240), Image.Resampling.LANCZOS)
        frames.append((label, int(item["step"]), frame.copy()))
    cap.release()

    tile_w, tile_h, label_h = 320, 240, 32
    sheet = Image.new("RGB", (tile_w * 2, (tile_h + label_h) * 4), "#202020")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, step, frame) in enumerate(frames):
        x, y = (index % 2) * tile_w, (index // 2) * (tile_h + label_h)
        sheet.paste(frame, (x, y + label_h))
        draw.text((x + 8, y + 5), f"{label} | step {step}", fill="white", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Saved contact sheet: {args.output}")
    print(" ".join(f"{label}={step}" for label, step, _ in frames))


if __name__ == "__main__":
    main()
