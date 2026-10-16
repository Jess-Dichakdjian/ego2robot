"""Create the README's paired phone/Panda visual from the saved MP4 files."""

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM = ROOT / "results" / "presentation" / "phone_demo_001_presentation.mp4"
DEFAULT_OUTPUT = ROOT / "results" / "presentation" / "phone_panda_demo.gif"
PANEL_W, PANEL_H, LABEL_H = 360, 203, 30
FRAME_COUNT = 36
FRAME_MS = 100


def read_normalized_frames(path: Path, count: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[Image.Image] = []
    for index in range(count):
        source_index = round(index * (total - 1) / max(1, count - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"cannot read frame {source_index} from {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((PANEL_W, PANEL_H), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (PANEL_W, PANEL_H), "#161b22")
        panel.paste(image, ((PANEL_W - image.width) // 2, (PANEL_H - image.height) // 2))
        frames.append(panel)
    capture.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phone-video", type=Path, required=True, help="Local source clip; read only")
    parser.add_argument("--simulation-video", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    phone = read_normalized_frames(args.phone_video, FRAME_COUNT)
    sim = read_normalized_frames(args.simulation_video, FRAME_COUNT)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    frames: list[Image.Image] = []
    for phone_frame, sim_frame in zip(phone, sim):
        canvas = Image.new("RGB", (2 * PANEL_W, PANEL_H + LABEL_H), "#0d1117")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), "PHONE TRIAL 01 · original footage", fill="#f0f6fc", font=font)
        draw.text((PANEL_W + 10, 7), "LIBERO SIMULATION · Panda", fill="#f0f6fc", font=font)
        canvas.paste(phone_frame, (0, LABEL_H))
        canvas.paste(sim_frame, (PANEL_W, LABEL_H))
        frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Saved {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
