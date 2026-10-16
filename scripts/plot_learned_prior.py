"""Render sparse held-out trajectory and keypoint prediction errors as SVG."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("results/learned_prior/loo_predictions.csv"))
    parser.add_argument("--demo", default="phone-demo-001")
    parser.add_argument("--demo-json", type=Path, default=Path("data/processed/phone-demo-001.json"))
    parser.add_argument("--output", type=Path, default=Path("results/figures/learned_sparse_prior.svg"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pred_path = args.predictions if args.predictions.is_absolute() else root / args.predictions
    output = args.output if args.output.is_absolute() else root / args.output
    with pred_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = [r for r in rows if r["trial"] == args.demo]
    if len(selected) != 3:
        raise SystemExit(f"Expected 3 intermediate points for {args.demo}; got {len(selected)}")
    phases = ["start", "approach", "grasp", "transport", "release"]
    # Endpoints are shared observed endpoints; intermediate estimates are
    # out-of-fold predictions, not extra annotations.
    first = selected[0]
    last = selected[-1]
    def xy(row, x, y): return (float(row[x]), float(row[y]))
    demo_path = args.demo_json if args.demo_json.is_absolute() else root / args.demo_json
    demo_data = json.loads(demo_path.read_text(encoding="utf-8"))
    endpoints = {f["task_phase"]: tuple(f["object_position"]) for f in demo_data["frames"]
                 if f.get("object_position") is not None}
    start, release = endpoints["start"], endpoints["release"]
    truth_points = [start] + [xy(r, "truth_x", "truth_y") for r in selected] + [release]
    straight_points = [start] + [xy(r, "straight_x", "straight_y") for r in selected] + [release]
    pred_points = [start] + [xy(r, "predicted_x", "predicted_y") for r in selected] + [release]
    colors = {"human": "#e4572e", "straight": "#5b6472", "learned": "#0077b6"}
    def screen(point): return (90 + point[0] * 500, 125 + (1 - point[1]) * 430)
    def poly(points, color, dash=""):
        coords = " ".join(f"{x:.1f},{y:.1f}" for x,y in map(screen, points))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"{d}/>'
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="650" viewBox="0 0 1120 650">
<rect width="1120" height="650" fill="#fff"/><style>text{font-family:Arial,sans-serif;fill:#202733}.title{font-size:25px;font-weight:bold}.sub{font-size:14px;fill:#586271}.label{font-size:13px}.axis{stroke:#aab3bf;stroke-width:1}</style>
<text class="title" x="35" y="38">A small learned residual predicts sparse human motion</text>
<text class="sub" x="35" y="63">18-trial leave-one-recording-out evaluation · five annotated points per trial · coordinates normalized to the phone image</text>
<text x="35" y="96" font-size="17" font-weight="bold">Held-out example: phone-demo-001</text>
<text class="sub" x="640" y="96">Mean held-out keypoint error by phase</text>
''']
    # Image-normalized axes retain the original top-left origin convention.
    for i in range(6):
        x = 90 + i * 100
        parts.append(f'<line class="axis" x1="{x}" y1="125" x2="{x}" y2="555"/><text class="sub" x="{x-12}" y="578">{i/5:.1f}</text>')
        y = 125 + i * 86
        parts.append(f'<line class="axis" x1="90" y1="{y}" x2="590" y2="{y}"/><text class="sub" x="45" y="{y+5}">{1-i/5:.1f}</text>')
    parts.append('<rect x="90" y="125" width="500" height="430" fill="none" stroke="#667085"/>')
    parts += [poly(truth_points, colors["human"]), poly(straight_points, colors["straight"], "7 6"), poly(pred_points, colors["learned"])]
    for i, p in enumerate(truth_points):
        sx, sy = screen(p)
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="{colors["human"]}" stroke="white" stroke-width="2"/><text class="label" x="{sx+8:.1f}" y="{sy-8:.1f}">{phases[i]}</text>')
        if 0 < i < 4:
            px, py = screen(pred_points[i])
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{colors["learned"]}" stroke="white" stroke-width="1.5"/>')
    for j, (name, color, dash) in enumerate((("Human annotation", colors["human"], ""), ("Straight interpolation", colors["straight"], "7 6"), ("LOO model prediction", colors["learned"], ""))):
        y=610
        parts.append(f'<line x1="{100+j*180}" y1="{y}" x2="{130+j*180}" y2="{y}" stroke="{color}" stroke-width="3"/><text class="label" x="{136+j*180}" y="{y+4}">{name}</text>')
    # Grouped horizontal bars of mean Euclidean error per phase over 18 trials.
    means=[]
    for phase in ("approach", "grasp", "transport"):
        pr=[r for r in rows if r["phase"]==phase]
        learned=sum(float(r["learned_error_norm"]) for r in pr)/len(pr)
        straight=sum(float(r["straight_error_norm"]) for r in pr)/len(pr)
        means.append((phase, learned, straight))
    scale=1050
    for i,(phase, learned, straight) in enumerate(means):
        yy=145+i*110
        parts.append(f'<text class="label" x="640" y="{yy}">{escape(phase)}</text>')
        parts.append(f'<rect x="730" y="{yy-23}" width="{learned*scale:.1f}" height="22" fill="#0077b6" rx="3"/><text class="label" x="{735+learned*scale:.1f}" y="{yy-7}">{learned:.3f}</text>')
        parts.append(f'<rect x="730" y="{yy+7}" width="{straight*scale:.1f}" height="22" fill="#aeb6c2" rx="3"/><text class="label" x="{735+straight*scale:.1f}" y="{yy+23}">{straight:.3f}</text>')
    parts += ['<rect x="730" y="490" width="12" height="12" fill="#0077b6"/><text class="label" x="748" y="501">LOO ridge model</text><rect x="870" y="490" width="12" height="12" fill="#aeb6c2"/><text class="label" x="888" y="501">Straight</text>',
              '<text class="sub" x="640" y="540">Lower is better. Bars average 18 held-out trials per phase.</text>',
              '<text class="sub" x="640" y="562">Example endpoints are retained; only 3 intermediate coordinates are predicted.</text>',
              '</svg>']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
