"""Fit and trial-level leave-one-out evaluate a small human motion prior.

Only annotated intermediate keyframes are targets. This predicts sparse
keyframes, not a continuous policy or new independent ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np

PHASES = ("approach", "grasp", "transport")


def load_trials(root: Path):
    trials = []
    for path in sorted(root.glob("phone-demo-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = [f for f in data["frames"] if f.get("object_position") is not None]
        phases = {f["task_phase"]: f for f in frames}
        if not all(p in phases for p in ("start", *PHASES, "release")):
            continue
        if len(frames) != 5 or len({f["task_phase"] for f in frames}) != 5:
            continue
        trials.append({"path": path, "data": data, "frames": phases})
    return trials


def feature(trial, phase):
    f = trial["frames"]
    start = np.asarray(f["start"]["object_position"], dtype=float)
    end = np.asarray(f["release"]["object_position"], dtype=float)
    ts, te = float(f["start"]["timestamp_s"]), float(f["release"]["timestamp_s"])
    t = (float(f[phase]["timestamp_s"]) - ts) / max(te - ts, 1e-8)
    onehot = np.asarray([float(p == phase) for p in PHASES])
    x = np.concatenate(([1.0], onehot, [t], start, end - start))
    base = start + t * (end - start)
    y = np.asarray(f[phase]["object_position"], dtype=float) - base
    return x, y, base, t


def fit(x, y, alpha=0.03):
    mu, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    # Keep the explicit intercept column equal to one after normalization.
    mu[0], scale[0] = 0.0, 1.0
    z = (x - mu) / scale
    # Do not penalize the intercept.
    penalty = np.eye(z.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return mu, scale, coef


def predict(model, x):
    mu, scale, coef = model
    return ((x - mu) / scale) @ coef


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/learned_prior"))
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--export-heldout-demo", help="Also write this trial's leave-one-out predicted route JSON")
    args = parser.parse_args()
    root = args.data_root.resolve()
    out = args.out if args.out.is_absolute() else Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)
    trials = load_trials(root)
    if len(trials) < 5:
        raise SystemExit(f"Need >=5 eligible trials; found {len(trials)}")
    rows = []
    heldout_errors, baseline_errors, train_errors = [], [], []
    per_trial = []
    all_x, all_y = [], []
    for trial in trials:
        for phase in PHASES:
            x, y, _, _ = feature(trial, phase)
            all_x.append(x); all_y.append(y)
    for heldout_i, test in enumerate(trials):
        train_x, train_y = [], []
        for j, other in enumerate(trials):
            if j == heldout_i:
                continue
            for phase in PHASES:
                x, y, _, _ = feature(other, phase)
                train_x.append(x); train_y.append(y)
        model = fit(np.asarray(train_x), np.asarray(train_y), args.alpha)
        errors_t, errors_b = [], []
        for phase in PHASES:
            x, _, base, _ = feature(test, phase)
            pred_resid = predict(model, x)
            pred = np.clip(base + pred_resid, 0.0, 1.0)
            truth = np.asarray(test["frames"][phase]["object_position"], dtype=float)
            pred_err = float(np.linalg.norm(pred - truth))
            base_err = float(np.linalg.norm(base - truth))
            heldout_errors.append(pred_err); baseline_errors.append(base_err)
            errors_t.append(pred_err); errors_b.append(base_err)
            rows.append({"trial": test["data"]["demo_id"], "phase": phase,
                         "timestamp_s": test["frames"][phase]["timestamp_s"],
                         "truth_x": truth[0], "truth_y": truth[1],
                         "straight_x": base[0], "straight_y": base[1],
                         "predicted_x": pred[0], "predicted_y": pred[1],
                         "learned_error_norm": pred_err, "straight_error_norm": base_err})
        per_trial.append({"trial": test["data"]["demo_id"],
                          "learned_mean_error": float(np.mean(errors_t)),
                          "straight_mean_error": float(np.mean(errors_b))})
    trial_deltas = np.asarray([r["straight_mean_error"] - r["learned_mean_error"] for r in per_trial])
    rng = np.random.default_rng(20261016)
    bootstrap_means = np.mean(rng.choice(trial_deltas, size=(10000, len(trial_deltas)), replace=True), axis=1)
    delta_ci = np.quantile(bootstrap_means, [0.025, 0.975])
    # Training fit reported separately; this is resubstitution, not validation.
    model_all = fit(np.asarray(all_x), np.asarray(all_y), args.alpha)
    for x, y in zip(all_x, all_y):
        train_errors.append(float(np.linalg.norm(predict(model_all, x) - y)))
    summary = {
        "method": "ridge residual predictor; trial-level leave-one-out",
        "annotated_trial_ids": len(trials), "source_recording_sessions": 1,
        "annotated_keypoints_per_trial": 5,
        "training_samples_per_fold": (len(trials)-1)*3,
        "heldout_keypoints": len(heldout_errors), "ridge_alpha_fixed": args.alpha,
        "inputs": ["phase one-hot", "normalized phase timestamp", "start xy", "endpoint displacement"],
        "target": "observed intermediate normalized xy minus straight start-end interpolation",
        "training_resubstitution_rmse_norm": float(np.sqrt(np.mean(np.square(train_errors)))),
        "loo_learned_mean_euclidean_error_norm": float(np.mean(heldout_errors)),
        "loo_learned_rmse_norm": float(np.sqrt(np.mean(np.square(heldout_errors)))),
        "loo_straight_mean_euclidean_error_norm": float(np.mean(baseline_errors)),
        "loo_straight_rmse_norm": float(np.sqrt(np.mean(np.square(baseline_errors)))),
        "paired_trial_mean_error_reduction_norm": float(np.mean(trial_deltas)),
        "paired_trial_bootstrap_95pct_ci_norm": [float(delta_ci[0]), float(delta_ci[1])],
        "trials_improved_vs_straight": int(np.sum(trial_deltas > 0)),
        "trials_equal_vs_straight": int(np.sum(trial_deltas == 0)),
        "trials_worse_vs_straight": int(np.sum(trial_deltas < 0)),
        "bootstrap_resampling_unit": "trial ID within one source recording session",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20261016,
        "confidence_used": False,
        "raw_video_used_for_model": False,
        "notes": "Each leave-one-out fold withholds one full trial clip. All clips were split from one concatenated source recording session; folds do not estimate cross-session generalization. Five sparse labels are not dense motion ground truth."
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / "loo_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with (out / "per_trial_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=list(per_trial[0]))
        w.writeheader(); w.writerows(per_trial)
    if args.export_heldout_demo:
        selected = next((t for t in trials if t["data"]["demo_id"] == args.export_heldout_demo), None)
        if selected is None:
            raise SystemExit(f"Held-out demo not found: {args.export_heldout_demo}")
        exported = json.loads(json.dumps(selected["data"]))
        by_phase = {r["phase"]: r for r in rows if r["trial"] == args.export_heldout_demo}
        for frame in exported["frames"]:
            if frame.get("task_phase") in by_phase:
                r = by_phase[frame["task_phase"]]
                frame["object_position"] = [float(r["predicted_x"]), float(r["predicted_y"])]
        exported.setdefault("metadata", {})["trajectory_prior"] = {
            "type": "ridge residual predictor",
            "evaluation": "leave-one-original-trial-out",
            "training_excluded_trial": args.export_heldout_demo,
            "alpha": args.alpha,
            "note": "Only intermediate coordinates were predicted; start/release endpoints and phase timestamps are retained from this held-out annotation."
        }
        (out / f"{args.export_heldout_demo}-loo-predicted.json").write_text(
            json.dumps(exported, indent=2) + "\n", encoding="utf-8")
    # Persist the final fitted residual model for reproducible prediction.
    (out / "model.json").write_text(json.dumps({"mean": model_all[0].tolist(),
        "scale": model_all[1].tolist(), "coefficients": model_all[2].tolist(),
        "alpha": args.alpha, "features": summary["inputs"], "target": summary["target"]}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
