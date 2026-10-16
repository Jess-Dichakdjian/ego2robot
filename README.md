# Ego2Robot

[![Python tests](https://github.com/Jess-Dichakdjian/ego2robot/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Jess-Dichakdjian/ego2robot/actions/workflows/tests.yml)

**Can sparse points from a phone-recorded pick-and-place demonstration change a simulated Panda's route?** Ego2Robot tests this with a fixed, hand-coded controller in LIBERO. Human annotations provide a route prior; simulator state anchors metric placement.

<p align="center">
  <img src="results/presentation/phone_panda_demo_original.gif" alt="Original phone demonstration alongside its LIBERO Panda simulation" width="760">
</p>

Phone trial 01 is shown beside the corresponding Panda run using the original, unblurred phone footage. The clips are not synchronized or geometrically calibrated.

## At a glance

- **Problem and working system:** pick up a can and place it in a basket with a LIBERO Panda; retarget sparse can-centre clicks from phone video into its route. The fixed geometric controller completed the six original demo runs; simulator ground truth supplies metric can/basket poses.
- **Data:** 22 canonical annotated trials and 12 supplementary interaction exports with 28 manual hand points, all from one recording session ([canonical JSONs](data/processed/) · [interaction exports](data/interaction_annotations/)). The six MVP runs use one simulator state; the matched route comparison uses three states.
- **What changes the robot:** human points shape relative XY route and initial approach direction. Hand-coded logic supplies offsets, phase transitions, grasp/release schedule, and control; timestamps/confidence are stored but do not drive actions.
- **Matched route comparison:** human, straight, and shuffled routes each succeeded in 3/3 saved states; the human route did not improve success or placement over straight. [Executed trajectories and route metrics](results/presentation/eef_trajectory_overlay.svg) · [CSV](results/route_ablation.csv).
- **Offline learning:** trial-held-out sparse-keypoint error fell from 0.2034 to 0.0701 normalized image units over 18 trials. This is offline prediction, not a learned manipulation policy. [Experiment details](docs/research_extensions.md#offline-trajectory-keypoint-prediction) · [summary](results/learned_prior/summary.json) · [saved model](results/learned_prior/model.json).
- **What failed:** the separate learned hand-approach experiment scored worse than its constant-direction baseline and achieved 0/3 grasps versus 3/3 for each geometric route. Its target left one-finger contact and did not reach the controller's close tolerance. [Experiment](docs/interaction_approach_experiment.md) · [failure diagnostic](docs/learned_grasp_failure_diagnostic.md) · [approach model](results/interaction_learning/approach_model.json).
- **Inspect the work:** [side-by-side preview](results/presentation/phone_panda_demo_original.gif) · [geometric controller](src/ego2robot/policies/pick_place.py) · [retargeter](src/ego2robot/retargeting/simple_geometric.py) · [full methods and limits](docs/research_extensions.md) · [reproduction steps](#reproduce) · [pinned simulator setup](docs/setup.md) · [successful Panda video](results/runs/phone_demo_001_success.mp4).

## Pipeline

![Ego2Robot architecture: phone video, annotation, canonical demo, retargeting, LIBERO Panda, and evaluation](results/presentation/architecture.svg)

## Method

The person clicks the can's image center at five ordered phases: `start`, `approach`, `grasp`, `transport`, and `release`. These normalized 2D points describe object motion; they do not measure depth, hand pose, or force.

| Input | Controls | Does not control |
|---|---|---|
| Human phase keypoints | Relative XY path shape and start-to-grasp approach direction | Metric XYZ, grasp offset, gripper force, phase duration |
| LIBERO state | Live can and basket poses, Panda pose, task success | Demonstration route shape |
| Fixed controller | Hover/lift/release offsets, approach stand-off, reach tolerances, gripper open/close schedule, action scaling | Learned adaptation |
| Annotation timestamps/confidence | Stored in canonical JSON | Not used to schedule actions or change control |

`SimpleGeometricRetargeter` maps image-plane displacements between the live simulated can and basket endpoints. `DemoConditionedPolicy` follows phase waypoints, but uses live simulator poses for start, grasp, and release. The approach direction comes from the annotated start-to-grasp vector; transport XY uses the retargeted route point. Hand-coded offsets include 8 cm approach stand-off and 12 cm hover/carry lift. This is geometric retargeting with a motion prior, not policy training.

Geometric retargeting was selected over VLA fine-tuning because the labelled dataset is limited to six short demonstrations with five sparse points each, and the monocular videos have no camera calibration or depth. A VLA would add compute and training complexity without resolving that missing geometry. The mapping is computationally simple and interpretable: it keeps the annotated relative XY route while simulator poses provide metric anchors. No VLA was trained or fine-tuned.

## Experiment

We compare three route conditions with the same `DemoConditionedPolicy`, grasp/release behavior, task, and evaluator:

- **Human:** the five annotated points.
- **Straight:** preserve endpoints and project each interior (`approach`, `grasp`, `transport`) point onto the start-to-release line.
- **Shuffled:** preserve endpoints and phase order, but deterministically permute the three interior XY points (shuffle seed 23).

Each route is evaluated at the same LIBERO task 0 initial-state IDs `0`, `1`, and `2`, with seed 0, 10 settle steps, a 500-step limit, and 20 post-release steps. These are distinct saved simulator initializations; the can start pose is identical, while basket and Panda starting poses vary modestly. Treat this as a small matched comparison, not a broad robustness test.

| Route prior | Episodes | Success | Final distance m: median (range) | Object path m: median (range) | Steps: median (range) |
|---|---:|---:|---:|---:|---:|
| Human | 3 | 3/3 (100%) | 0.0610 (0.0608–0.0610) | 0.8681 (0.8652–0.8763) | 133 (133–135) |
| Straight | 3 | 3/3 (100%) | 0.0600 (0.0600–0.0600) | 0.7960 (0.7894–0.8015) | 121 (121–121) |
| Shuffled | 3 | 3/3 (100%) | 0.0600 (0.0600–0.0600) | 0.9366 (0.9357–0.9464) | 145 (145–147) |

**Observed:** all nine episodes succeeded, and the measured object paths differed. Final object-to-target distances were close across routes. Straight paths were shortest (median 0.796 m, 121 steps); shuffled paths were longest (0.937 m, 145 steps); the human path was between them (0.868 m, 133 steps). **Interpretation:** the annotations change robot movement, but did not improve task completion or final placement over the straight route in these three states. The fixed metric anchors and controller dominate the placement endpoint. This is evidence from one annotated demo and three modestly varied simulator initializations only.

The six personal annotations were each run once with the same demo-informed controller at LIBERO initial-state ID 0 (seed 0); all six completed. This is a separate fixed-state demonstration set, not the three-state route ablation below.

| Demonstration | Success | Final distance (m) | Object path (m) | Steps | Grasp contact |
|---|---:|---:|---:|---:|---:|
| phone_demo_001 | Yes | 0.061 | 0.865 | 133 | Yes |
| phone_demo_002 | Yes | 0.060 | 0.905 | 136 | Yes |
| phone_demo_003 | Yes | 0.060 | 0.964 | 141 | Yes |
| phone_demo_004 | Yes | 0.060 | 1.064 | 188 | Yes |
| phone_demo_005 | Yes | 0.061 | 0.857 | 124 | Yes |
| phone_demo_006 | Yes | 0.060 | 0.804 | 123 | Yes |

These are six single runs at the same initial state, not six independently sampled simulator conditions.

![Matched object-path overlays for three LIBERO initial states](results/presentation/trajectory_overlay.png)

The following figure uses the per-step `robot0_eef_pos` values recorded during those same nine episodes. It compares the **executed Panda end-effector XY trajectories**, rather than the annotated or retargeted waypoints.

![Executed Panda end-effector XY paths for human, straight, and shuffled routes across three matched LIBERO initial states](results/presentation/eef_trajectory_overlay.svg)

The episode-level metrics are in [`results/route_ablation.csv`](results/route_ablation.csv); per-step object and end-effector paths are in [`results/route_ablation_trajectories.csv`](results/route_ablation_trajectories.csv). The summary plots show success, final distance, object path length, and episode steps.

<table>
  <tr>
    <td><img src="results/presentation/success_rate.png" alt="Success rate by route prior" width="380"></td>
    <td><img src="results/presentation/final_object_target_distance.png" alt="Final object-to-target distance by route prior" width="380"></td>
  </tr>
  <tr>
    <td><img src="results/presentation/trajectory_length.png" alt="Object path length by route prior" width="380"></td>
    <td><img src="results/presentation/episode_steps.png" alt="Episode steps by route prior" width="380"></td>
  </tr>
</table>

The older generic-controller baseline uses different control logic and is reported only as a historical reference, not as evidence that human routes improve performance. Its one fixed-state episode failed to place the can within 500 steps (final distance 0.513 m; path length 0.006 m; finger-pad contact was observed). The six fixed-state annotation runs above are also not a matched baseline comparison. See [`results/controller_comparison.md`](results/controller_comparison.md) and [`results/phone_demo_generalization.md`](results/phone_demo_generalization.md).

## Research Extension: Learning from Sparse Human Demonstrations

Separate from the working geometric MVP, trial-level leave-one-out ridge regression reduced sparse can-keypoint prediction error from **0.2034** to **0.0701** normalized image units across 18 trials from one recording session. This is offline prediction of sparse labels, not dense motion reconstruction or a demonstrated robot-performance gain.

A distinct hand-informed approach predictor had **18.30°** held-out angular error versus **15.57°** for the constant-direction baseline. In matched simulator runs, it achieved **0/3 grasps**, versus **3/3** for both straight and human-geometric approaches. The episode bridge uses fixed task-relative features `(0, 0)` at initialization; it is not dynamically perception-conditioned. The diagnostic identifies poor approach geometry and incomplete gripper alignment as the proximal failure, while the image-to-robot mapping remains uncalibrated. Neither model is a successful learned manipulation policy; the geometric controller and its results are unchanged. Full methods, data audit, reproduction instructions, and limitations are in [Research Extensions](docs/research_extensions.md), with the [paired failure diagnostic](docs/learned_grasp_failure_diagnostic.md).

## What worked and what did not

- The frozen demo-informed controller completed the successful `phone_demo_001` sequence and all six previously recorded fixed-state demo runs.
- The current route experiment measures how changing the human path changes the object trajectory, while keeping grasp/release logic fixed.
- The earlier absolute video-to-world attempt was unreliable because the videos have no camera calibration or depth. Simulator ground truth remains necessary for metric anchors.
- The fixed-state generic baseline contacted the can but failed to place it.

## Reproduce

The package requires Python **3.10 or newer**. The checks have different dependency needs.

### A. Unit tests — no simulator

From the repository root, install the small package and run the suite:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

### B. Offline learning and data audit — no source videos or LIBERO

The editable install above includes NumPy, which is sufficient for these commands. Outputs go under the ignored `results/reproduction/` directory.

```bash
python scripts/train_trajectory_prior.py --data-root data/processed --out results/reproduction/learned_prior
python scripts/audit_interaction_dataset.py --interaction-dir data/interaction_annotations --canonical-dir data/processed --output results/reproduction/interaction_audit.json
python scripts/train_interaction_approach.py --interaction-dir data/interaction_annotations --canonical-dir data/processed --output-dir results/reproduction/interaction_learning
```

### C. Panda simulation replay — pinned LIBERO environment required

The simulator is **not** installed by `pip install -e .`. To preserve the committed metrics and figures, clone the repository, set `PROJECT_ROOT` to that clone, and complete the pinned WSL/LIBERO, PyTorch, Robosuite, and MuJoCo setup in [`docs/setup.md`](docs/setup.md). Then run the replay from the documented WSL virtual environment:

```bash
git clone https://github.com/Jess-Dichakdjian/ego2robot.git ego2robot-replay
cd ego2robot-replay
export PROJECT_ROOT="$PWD"
```

After completing [`docs/setup.md`](docs/setup.md) with this `PROJECT_ROOT`:

```bash
"$HOME/.venvs/ego2robot/bin/python" scripts/validate_demo.py data/processed/phone-demo-001.json
"$HOME/.venvs/ego2robot/bin/python" scripts/evaluate.py --policy demo --demo-json data/processed/phone-demo-001.json --seed 0 --initial-state-id 0 --max-steps 500 --settle-steps 10 --post-release-steps 20 --output results/reproduction/phone_demo_001.csv --trajectory-output results/reproduction/phone_demo_001_trajectory.csv
"$HOME/.venvs/ego2robot/bin/python" scripts/evaluate_route_ablation.py --seeds 0 --initial-states 0 1 2
```

The final route-ablation command runs nine episodes and writes its summary and trajectory CSVs into the clone. Regenerate the plots there, after the evaluation (Matplotlib is listed in `requirements-libero-smoke.txt`):

```bash
python results/presentation/generate_plots.py
python results/presentation/generate_eef_trajectory_plot.py
```

To recreate the paired preview from local source footage, provide `--phone-video` to `results/presentation/generate_demo_gif.py`. The script reads the original phone clip without altering its frames and writes the paired preview to the requested output path. Raw phone clips are not included in the publication candidate. The browser annotator is [`tools/manual_video_annotator.html`](tools/manual_video_annotator.html).

## Scope and limits

The original geometric MVP is six demonstrations, one LIBERO task, and a hand-designed controller. The expanded canonical set contains 22 annotated clips split from one original phone-recording session, not 22 independent sessions. The optional extension adds an offline keypoint predictor and a failed hand-approach experiment; it does not add a successful learned manipulation policy. Annotation timestamps and confidence scores are stored in the canonical JSON but are not used to schedule actions, weight waypoints, or otherwise change control. Recovery-event annotations are supplementary human data; they are not evidence of implemented or evaluated robotic recovery. No RL, VLA fine-tuning, physical Panda evaluation, calibrated video-to-world transform, or uncertainty estimate was implemented. The matched route ablations use a few saved simulator states and do not establish broad generalization.

Future work could quantify uncertainty in clicks and retargeted waypoints, detect stalled execution, and request a human correction before continuing. That is a proposed extension, not a result of this project.

## A Note on Scope

In roughly 24 hours, I took Ego2Robot from phone-recorded demonstrations to a working, evaluated LIBERO manipulation simulation. I chose to complete a reproducible experiment rather than start a larger model I could not validate. With more time, I would explore learning from human corrections so a robot can recover from failed grasps. There is much more I want to investigate from this starting point; I would welcome the chance to show what I can build with more time and a harder problem.

## License and data

The software is MIT-licensed. The annotation JSON files are included for reproducibility; no separate license is specified for the personal recordings or labels. Full source videos are kept out of this publication candidate. The paired preview is an unblurred excerpt from phone trial 01.
