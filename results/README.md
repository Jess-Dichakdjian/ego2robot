# Results summary

## Matched route ablation

All conditions use the same `DemoConditionedPolicy`, grasp/release logic, LIBERO task 0, seed 0, 10 settle steps, 500-step limit, and 20 post-release steps. Each route was run at initial-state IDs 0, 1, and 2. Can start pose was identical; basket and Panda starting poses varied modestly. Straight projects all three interior XY annotations onto the start-to-release line. Shuffled deterministically permutes the same three interior coordinates (shuffle seed 23) while keeping phase labels and order fixed.

| Route | Episodes | Success | Final distance m: median (range) | Object path m: median (range) | Steps: median (range) |
|---|---:|---:|---:|---:|---:|
| Human | 3 | 3/3 | 0.0610 (0.0608–0.0610) | 0.8681 (0.8652–0.8763) | 133 (133–135) |
| Straight | 3 | 3/3 | 0.0600 (0.0600–0.0600) | 0.7960 (0.7894–0.8015) | 121 (121–121) |
| Shuffled | 3 | 3/3 | 0.0600 (0.0600–0.0600) | 0.9366 (0.9357–0.9464) | 145 (145–147) |

Observed: every condition succeeded, with similar final distances and different object paths. The human route changed the trajectory but did not improve the task outcome over the straight route in these runs. The shuffled route remained successful but produced the longest path and used the most steps. This is one annotated demo tested on three modestly varied saved states; it does not establish a general performance advantage.

The six personal demonstrations were a separate set: each was run once at initial-state ID 0 (seed 0) with the same demo-informed controller, and all six succeeded. Do not interpret those fixed-state runs as the matched three-route ablation.

Episode rows: [`route_ablation.csv`](route_ablation.csv). Per-step object/target/EEF data: [`route_ablation_trajectories.csv`](route_ablation_trajectories.csv). Generated table and plots: [`presentation/`](presentation/).

[`presentation/eef_trajectory_overlay.svg`](presentation/eef_trajectory_overlay.svg) plots the recorded `robot0_eef_pos` XY trace for human, straight, and shuffled routes in each of the three matched states. Recreate it with `python results/presentation/generate_eef_trajectory_plot.py` (standard library only).

## Historical generic baseline

The generic baseline uses a different controller and is not a route-prior ablation. Its one fixed-state reference episode failed to place the can within 500 steps (final distance 0.513 m, object path 0.006 m; finger-pad contact was observed). Six demo-informed annotations each completed once at that same state. See [`controller_comparison.md`](controller_comparison.md) and [`phone_demo_generalization.md`](phone_demo_generalization.md).

## Limits

No policy training, calibrated video-to-world transform, physical robot, randomized object start pose, or broad multi-task evaluation was used. Annotation timestamps/confidence are stored but do not control actions.

Recovery annotations, if collected, are supplementary human labels; no robotic recovery behavior was implemented or evaluated. The older generic controller reference uses different control logic and is not a fair measure of the route prior's effect.
