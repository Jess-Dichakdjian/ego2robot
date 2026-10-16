# Research extensions

These optional experiments are separate from the frozen geometric MVP. Both use ridge regression, but only the first shows a positive result, and only for offline sparse-keypoint prediction. Neither is a successful learned manipulation policy.

## Offline trajectory-keypoint prediction

A ridge residual model predicts the three intermediate can-centre keypoints from five sparse annotations per trial. Trial-level leave-one-out evaluation over 18 trials from one recording session reduced mean held-out error from **0.2034** normalized image units for straight interpolation to **0.0701** for ridge regression. Ridge had lower per-trial error on 18/18 held-out trials (paired trial bootstrap 95% interval for mean reduction: 0.1097–0.1555). This is evidence for improved prediction of the annotated sparse points within this dataset; it is not dense tracking, cross-session generalization, or downstream robot improvement.

The model, training script, annotated JSON data, held-out predictions, per-trial scores, and figure are included. No video is needed to reproduce this offline result.

## Human hand-side approach experiment

A separate ridge model predicts a 2D human hand-to-can approach direction from five positive approach examples, grouped leave-one-trial-out. Its mean angular error was **18.30°**, worse than the constant training-direction baseline (**15.57°**). The mapped approach was passed through an evaluation wrapper into the same fixed Panda controller; the controller itself was not changed.

| Route | Matched episodes | Successful grasps |
|---|---:|---:|
| Scripted straight route | 3 | 3/3 |
| Human-geometric route | 3 | 3/3 |
| Learned approach direction | 3 | 0/3 |

The learned direction changed the Panda's path but led to sustained one-finger contact without a grasp in all three saved states. This is a negative manipulation result, not a successful learned policy. [The diagnostic](learned_grasp_failure_diagnostic.md) compares a failed run with a successful geometric run.

The supplementary interaction JSONs, canonical annotations, model weights, training/evaluation code, metrics, contact logs, and per-step traces are included. Neither training nor the diagnostic requires private source videos. Replaying the simulator comparison requires the pinned LIBERO/Robosuite/MuJoCo environment described in [setup](setup.md); the committed metrics and traces are the recorded runs.

## Reproduction

From the repository root with the documented environment installed:

```bash
python scripts/train_trajectory_prior.py --data-root data/processed --out results/reproduction/learned_prior
python scripts/plot_learned_prior.py --predictions results/reproduction/learned_prior/loo_predictions.csv --output results/reproduction/learned_prior/learned_sparse_prior.svg
python scripts/audit_interaction_dataset.py --interaction-dir data/interaction_annotations --canonical-dir data/processed --output results/reproduction/interaction_audit.json
python scripts/train_interaction_approach.py --output-dir results/reproduction/interaction_learning
python scripts/evaluate_interaction_approach.py --initial-state-ids 0 1 2 --seed 41 --output-dir results/reproduction/interaction_learning/matched
python scripts/plot_interaction_approach.py --results-dir results/reproduction/interaction_learning/matched --output results/reproduction/interaction_learning/approach_to_robot_action.svg
```

These commands keep reproduced outputs separate from the committed result files. The first two reproduce offline keypoint predictions. The remaining commands recompute the interaction audit and angle metrics, then replay the three-route simulator comparison using the same task, seed, and saved initial-state IDs. They do not tune or modify the frozen controller.
