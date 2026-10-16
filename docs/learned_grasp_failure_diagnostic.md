# Learned-approach grasp failure: paired log diagnosis

This is a read-only comparison of the saved seed 41, LIBERO initial-state 0 runs. It does not modify the controller or rerun simulation.

## Matched conditions

Both runs start with the same can pose `(-0.11909, -0.23978, 0.03839) m` and basket pose `(0.01484, 0.25212, -0.00458) m`; they use the same `DemoConditionedPolicy`, gripper schedule, and 500-step cap.

| Evidence | Learned approach (failed) | Straight geometric (successful) |
|---|---:|---:|
| Outcome | 0.5170 m final distance; 500 steps | 0.0600 m final distance; 121 steps |
| Requested pregrasp EEF target | `(-0.19742, -0.22350, 0.15839) m` | `(-0.14011, -0.31697, 0.15839) m` |
| Pregrasp target separation | — | 0.10964 m from learned target |
| Direction difference | — | 86.51° from learned direction |
| Grasp contact near timeout/success | One finger only | Both finger pads |
| Gripper at contact | Remained open; close threshold never reached | Closed after reaching grasp tolerance |

At learned-run step 499, the live can centre was `(-0.11573, -0.24580, 0.04328) m`, the requested EEF target was `(-0.11573, -0.24580, 0.06328) m`, and actual EEF was `(-0.12004, -0.26218, 0.08150) m`. The EEF remained about 2.5 cm from the requested target, outside the 2 cm grasp tolerance, and only `finger2` collision was logged. The requested gripper action stayed open (`-1`, actuator controls `[0.04, -0.04]`). In the successful route, the EEF came within grasp tolerance; both pads contacted the can and the close action was issued.

## Assessment

**Observed proximal failure: gripper alignment/contact geometry.** The learned route reaches its pregrasp phase, but enters descent from a different side and never aligns the EEF closely enough for the unchanged controller to close. There is no evidence of a gripper actuator malfunction or a missing close command once the grasp tolerance is met.

**Likely upstream contributor: approach-side choice and its retargeting.** The predicted approach direction places the pregrasp point 10.96 cm from the successful straight-route point (86.51° direction difference). The model's leave-one-trial-out direction error (18.30°) was worse than the constant mean (15.57°). Its inference features are near the training mean at episode start, so it provides little state-specific adaptation. The endpoint-basis conversion is internally consistent with the geometric retargeter, but the human hand point is an uncalibrated monocular image vector; logs cannot establish that its direction is physically equivalent to a Panda approach direction. Thus the logs do not distinguish prediction error from camera-to-robot semantic mismatch as the source of the bad side.

## Reproducibility

The model, held-out predictions, matched metrics, path traces, and sanitized contact logs are included under `results/interaction_learning/`. The comparison uses only the checked-in canonical/interaction JSONs and the pinned LIBERO simulator dependencies; it does not require source videos. The committed traces show the measured failure and are sufficient to inspect the diagnosis without rerunning the simulation.
