# Historical controller reference

This comparison uses different controllers and must not be interpreted as an isolated ablation of the human route prior.

| Controller | Runs | Success | Final distance (m) | Object path (m) | Steps | Contact |
|---|---:|---:|---:|---:|---:|---|
| Generic baseline | 1 | 0/1 | 0.513 | 0.006 | 500 | Finger pad |
| Demo-informed controller, six annotations | 6 | 6/6 | 0.060 median | 0.885 median | 134.5 median | Recorded in all runs |

All runs used LIBERO task 0 and initial state 0. The baseline result is a single reference episode; the six demo rows are also single runs per annotation. The fair same-controller route ablation is [`route_ablation.csv`](route_ablation.csv).
