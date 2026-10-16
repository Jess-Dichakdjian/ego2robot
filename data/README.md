# Data

The repository includes 22 canonical annotations (`processed/phone-demo-001.json` through `processed/phone-demo-022.json`) and a synthetic schema example. Normal demonstrations use five normalized 2D can-centre points at `start`, `approach`, `grasp`, `transport`, and `release`; recovery trials contain chronological event labels. These are image-plane labels, not calibrated 3D trajectories. Twelve separate interaction JSONs add sparse hand points for the optional approach experiment; missing or occluded points remain missing. The interaction exports have a stale `annotation_source` field on supplied hand points, recorded in the audit report.

The full phone videos and split clips are stored locally and are not distributed here. The paired preview appears in the root README. For local annotation, open `tools/manual_video_annotator.html` and select a video from your local `data/processed/trials_browser/` folder. Validate an exported annotation with:

```bash
python scripts/validate_demo.py data/processed/<demo>.json
```

Software is MIT-licensed. No separate license is specified for the personal annotations or recordings; obtain permission before reuse or redistribution.
