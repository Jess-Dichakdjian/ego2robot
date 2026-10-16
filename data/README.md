# Data

The repository includes six canonical annotations (`processed/phone-demo-001.json` through `phone-demo-006.json`) and a synthetic schema example. Each annotation has normalized 2D can-center points at `start`, `approach`, `grasp`, `transport`, and `release`; they are image-plane labels, not calibrated 3D trajectories.

The full phone videos and split clips are stored locally and are not distributed with this publication candidate. A privacy-obscured paired preview appears in the root README. For local annotation, open `tools/manual_video_annotator.html` and select a video from your local `data/processed/trials_browser/` folder. Validate an exported annotation with:

```bash
python scripts/validate_demo.py data/processed/<demo>.json
```

Software is MIT-licensed. No separate license is specified for the personal annotations or recordings; obtain permission before reuse or redistribution.
