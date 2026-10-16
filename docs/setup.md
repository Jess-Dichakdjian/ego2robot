# Environment and simulator setup

## Tested environment

The recorded simulator run used WSL2 Ubuntu 22.04, Python 3.10.12, CPU PyTorch, and MuJoCo's EGL renderer. GPU execution was not part of the reported experiment. The setup below pins the simulator source and package versions; host-specific machine specifications are intentionally omitted.

## Pinned simulator source

- Official repository: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- LIBERO package version: `0.1.0`
- Git commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- WSL Python: `3.10.12` (Ubuntu 22.04 default; the upstream guide recommends Python 3.8.13)
- PyTorch: `1.11.0+cpu`
- Robosuite: `1.4.0`
- MuJoCo: `2.3.2`

The source checkout lives in the ignored `.libero-src/` directory. The isolated WSL environment lives at `$HOME/.venvs/ego2robot`; no Ubuntu-wide packages were installed. Package installation on the Windows-mounted drive was extremely slow, so the Python environment is kept on WSL's Linux filesystem.

## Reproduce the smoke test

In Ubuntu, set `PROJECT_ROOT` to the WSL path of this repository, then run:

```bash
export PROJECT_ROOT=/path/to/ego2robot
git init "$PROJECT_ROOT/.libero-src"
git -C "$PROJECT_ROOT/.libero-src" remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git
git -C "$PROJECT_ROOT/.libero-src" fetch --depth 1 origin 8f1084e3132a39270c3a13ebe37270a43ece2a01
git -C "$PROJECT_ROOT/.libero-src" checkout --detach FETCH_HEAD

python3 -m venv --without-pip "$HOME/.venvs/ego2robot"
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
"$HOME/.venvs/ego2robot/bin/python" /tmp/get-pip.py
"$HOME/.venvs/ego2robot/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu
"$HOME/.venvs/ego2robot/bin/python" -m pip install mujoco==2.3.2
"$HOME/.venvs/ego2robot/bin/python" -m pip install \
  -r "$PROJECT_ROOT/requirements-libero-smoke.txt"
"$HOME/.venvs/ego2robot/bin/python" -m pip install -e "$PROJECT_ROOT/.libero-src"
"$HOME/.venvs/ego2robot/bin/python" -m pip install -e "$PROJECT_ROOT"

MUJOCO_GL=egl "$HOME/.venvs/ego2robot/bin/python" \
  "$PROJECT_ROOT/scripts/libero_smoke_test.py" \
  --steps 150 --seed 0 --random-actions \
  --video "$PROJECT_ROOT/results/runs/libero_random.mp4"
```

The smoke script writes a local LIBERO path configuration under `.libero-config/` and prints the observation shapes. The `libero_10` task 0 reset and 10 no-op steps were also run successfully.

## Verified output

- Task: `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
- Observations included Panda joint and end-effector state, object positions, `agentview_image` and `robot0_eye_in_hand_image` at 128×128 RGB, and a 112-value `object-state` vector.
- Ten no-op steps returned reward `0.0`, `done=False`.
- Seed 0, 150 uniform-random-action steps returned reward `0.0`, `done=False`.
- The saved MP4 was reopened and decoded: 150 frames, 128×128, 20 fps, 195,154 bytes.
- `libero_object` task 0 (`pick_up_the_alphabet_soup_and_place_it_in_the_basket`) also loaded, reset, and completed 10 no-op steps. This is the planned one-object task for the human demonstration experiment.

The full upstream requirements also include `robomimic`; installing it attempted to build `egl_probe` and failed because this Ubuntu image lacks native build tools. `robomimic` is omitted because it is not needed by the geometric MVP or the included ridge experiments. The pinned environment was used for the committed matched Panda comparisons; the offline ridge scripts need NumPy, and the figure scripts use Matplotlib from `requirements-libero-smoke.txt`. The dataset-path and Robosuite private-macro messages are warnings; they did not block the recorded simulator runs.
