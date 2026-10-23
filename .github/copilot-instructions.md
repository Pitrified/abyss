# abyss - Copilot Instructions

## Project overview

`abyss` renders a scene from the viewer's point of view: track where the viewer is, know where the
screen is, and draw what they would see through it. Python 3.14, managed with **uv**.

The package name is `abyss` throughout the source.

## The pose-tools boundary - read this first

`abyss` depends on [`pose-tools`](https://github.com/Pitrified/pose-tools), pinned by git tag in
`pyproject.toml`. pose-tools owns **every general pose, video and geometry utility**, shared with
`climbing-wire` and `holo-table`:

| Need                                    | Where it lives                                              |
| --------------------------------------- | ----------------------------------------------------------- |
| MediaPipe landmarkers (pose, hand, face) | `pose_tools.landmark.pose` / `.hand` / `.face`             |
| Resolving and downloading `.task` models | `pose_tools.landmark.model_manager.ModelManager`            |
| Drawing landmarks on a frame            | `pose_tools.landmark.drawing`                               |
| Landmark arrays, visibility, distances  | `pose_tools.landmark.landmark_array` / `.distance`          |
| Video frames and iteration              | `pose_tools.video.frame` / `.load`                          |
| Homography and signal tracking          | `pose_tools.geometry.*`                                     |
| OpenCV / matplotlib display helpers     | `pose_tools.utils.cv` / `.plt`                              |
| MediaPipe result conversions            | `pose_tools.utils.mediapipe`                                |

**Do not reimplement any of these in abyss.** The whole point of the 2026 reboot was deleting
abyss's own copies of exactly this code. Before writing a utility, look in `pose_tools` first.

The test for new code: would `climbing-wire` want it? If yes it belongs upstream in pose-tools -
add it there, cut a tag, and bump the pin here. If it is about viewers, screens, or rendering a
scene, it belongs in abyss.

`abyss` must never be imported by `pose-tools`. The dependency is strictly one-way.

## Running & tooling

Use the Makefile; it exists so commands are discoverable and consistent.

```bash
make help        # list every target
make sync        # uv sync --all-extras --all-groups
make hooks       # pre-commit install, once per clone
make check       # lint + typecheck + test
make test        # uv run --no-sync pytest
make nbstrip     # strip notebook outputs (the hook only verifies)
make e2e         # browser tests: Chromium, a synthetic face clip as the camera
```

**The pre-commit hooks are installed and run on every commit**, so write to pass them rather than
fixing up afterwards: ruff check and format, pyright, stripped notebook outputs, no secrets, a
single trailing newline, no trailing whitespace. Never skip them with `--no-verify`; fix the finding.
`make check` covers the Python half, and `uv run --no-sync pre-commit run --files <paths>` the rest.

**The browser UIs are checked by running them.** `make e2e` serves `ui/`, opens it in headless
Chromium with a recorded face playing as the webcam, and reads the page's state; `scripts/drive_ui.py`
does the same by hand and prints one log stream. Use them before calling UI work done. Details in
`docs/guides/ui_harness.md`.

**Always run project code through `make`, or through `uv run --no-sync`.** A bare `uv run`
re-syncs the environment from `uv.lock` first, which silently reverts a local editable install of
pose-tools (`make dev-pose-tools`). `make undev` is the deliberate way back to the pinned tag.

## Architecture

| Layer       | Path                                  | Role                                                    |
| ----------- | ------------------------------------- | ------------------------------------------------------- |
| Params      | `src/abyss/params/abyss_params.py`    | Singleton `AbyssParams`; aggregates the paths           |
| Paths       | `src/abyss/params/abyss_paths.py`     | `AbyssPaths`; filesystem references                     |
| Metaclasses | `src/abyss/metaclasses/singleton.py`  | `Singleton` metaclass                                   |

The params layer is deliberately minimal: no env stage/location dispatch, no config models, no
`load_env()`. abyss has no secrets and runs in one place, so those were left out rather than
carried over from the template. **Add them when something needs them, not before** - the same rule
applies to new path entries in `AbyssPaths`.

Reach the paths through `get_abyss_paths()`, never by constructing `AbyssPaths()` directly.

## Environment constraints

**These are per machine, and the repo is developed on more than one.** They used to be written as
though one machine's constraints were the environment's; they are not.

| | g7 | g4 | Pixel 7 Pro |
| --- | --- | --- | --- |
| role | unattended home server: dev box, agent sessions, GPU provider | daily driver, usually at home | phone, capture and display |
| GPU | Quadro RTX 3000, 6 GB, driver 580, OpenGL 4.6 | integrated, old | phone GPU |
| display | X11 on `:1`, but nobody is usually in front of it | yes, with a person at it | yes |
| camera | Chicony 04f2:b6c8, measured, still reachable | HP HD Camera 04ca:7063, unmeasured | front camera, unmeasured |
| network | tailnet, plus the Cloudflare tunnel | tailnet | tailnet |

- **g7 is where things run, g4 is where a person is.** g7 hosts the code, the GPU and the measured
  camera, and serves anything that needs serving. g4 is the natural place to open a browser against
  it. g4 compatibility is welcome but not required: a renderer or viewer that only works on g7's GPU
  is acceptable, as long as something headless still covers it in the tests.
- **Write for an unattended run by default.** Everything must be checkable headless, on a recorded
  clip, with file output. That rule comes from the sequencing, not from the hardware: a phase that
  can only be run in front of a screen cannot be tested, and the machine with the GPU is the one
  nobody is sitting at.
- **MediaPipe runs on the CPU on both machines, and this was measured.** The `GPU` delegate exists
  in the enum but fails with `GPU processing is disabled in build flags`: the pip wheel is built
  without GPU support, so the driver, EGL and GLESv2 being present changes nothing. Do not write
  GPU-delegate code paths, and do not install drivers to fix it - only a source build of MediaPipe
  would, and at 11.2 ms per face landmarker call at 1920x1080 there is nothing to gain. The GPU is
  still real and still matters for OpenGL rendering on g7.
- **A viewer is not guaranteed.** `cv.imshow` and `cv.waitKey` work on both machines, but a window
  on g7 is only useful when someone is at it, and agent sessions on g7 cannot see it. Anything that
  needs a window is a manual script under `scripts/`, never part of the test suite.
  `scripts/camera.py` is the existing example.
- **The network is a tailnet.** g7, g4 and the phone are all on the same Tailscale network, so an
  endpoint on g7 is reachable from the other two directly, without going through the Cloudflare
  tunnel. The tunnel is still there for anything that has to be public.
- **The phone can run a web UI or a native app.** Both a browser page and a custom Flutter or Unity
  app have been installed on the Pixel 7 Pro before, so neither is new ground.
- **Data lives outside the repo.** Sample videos in `~/data/pose/` (`AbyssPaths.pose_fol`),
  MediaPipe models in `~/.mediapipe/models/`. Neither is in git, and neither is present on every
  machine. Models are no longer a manual step: `ModelManager().ensure_model("face_landmarker")`
  downloads what is missing. `~/data/pose/face01.mp4` is the clip with a face in it;
  `yoga01.mp4` has none that MediaPipe can find.

## Style rules

- Never use em dashes (`--`, `---`, or Unicode `—`). Use a hyphen `-` or rewrite the sentence.
- Use `loguru` (`from loguru import logger as lg`) for all logging.
- Raise descriptive named exceptions rather than bare `ValueError` / `RuntimeError`.
- Ruff runs with `select = ["ALL"]`. Fix findings rather than adding `noqa`.

## Docstring style

**Google style** throughout, matching the rest of the fleet:

```python
def example(value: int) -> str:
    """One-line summary.

    Extended description as plain prose.

    Args:
        value: Description of the argument.

    Returns:
        Description of the return value.

    Raises:
        KeyError: If the key is missing.
    """
```

Never use NumPy / Sphinx RST underline headers (`Args\n----`). Section labels always take a
trailing colon.

## Layout

- `tests/` mirrors `src/abyss/`.
- `scripts/` holds manual scripts, run directly and never collected by pytest.
- `ui/` holds the browser UIs, one folder each, plus `vendor.toml`, the manifest of their
  third-party files. `src/abyss/ui/` is what serves and tests them. `tests/e2e/` holds the browser
  tests, skipped by `make test`.
- `scratch_space/` is for throwaway prototyping notebooks.
- `plans/` holds the planning record: one folder per initiative, `tracking.md` first.
- `docs/` is plain markdown, read in the repo. `docs/guides/` for tooling and conventions,
  `docs/library/` for narrative notes on the code. There is no mkdocs site yet, so write files
  that a site could later consume unchanged.
