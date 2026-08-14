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
make check       # lint + typecheck + test
make test        # uv run --no-sync pytest
make nbstrip     # strip notebook outputs (the hook only verifies)
```

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

- **CPU only.** No Nvidia GPU here. MediaPipe defaults to the CPU delegate and its wheels carry no
  CUDA build, so this costs nothing - but do not write GPU-delegate code paths.
- **Headless.** No display: `cv.imshow` and `cv.waitKey` cannot run. `scripts/camera.py` needs a
  camera and a screen and is manual-only. Prefer matplotlib output or writing frames to a file.
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
- `scratch_space/` is for throwaway prototyping notebooks.
- `plans/` holds the planning record: one folder per initiative, `tracking.md` first.
- `docs/` is plain markdown, read in the repo. `docs/guides/` for tooling and conventions,
  `docs/library/` for narrative notes on the code. There is no mkdocs site yet, so write files
  that a site could later consume unchanged.
