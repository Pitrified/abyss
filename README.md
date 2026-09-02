# Stare into the Abyss

Render a scene from the viewer's point of view.

1. Compute the position of the viewer
1. Compute the position of the screen
1. Render the scene the viewer sees on the screen

Step 1 is pose tracking, which comes from [`pose-tools`](https://github.com/Pitrified/pose-tools).
Steps 2 and 3 are what abyss is for.

How it works, with the mathematics: [`docs/library/geometry_overview.md`](docs/library/geometry_overview.md).
Everything else that is written down is indexed in [`docs/README.md`](docs/README.md).

## Installation

### Setup `uv`

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

### Install the package

```bash
make sync          # uv sync --all-extras --all-groups
```

`pose-tools` is pinned by git tag in `pyproject.toml`, so a plain sync is enough.

### Data and models

Neither is stored in the repository:

- sample videos go in `~/data/pose/` (`AbyssPaths.pose_fol`),
- MediaPipe `.task` models go in `~/.mediapipe/models/`, resolved by pose-tools' `ModelManager`.
  Download them from
  [the MediaPipe solutions page](https://developers.google.com/mediapipe/solutions/vision).

## Usage

```bash
make help          # every available command
make check         # lint, typecheck, test
```

Run project code through `make`, or through `uv run --no-sync`. A bare `uv run` re-syncs the
environment and silently undoes a local editable install of pose-tools - see
[`docs/guides/makefile.md`](docs/guides/makefile.md).

## Layout

| Path             | What is in it                                                        |
| ---------------- | --------------------------------------------------------------------- |
| `src/abyss/`     | the package: params and paths, plus abyss's own code as it grows      |
| `tests/`         | mirrors `src/abyss/`                                                  |
| `scripts/`       | manual scripts, run directly, never collected by pytest               |
| `notebooks/`     | notebooks kept in the repo, outputs stripped                          |
| `scratch_space/` | throwaway prototyping, not part of the package                        |
| `docs/`          | plain markdown: `guides/` for tooling, `library/` for the code        |
| `plans/`         | the planning record, one folder per initiative                        |

`scratch_space/` is for things that may be deleted tomorrow; `plans/` is for decisions that should
outlive the session that made them. Start at
[`plans/00_template_alignment/tracking.md`](plans/00_template_alignment/tracking.md).

## Relationship to pose-tools

`pose-tools` owns every general pose, video and geometry utility, shared with `climbing-wire` and
`holo-table`. abyss owns viewers, screens and rendering. The dependency is one-way, and new general
code belongs upstream rather than here - see
[`docs/library/pose_tools_boundary.md`](docs/library/pose_tools_boundary.md).
