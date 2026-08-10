---
status: planned
---

# Phase 3 - abyss tooling migration

## Overview

The first phase that commits to `abyss` itself. Moves the repo from poetry/3.11 with no tooling
to the template's uv/3.14 setup with lint, types, hooks and tests. Source stays as it is here -
deduplication is phase 4 - so that a tooling failure and a refactor failure never get confused.

Context: [`00_feature_inventory.md`](00_feature_inventory.md), features #1-5, #12, #13, #16, #17.
Depends on phase 2 (a pinned `pose-tools` tag to declare) and, for the Makefile, phase 1.

## Goals

1. `uv sync` works on Python 3.14 with mediapipe installed.
2. `ruff`, `pyright`, `pytest` and `pre-commit` all run clean.
3. The template's folder skeleton and params layer are in place.

## Plan

- **pyproject rewrite**: `[project]` metadata, `requires-python = "==3.14.*"`, hatchling build
  with `packages = ["src/abyss"]`, `[tool.hatch.metadata] allow-direct-references = true` (needed
  the moment the pose-tools git dep is declared), `[tool.pyright]` with `venvPath`/`venv` and
  `include = ["src", "tests"]`. Dependency groups copied from the template: `test`, `lint`,
  `notebook`, with `dev` including them. **No `docs` group** - Q4 defers mkdocs, so nothing would
  consume `mkdocs`/`mkdocs-material`/`api-autonav`.
- **Runtime deps**: mediapipe, numpy, opencv-contrib-python, loguru, pydantic, python-dotenv, and
  `pose-tools @ git+https://github.com/Pitrified/pose-tools@v0.1.0`. Drop `ipykernel` from runtime
  deps - it belongs in the `notebook` group.
- **Delete** `poetry.lock`, generate `uv.lock`, add `.python-version` (`3.14`).
- Copy `ruff.toml` verbatim, adjusting `target-version` to `py314` (the template still says
  `py313`; check which the template settles on rather than diverging).
- Copy `.pre-commit-config.yaml` and `.secrets.baseline`, then `pre-commit install`.
  The `nbstripout --verify` hook matters here: `notebooks/sample01.ipynb` is tracked and has
  never been stripped.
- Add the folder skeleton with `.gitkeep`: `data/`, `cache/`, `scripts/`, `scratch_space/`, and
  `docs/{guides,library}/` (plain markdown, no site build - Q4/Q6), plus `.vscode/settings.json`
  and `.gitattributes`.
- Port the params/config layer (`params/`, `metaclasses/singleton.py`,
  `data_models/basemodel_kwargs.py`, `config/`), renaming `project_name` to `abyss`. **Only the
  branches abyss needs now**: no speculative env types, no path entries for folders nothing reads.
  `AbyssPaths` starts from the paths actually referenced today and grows when something needs one.
  This is what replaces `utils/data.py:get_resource`, but that file is deleted in phase 4.
- Move `tests/video/camera.py` to `scripts/` - it is a manual camera script, not a test, and it
  will fail collection once pytest actually runs.
- Add `tests/conftest.py` and the first real tests (params and paths, mirroring pose-tools'
  `tests/params/`), so the suite is not empty.
- Add the Makefile from phase 1 with abyss's own targets left for later.
- `src/abyss/` is left alone in this phase - files are added beside the existing tree, imports are
  not touched. That is sequencing, not preservation: it keeps a tooling failure distinguishable
  from a refactor failure. Phase 4 then rewrites imports and deletes modules freely.
- Expect the first `ruff check` under `select = ["ALL"]` on 2023 code to produce a large number of
  findings, mostly docstrings and typing. Fix them here rather than carrying `noqa` forward - but
  a module slated for deletion in phase 4 is not worth polishing; leave those and let phase 4
  remove them, then re-run the linter.

## Environment constraints

Both are properties of this box, and both belong in the copilot instructions in phase 5.

- **No Nvidia GPU** (`nvidia-smi` absent). This costs nothing: mediapipe's `BaseOptions` defaults
  to the CPU delegate, and neither abyss nor pose-tools sets `delegate` anywhere today. The wheel
  in the lock is `py3-none-manylinux_2_28_x86_64`, i.e. no CUDA build in the first place. The rule
  to record is that GPU-delegate code paths are out of scope, not that anything needs changing.
- **Headless** (`DISPLAY` unset, SSH box). `utils/cv.py:cv_imshow_rgb` calls `cv.imshow`, and
  `tests/video/camera.py` opens a window and calls `waitKey` - neither can run here. This is a
  second reason that script belongs in `scripts/` rather than under pytest collection, and it
  means any verification that needs a window has to happen on a machine with a display or be
  replaced by writing frames to a file.

## Out of scope

- Deleting the duplicated modules or importing from `pose_tools` (phase 4).
- mkdocs, the docs workflow, and agent instruction files (phase 5).
- Any new functionality, including the render layer.

## Done when

- `uv sync --all-extras --all-groups` succeeds on 3.14 and `python -c "import mediapipe"` works.
- `uv run ruff check .` and `uv run ruff format --check .` are clean.
- `uv run pyright` is clean.
- `uv run pytest` passes with at least the params tests present.
- `pre-commit run --all-files` passes, including `nbstripout --verify`.
