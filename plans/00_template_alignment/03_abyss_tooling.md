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
  `notebook`, `docs`, with `dev` including them.
- **Runtime deps**: mediapipe, numpy, opencv-contrib-python, loguru, pydantic, python-dotenv, and
  `pose-tools @ git+https://github.com/Pitrified/pose-tools@v0.1.0`. Drop `ipykernel` from runtime
  deps - it belongs in the `notebook` group.
- **Delete** `poetry.lock`, generate `uv.lock`, add `.python-version` (`3.14`).
- Copy `ruff.toml` verbatim, adjusting `target-version` to `py314` (the template still says
  `py313`; check which the template settles on rather than diverging).
- Copy `.pre-commit-config.yaml` and `.secrets.baseline`, then `pre-commit install`.
  The `nbstripout --verify` hook matters here: `notebooks/sample01.ipynb` is tracked and has
  never been stripped.
- Add the folder skeleton with `.gitkeep`: `data/`, `cache/`, `scripts/`, `docs/`, plus
  `.vscode/settings.json` and `.gitattributes`.
- Port the params/config layer (`params/{abyss_params,abyss_paths,env_type,load_env,sample_params}.py`,
  `metaclasses/singleton.py`, `data_models/basemodel_kwargs.py`, `config/`), renaming
  `project_name` to `abyss`. This is what replaces `utils/data.py:get_resource`, but the deletion
  of that file happens in phase 4.
- Move `tests/video/camera.py` to `scripts/` - it is a manual camera script, not a test, and it
  will fail collection once pytest actually runs.
- Add `tests/conftest.py` and the first real tests (params and paths, mirroring pose-tools'
  `tests/params/`), so the suite is not empty.
- Add the Makefile from phase 1 with abyss's own targets left for later.
- Expect the first `ruff check` under `select = ["ALL"]` on 2023 code to produce a large number of
  findings, mostly docstrings and typing. Fix them here rather than carrying `noqa` forward -
  but if a module is slated for deletion in phase 4, delete it there instead of polishing it now.

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
