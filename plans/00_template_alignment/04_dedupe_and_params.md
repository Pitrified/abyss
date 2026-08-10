---
status: planned
---

# Phase 4 - dedupe against pose-tools

## Overview

Removes the half of `src/abyss/` that `pose-tools` already carries, so there is one implementation
of each utility rather than two drifting copies. This is the phase the whole reboot exists for.

Q2 and Q3 are answered: **diff first, upstream anything unique, then delete** - and
**migrate rather than regenerate**. The `src/abyss/` scaffold stays; only the duplicated modules
leave. No renamer run, no wholesale `git rm -r src/`.

Depends on phase 3 (the pose-tools dep must already be declared and installable).

## Goals

1. No module in `abyss` duplicates a `pose_tools` module.
2. Everything abyss keeps is either specific to abyss or genuinely absent from pose-tools.
3. The notebook still runs against the new imports.

## Plan

- **Diff each pair before deleting anything.** The pose-tools versions are uniformly larger
  (`utils/mediapipe.py` 126 vs 180, `utils/cv.py` 36 vs 65, `video/load.py` 97 vs 120), but
  "larger" is not "a superset" - `video/frame.py` is the one that shrank, 90 to 84, so it is the
  first place to look for something lost.
- Anything abyss has that pose-tools lacks goes **upstream into pose-tools**, released as a
  `v0.2.0` there, and abyss bumps its pin. Do not keep a local fork of a general utility.
- Then delete `src/abyss/{utils/cv.py,utils/plt.py,utils/mediapipe.py,video/frame.py,video/load.py,
  landmarker/pose.py,landmarker/drawing.py}` and import from `pose_tools` instead. Note the module
  rename: abyss's `landmarker/` is `landmark/` in pose-tools.
- **No shims.** Call sites are rewritten to import from `pose_tools` directly. No re-export module
  keeping the old `abyss.utils.cv` path alive, no aliases, no deprecation wrappers, no
  `try: from pose_tools ... except ImportError` fallbacks. A file that has been folded into
  pose-tools leaves abyss entirely, and empty packages (`utils/`, `video/`, `landmarker/`) go with
  it rather than lingering as `__init__.py`-only directories. There are no external consumers of
  `abyss` to keep compatible - it is an app, and this is the phase that is allowed to break imports.
- Delete `utils/data.py` outright. `get_resource()` is a hand-rolled path registry with a
  `Literal` of five keys and an implicit `None` return on unknown input; the params layer added in
  phase 3 replaces it. Port **only the branches something actually uses** - of its five keys, carry
  over the ones with a live caller and drop the rest rather than transcribing all five into
  `AbyssPaths`. Same rule for env types and config models: no speculative entries, add them when a
  caller appears.
- Update `notebooks/sample01.ipynb` to the new imports and re-run it end to end. It is the only
  executable proof that the swap preserved behaviour, since there are no tests over this code.
  Caveat from phase 3: this box is headless, so any cell that opens an OpenCV window cannot run
  here - convert those to matplotlib/inline output or file writes, or run the notebook on a
  machine with a display.
- Whatever remains in `src/abyss/` after this is, by definition, abyss's own - which is the input
  to the expansion initiative, not a decision to take here.

## Out of scope

- Adding functionality to replace what was deleted.
- Restructuring what remains into a viewer/screen/render architecture - that is
  [`../01_abyss_expansion/00_start.md`](../01_abyss_expansion/00_start.md).
- Improving pose-tools beyond accepting whatever abyss upstreams.

## Done when

- No file under `src/abyss/` has a same-purpose counterpart in `pose_tools`, and nothing in
  `src/abyss/` merely re-exports from `pose_tools`.
- `uv run ruff check .`, `uv run pyright` and `uv run pytest` still pass.
- `notebooks/sample01.ipynb` runs top to bottom against the installed `pose-tools` (headless, so
  no `cv.imshow` cells), and its output is stripped before commit.
- If anything was upstreamed: a new pose-tools tag exists and abyss's pin points at it.
