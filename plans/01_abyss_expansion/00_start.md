---
status: draft
---

# Abyss expansion - own functionality and the render layer

Spun off from [`../00_template_alignment/00_feature_inventory.md`](../00_template_alignment/00_feature_inventory.md),
which covers the reboot: tooling, and deduplication against `pose-tools`. That initiative
deliberately stops at "abyss builds, lints, tests, and contains no duplicated code". Everything
about what abyss *does* lives here.

Not planned in phases yet - it starts when the template alignment is done and there is a clean
repo to build in.

## What this is

The README states the goal, and it has not changed since the first commit:

1. Compute the position of the viewer.
2. Compute the position of the screen.
3. Render the scene the viewer sees on the screen.

Step 1 is pose tracking, which `pose-tools` covers. Steps 2 and 3 are abyss's own, and step 3 -
the render layer - does not exist in any form today.

## The split with pose-tools

`pose-tools` is the **general** library: mediapipe landmarkers, video frame loading, landmark
arrays with visibility masking, homography, landmark distance, signal tracking. It is shared with
`climbing-wire` and `holo-table`, so nothing abyss-specific belongs there.

abyss adds its own functionality on top plus the render layer. The boundary test when something
new is written: would `climbing-wire` want this? If yes it goes upstream to pose-tools; if it is
about viewers, screens, or rendering a scene, it stays here.

abyss's identity is an **output** of this initiative, not an input. It gets assessed as the
functionality accumulates rather than declared up front.

## Known unknowns

Nothing here is decided; these are the questions this folder will have to open with.

- Screen geometry: is the screen pose known/calibrated up front, or recovered from the video?
  `pose-tools` has `geometry/homography.py` and `geometry/landmark_geometry.py`, which is the
  obvious starting material either way.
- Render target: offline frames, a live OpenCV window, or a browser view. The last one would
  reopen the FastAPI webapp scaffold question (#15 in the inventory), currently declined.
- Whether any of this justifies a Typer CLI entry point (#14, also currently declined).
- What "the scene" is: a 3D model rendered per viewpoint, or a reprojection of captured content.
  the deleted `utils/data.py` knew a `~/data/3d_models` folder, which hints at the former. Nothing
  reads it now - `AbyssPaths` deliberately dropped that entry - so it is a hint from history, not a
  live path.

## Prerequisites - all met as of 2026-08-10

- Template alignment is done, all five phases. `make check` is green and `src/abyss/` is
  `params/` + `metaclasses/` only.
- `pose-tools` is pinned at a released tag. Bump the pin to `v0.2.0` before starting - see the
  carried items in [`../00_template_alignment/tracking.md`](../00_template_alignment/tracking.md).
