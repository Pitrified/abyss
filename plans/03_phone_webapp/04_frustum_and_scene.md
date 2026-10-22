---
status: planned
---

# Phase 3 - frustum and scene in the page

## Overview

The eye position from phase 2 becomes an off-axis projection, and three.js draws a scene through
it. The end of this phase is the effect itself, in headless Chromium, driven by the synthetic clip.

Context: phase 2 ([`03_tracking_in_page.md`](03_tracking_in_page.md)). The maths being ported is
`src/abyss/render/frustum.py`, explained in
[`../../docs/library/off_axis_projection.md`](../../docs/library/off_axis_projection.md).

## What gets built

**The frustum port.** `eye_in_screen_frame`, `frustum_for_eye`, `projection_matrix`,
`view_projection_matrix`, with the same named errors. Handed to three.js by setting
`camera.projectionMatrix` and its inverse directly; three.js's own perspective camera is not used,
since it cannot express an off-axis frustum.

**The scene.** The same box, grid and floating cube as the expansion's minimal scene
(`src/abyss/render/scene.py`), so the first three.js picture can be compared against a picture the
Python wireframe renderer already draws. A real scene is
[`../02_scene_rendering/`](../02_scene_rendering/)'s to choose, and swapping it here is a content
change.

**The loop.** Each frame: landmarker, eye position, smoother, matrix, draw. Frame time and each
stage's share published on `window.abyss`, as the Python loop's benchmark does.

## Tests

- **Golden vectors.** Python generates eye positions and screen rectangles, including the cases its
  own tests pin (centred eye gives a symmetric frustum, an eye behind the screen raises), and the
  resulting matrices, at test time. The page computes the same and the test compares to float32
  precision. Generated, not committed, so they never go stale.
- **Projected points.** Scene vertices projected by the JS matrix against `project_points` in
  Python, in pixels.
- **The mirror hazard.** A fixed-matrix render with an asymmetric scene, checked for which side the
  cube is on. The one defect the live demo cannot reveal, as the renderer survey says.
- **A picture.** Screenshots at a few fixed eye positions, compared against references the way
  klide's frame gate does, once the scene stops changing. Until then, saved for a person to look at.

## Out of scope

Any real device. Choosing the real scene.

## Done when

- `make e2e` is green with the golden-vector and projected-point tests.
- A screenshot sequence from the synthetic clip shows the scene shifting with the head, the right way.
