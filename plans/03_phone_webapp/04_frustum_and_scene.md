---
status: done
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

## What the implementation found

Done on 2026-10-22 in a cloud session, verified from a clean clone: `make check` 294 passed,
`make e2e` 29 passed in about 33 s.

- **The scene is served, not ported.** The plan had three.js draw "the same box, grid and cube".
  Rather than port `scene.py`, `/scene.json?screen=<key>` serves the segments and colours
  `window_box` builds, and the page draws whatever arrives. The geometry cannot drift from Python's,
  and swapping in a real scene from `../02_scene_rendering/` is a different document, not a change
  to the page. The constants the frustum needs (near and far planes, the camera-to-screen axis
  signs, the aspect tolerance) come through `/config.json` like the phase 2 ones.
- **The handoff to three.js is a bare `THREE.Camera`** at the origin with `matrixAutoUpdate` off,
  whose projection matrix is abyss's whole view projection. three then applies projection x
  identity x identity, which is exactly the matrix computed. `Matrix4.set` takes row-major
  arguments; `fromArray` would take the same numbers column-major.
- **The picture check reads the canvas, not a reference image.** `preserveDrawingBuffer` lets a
  test pull the WebGL frame back as a PNG. For seven points along the inner part of every segment,
  projected by Python, at least 97% must land on a drawn pixel (within one pixel). That catches what
  the numbers cannot: transposing the handoff left every numeric test green and failed all four
  picture tests. Committed reference images, as klide does, are still deferred until the scene
  stops changing.
- **The mirror hazard is checked in pixels.** The cube sits at +8 cm, so it must draw right of
  centre for an eye on axis, and moving the eye to the viewer's right (camera frame -x, unmirrored)
  must move it further right. Dropping the x flip in the frame conversion fails five tests,
  including this one.
- **`?eye=x,y,z`** draws one frame with no tracker at all, which is what makes the picture tests
  deterministic. `?size=WxH` fixes the output size; without it the page fits the panel's aspect
  inside the window and letterboxes, since the frustum is built for the physical panel and a
  stretched picture is wrong everywhere.
- **The clip drives the scene.** With `screen=`, each track entry carries the matrix it drew with;
  every one equals Python's `view_projection_matrix` for the page's own smoothed eye to 1e-9
  relative, and frames before the head scale freezes draw nothing.
- **Test file names must be unique across folders**, since `tests/` subfolders are not packages:
  `tests/e2e/test_scene.py` collided with `tests/render/test_scene.py` and broke collection of the
  whole suite. It is `test_page_scene.py`.

No screen for the Pixel exists yet, so every check runs against `g7_internal`; the phone's panel is
phase 4.

