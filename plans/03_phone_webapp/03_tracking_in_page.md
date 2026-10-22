---
status: planned
---

# Phase 2 - tracking in the page

## Overview

The page finds the face and turns it into a metric eye position, with the same maths as the Python
pipeline, checked against it on the same frames.

Context: phase 1 ([`02_harness_skeleton.md`](02_harness_skeleton.md)) for the harness. The maths
being ported is `src/abyss/viewer/eye_position.py` and `smoothing.py`, explained in
[`../../docs/library/viewer_position.md`](../../docs/library/viewer_position.md).

## What gets built

**Face landmarker in the page.** `FaceLandmarker` from the vendored `tasks-vision`, VIDEO mode, one
face, with facial transformation matrices on, since the depth comes from the matrix and not from
apparent IPD. The model is served by g7 from the vendor cache.

**`?source=<clip>`.** Reads frames from a `<video>` element instead of the camera and runs one
inference per frame with `requestVideoFrameCallback`, so every frame is seen exactly once and
results repeat. Writes per-frame results to `window.abyss` for a test to collect.

**The port**, as small ES modules, one per Python module: `extract_eye_sample`, head scale
(`estimate_head_scale` and `LiveScale`), `eye_position_m` with the frame geometry it needs, and
`PositionSmoother`. Names and frame conventions as in Python, so the two read side by side.

**Device numbers from Python.** The server exposes the camera, screen and viewer configs from
`abyss_devices.py` as JSON at a fixed path. The page fetches them; no device number is typed in JS.
Which device keys a page uses is a query parameter, defaulting to the Pixel's.

## Tests

- **Cross-implementation.** The synthetic clip through the Python pipeline and through the page with
  `?source=`. Per frame: iris pixels, depth, eye position. The two MediaPipe builds will not agree
  exactly, so the tolerance is measured on the first run, written down here with its date, and then
  enforced. A frame one side detects and the other does not is a failure, not a skip.
- **Against the path.** The clip moves along a known path, so the direction and the relative size of
  each movement are known exactly. Left in the clip is the sign of x, a scale change is a ratio of
  depths. Catches a sign error that both implementations share, which the first test cannot.
- **Unit tests of the port**, run inside the browser by Playwright, on the same inputs the Python
  unit tests use where those are plain numbers.
- **Fake camera path.** Through the real `getUserMedia` route: a face is found and the eye position
  moves the right way. Loose, since timing is not frame exact.

## Out of scope

The frustum and the scene. Performance on the phone, which is phase 4's to measure.

## Open questions

- Q1: **Does the WASM build disagree with the Python build enough to matter?** Measured by the first
  test. If it does, the fix is in the comparison's tolerance and in understanding why, not in
  tuning either side to match.
  ANS: ...

## Done when

- The cross-implementation and path tests are green under `make e2e`, with the tolerance recorded.
- `scripts/drive_ui.py` shows the clip with the eye position overlaid.
