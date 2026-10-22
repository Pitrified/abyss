---
status: done
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
  ANS: no. Over the 120 frames of the synthetic clip, tasks-vision 1.0.1 against mediapipe 1.0.0,
  both on the CPU in VIDEO mode, the worst per-frame differences were 0.20 px on the iris midpoint,
  0.26 px on apparent IPD, 0.30 mm on depth and 0.054 deg on yaw, and both found the face in every
  frame. Measured 2026-10-22.

## Done when

- The cross-implementation and path tests are green under `make e2e`, with the tolerance recorded.
- `scripts/drive_ui.py` shows the clip with the eye position overlaid.

## What the implementation found

Done on 2026-10-22 in a cloud session. `make e2e` runs 20 tests in about 27 s; `make check` is
unchanged in what it runs.

- **`?source=` reads the MJPEG itself**, not through a `<video>` element. The page fetches the clip,
  splits it at JPEG start markers and decodes each frame with `createImageBitmap`, so every frame is
  seen once and in order, and the bytes are the ones Python decodes. `requestVideoFrameCallback`
  stays for the camera path.
- **The page retypes no number.** `/config.json` carries the device configs and every constant the
  port needs (MediaPipe's assumed field of view, the eye offset in the face model, the iris landmark
  indices, the bootstrap length, the filter width), read from the modules that define them.
- **The matrix is column-major.** tasks-vision returns the facial transformation matrix as a flat
  array without saying its order. Read column-major, the clip agrees with Python to 0.3 mm in depth;
  read row-major, five tests fail.
- **VIDEO mode lags fast motion, in both builds.** At peak speed the clip moves 22 px a frame and
  the tracked iris trails it by up to 10.5 px, against 0.81 px in IMAGE mode on single frames
  (phase 1). The two builds lag identically, so this is MediaPipe's tracking and not the port. The
  path test checks pixels only where the clip is still, and direction and depth everywhere.
- **Tolerances** are about twice the worst value seen: 0.5 px, 1 mm, 0.15 deg against Python;
  2 mm on positions; 2 px on the still frames; 6% on depth against the path, where 3.5% was seen.
- **Golden cases instead of a JS test runner.** `tests/e2e/test_port.py` builds inputs in Python,
  computes the expected outputs with the real Python functions, and runs the page's modules on the
  same inputs through DevTools. It covers the branches the clip never reaches: the mirrored Pixel
  camera, a measured focal length, a field of view, viewers turned away, a frame with no face.
- **The usage log is now seen being refused.** Closing the landmarker at the end of a clip flushes
  tasks-vision's log; the page's console shows the CSP refusing `odml.pa.googleapis.com`, and a test
  requires both the attempt and the refusal, so a green request check means something.
- **A config detail found by the tests:** `model_dump()` keeps tuples, which JSON returns as lists.
  The page config uses `model_dump(mode="json")`.

Every new check was seen failing: matrix order (five tests), the mirror flip, the even-count median.
The page tracks at roughly 55 ms a frame headless on the container's CPU, which says nothing about
the phone.

