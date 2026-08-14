---
status: planned
---

# Phase 1 - viewer position from a clip

## Overview

Recorded video in, a per-frame 3D eye position out, written to a CSV and a plot so it is checkable
without a display. Context: [`00_start.md`](00_start.md), depends on phase 0, which shipped as
`pose-tools@v0.4.0` and is pinned.

This is the phase where the scale problem gets answered rather than deferred, so the measurements
below were taken **before** planning, against `~/data/pose/face01.mp4` (250 frames, a face in every
one).

## What the clip already told us

All figures measured, not assumed. `depth` is `-tz` from `facial_transformation_matrixes`.

| Measurement | Value | What it means |
| ----------- | ----- | ------------- |
| `ipd_px * depth` | constant to **2.1%** | the matrix depth obeys a pinhole model: apparent size times distance is fixed |
| `iris_px * depth` | constant to **4.1%** | same, via a different feature |
| implied focal from IPD (63 mm) | **984 px** | on a 1920-wide frame, i.e. ~88 degrees horizontal |
| implied focal from iris (11.7 mm) | **1031 px** | agrees with the above to 5% |
| `corr(ipd_px, abs(yaw))` | **-0.76** | yaw foreshortens the interpupillary distance badly |
| `corr(iris_px, abs(yaw))` | **-0.21** | the iris is nearly rotation-invariant |
| `corr(iris_px, depth)` | +0.02 | over this clip the subject barely moves in depth, so neither cue is exercised much |

Three conclusions the design follows from:

1. **MediaPipe's metric depth is an assumption wearing a number.** ~88 degrees is far wider than any
   interview camera; it is MediaPipe's own default camera, not this one. Two independent assumptions
   are buried in the output, and they do *different* things - worth being precise, because the first
   draft of this plan got it wrong:

   From the pinhole relations, with `S` the true size of a facial feature and `s_px` its apparent
   size: `Z = f * S / s_px`, and `X = (u - cx) * Z / f = (u - cx) * S / s_px`.

   | Assumption wrong by | Effect on depth `Z` | Effect on lateral `X`, `Y` |
   | ------------------- | ------------------- | -------------------------- |
   | focal length, factor `k = f_real / f_assumed` | scales by `k` | **unchanged** - the `f` cancels |
   | head size / IPD, factor `h` | scales by `h` | scales by `h` |

   So a wrong FOV does not scale the trajectory uniformly, it **stretches depth against lateral**,
   which changes the shape of the viewer's path and therefore the frustum. It is a distortion, not a
   unit conversion. Phase 2 supplies `f_real` and fixes it; nothing here waits for that, but the
   phase must keep the two factors separate so each can be corrected on its own.
2. **Do not derive depth from interpupillary distance.** It is the textbook choice and it is the
   wrong one here: it collapses under yaw, which is exactly what a viewer's head does. The iris
   diameter is the robust apparent-size cue, and the transformation matrix is better still because
   it already accounts for head pose.
3. **The self-consistency check is free and worth keeping.** `iris_px * depth` staying constant is a
   cheap invariant that catches a broken pipeline immediately.

## Goals

1. A per-frame eye position in the camera frame, from a recorded clip.
2. The scale assumption explicit, isolated, and replaceable by phase 2 without touching callers.
3. Output that a headless box can check: a CSV and a plot.

## Plan

- **Eye position.** Midpoint of the two iris centres (`FACE_RIGHT_IRIS_CENTER` 468,
  `FACE_LEFT_IRIS_CENTER` 473 from `pose_tools.utils.mediapipe`). Single cyclopean eye, per Q6.
- **Depth** from `get_facial_transformation_matrix()`, times the scale factor `k`. Where the matrix
  is absent (option off, or no face) the frame yields no position rather than a guessed one.
- **Lateral position** from the pinhole model: `X = (u - cx) * Z / f`, `Y = (v - cy) * Z / f`, with
  `f` and the principal point from the same placeholder config as `k`. Not from the matrix's `tx` /
  `ty`, which are in the canonical model's frame rather than pixels.
- **The scale seam.** One small module owning `f_assumed`, `f_real`, the principal point, and the
  physical size reference (interpupillary distance, per Q6). Today's values are the measured
  `f_assumed = 1000 px` with `f_real = f_assumed`, so `k = 1` and the numbers come out as MediaPipe
  intends them. Phase 2 replaces it with the real camera config. This is a placeholder to be
  deleted, not a parallel config system - it holds four numbers and says so in its docstring.
- **Smoothing** with `create_left_triangle_filter` and `roll_append_smooth` from
  `pose_tools.utils.np_signal`, per axis, on the metric position. Raw and smoothed both go to the
  CSV so the effect is visible rather than asserted. The filter is causal - weights rise towards the
  newest sample and sum to 1 - which is what a live loop will need later, so offline and live behave
  the same.

  **Not `SignalTracker`**, despite it being named for this job in `00_start.md` and in
  `tracking.md`. Reading it settled the matter: it is a gesture classifier. `update()` returns the
  doubly-smoothed *derivative* when the signal passes three threshold checks and `0.0` otherwise,
  and the smoothed value is only reachable as a side attribute (`all_values_s[-1]`). Using it would
  mean inventing `sd_min` / `sd_max` / `sdsd_max` values that mean nothing for an eye position, to
  reach a number it computes internally with the primitives above. Those primitives are the actual
  smoothing layer, and `SignalTracker` is a consumer of them, not a wrapper worth borrowing.
- **Outputs**: a CSV of `idx, msec, iris_u_px, iris_v_px, iris_dia_px, yaw_deg, depth_raw,
  x/y/z raw and smoothed`, and a matplotlib figure of the three axes over time, raw against
  smoothed. Both go under `AbyssPaths.cache_fol`, which already exists and is gitignored, so this
  phase adds no path entry - the params layer stays minimal until something actually needs more.
- **Where the code goes**: `src/abyss/viewer/` - `eye_position.py` for the extraction and
  `smoothing.py` for the filter state, with the script driving them in `scripts/`. First real
  subpackage in abyss; `tests/` mirrors `src/abyss/` without repeating the package name, so the
  tests land in `tests/viewer/`, matching the existing `tests/params/`.
- **Tests** cover the maths, not MediaPipe: the pinhole conversion against hand-computed values, the
  scale factor applied where it should be, a missing matrix producing no position, and the
  `iris_px * depth` invariant on synthetic input. The clip is not in git, so no test may read it.

## Out of scope

- Real camera intrinsics and screen geometry: phase 2. This phase must not grow a config system, it
  gets a placeholder with three numbers in it.
- Calibration of any kind. Q2 deferred it and nothing here reopens that.
- Any use of the head orientation beyond recording yaw in the CSV. A frustum needs eye position, not
  gaze direction. Head pose becomes interesting only if the scene ever reacts to where the viewer
  looks, which is not this project.
- Live capture. Q5 says offline; the frame source is a seam, so the switch is a later, small change.

## Done when

- The script processes `face01.mp4` end to end and writes both artefacts.
- Eye depth over the clip is stable to within a few percent, matching what the matrix already shows,
  and the `iris_px * depth` invariant holds in the produced numbers.
- Changing `f_real` in the placeholder scales depth and leaves lateral position untouched, while
  changing the size reference scales all three. Both are asserted in tests, because this is the
  property phase 2 depends on and the one I got wrong when drafting.
- Smoothed output is visibly less jittery than raw in the plot, with the tracker's parameters
  recorded in the log rather than tuned silently.
- `make check` is green, and the suite still passes with no clip and no model present.
