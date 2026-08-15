---
status: planned
---

# Phase 1 - viewer position from a clip

## Overview

Recorded video in, a per-frame 3D eye position out, written to a CSV and a plot so it is checkable
without a display. Context: [`00_start.md`](00_start.md), depends on phase 0, which shipped as
`pose-tools@v0.4.0` and is pinned.

This is where the scale problem gets answered rather than deferred, so everything below was measured
before planning. The first draft of this plan was reviewed by a second agent with fresh context and
several of its numbers did not survive; what follows is the corrected version.

## What the clips tell us

Three assets, all in `~/data/pose/`, none in git. Provenance in that folder's README.

| Clip | Geometry | Role |
| ---- | -------- | ---- |
| `face01.mp4` | 1920x1080, 250 frames | the reference: a face in every frame, near-constant depth |
| `face02_portrait.mp4` | 1080x1920, 239 frames | a second camera geometry and a second subject |
| `face03_zoom.mp4` | 1920x1080, 250 frames | `face01` with a **known** 1.00x to 1.60x zoom ramp: ground truth for depth |

### MediaPipe's assumed camera is a function of frame height

The landmarker gets no intrinsics, so it assumes a **63 degree vertical** field of view. Fitting the
focal length by reprojecting MediaPipe's own metric pose onto the measured iris pixels:

| Clip | Height | Fitted `f` | `(H/2)/tan(31.5 deg)` | Ratio |
| ---- | ------ | ---------- | --------------------- | ----- |
| `face01` | 1080 | 900 px | 881 px | 1.021 |
| `face02_portrait` | 1920 | 1600 px | 1567 px | 1.021 |

The same 2% offset at two different heights, on two different subjects: the law holds, and the
residual says the true default is nearer 62 degrees. So `f_assumed` is **computed from the frame**,
never configured. Two consequences the code must respect:

- Reprojection rms is 2.5 px at the fitted focal and **24 px at 984 px**, the figure the first draft
  quoted. That number was `ipd_px * depth / 63 mm`, which multiplies the camera assumption by a
  head-size assumption and so measures neither. It is gone.
- Anything that changes frame height changes depth. Measured: padding `face01` to 1920x1920 with
  identical pixel content moves `tz` from -50.7 to -88.1 cm. Letterboxing is not a no-op.

### The metric scale is per-identity, and that is the real problem

With `f` fixed by the law above, `ipd_px * depth / f` recovers the head size the solve is implicitly
using:

| Clip | Implied interpupillary distance |
| ---- | ------------------------------- |
| `face01` | 70.4 mm (sd 1.5) |
| `face03_zoom` (same subject) | 71.7 mm (sd 1.4) |
| `face02_portrait` (different subject) | 62.2 mm (sd 0.7) |

Same library, same law, **13% apart between subjects**. The solve fits an identity-dependent mesh,
so its metric output carries a per-person scale error of that order. That is the scale problem in
its true form: not one 63-versus-70 mm constant to correct once, but a per-person factor. Q6 already
chose interpupillary distance as the reference, which is what makes it correctable.

### Depth is validated against ground truth, not against footage

No recorded clip carries a measured distance, so a real "walk towards the camera" video would only
produce a plausible curve. `face03_zoom` produces a checkable one: a digital zoom of `z` multiplies
apparent size by `z`, so reported depth must divide by `z`.

Measured over the ramp against per-frame ground truth: **mean error +1.97%, sd 1.35%, worst 5.83%**,
across a 1.56x depth range with detection on all 250 frames. That is the acceptance test this phase
needs, and it is worth more than footage because the answer is known in advance.

Its limitation, stated so nobody mistakes it for the real thing: a digital zoom changes apparent
size without changing perspective or parallax. It validates the size-to-depth conversion, not how
the tracker behaves when a real head approaches.

### The cue: the matrix, corrected to the eye

Interpupillary distance in pixels is the textbook depth cue and the wrong one here:
`corr(ipd_px, abs(yaw)) = -0.76`, so it collapses exactly when the viewer turns their head. Iris
diameter is nearly pose-invariant but noisy - 24 px wide, and its derived depth has a 4.7% spread
against 1.0% for the matrix.

The matrix wins, with one correction. `tz` locates the **model origin**, not the eye. Fitting a
fixed eye point in the model frame puts it 2.5 cm above and 3.0 cm in front of that origin, so on
`face01`:

```
Zeye - tz : mean -2.69 cm, peak-to-peak 0.92 cm, corr(yaw) -0.66
```

A yaw-coupled error of about a centimetre over only ±19 degrees of yaw - the same failure the IPD
cue was rejected for. The fix is one line: `Z_eye = -(R @ e_model + t)_z`.

## Decisions taken

| Question | Decision |
| -------- | -------- |
| Units | **Metres** throughout abyss. MediaPipe emits centimetres; convert once where the matrix is read and never carry cm past it. Phase 2's screen geometry and phase 3's frustum are both metric, so metres is the low-friction choice and cm lives inside one function. |
| Head-size scale | **Corrected in this phase**, not deferred, as a per-session identity factor. |
| Mirroring | **A field on the camera config**, because it varies per capture device. Defaults to false; a phone front camera will want true. |
| Smoothing location | Its own `smoothing.py`. The review argued for folding it into `eye_position.py`; keeping it separate is the call, so filter state does not spread through the extraction code. |

The identity factor, concretely: estimate `ipd_model = median(ipd_px * Z / f)` over the frames where
`abs(yaw)` is small, once, then hold it. Scale positions by `ipd_person / ipd_model`, with
`ipd_person` a config value defaulting to 63 mm. That removes the per-subject bias while staying a
constant, so it does not reintroduce the yaw sensitivity that ruled IPD out as a per-frame cue.

## Goals

1. A per-frame eye position in the camera frame, in metres, from a recorded clip.
2. The camera and identity assumptions isolated, each correctable on its own.
3. Output a headless box can check: a CSV and a plot.

## Plan

- **Frame convention**, in the output type's docstring and asserted in a test: origin at the camera
  optical centre, **+X image right, +Y image down, +Z away from the camera** (OpenCV), units metres.
  MediaPipe's matrix is **y-up**, so `ty` flips sign on the way in - measured, `corr(pinhole Y,
  matrix ty) = -0.967`.
- **Eye position.** Midpoint of the two iris centres for the pixel location
  (`FACE_RIGHT_IRIS_CENTER` 468, `FACE_LEFT_IRIS_CENTER` 473) and `Z_eye = -(R @ e_model + t)_z` for
  depth, with `e_model` the fitted constant. Single cyclopean eye, per Q6.
- **Lateral position** by pinhole: `X = (u - cx) * Z / f_real`, `Y = (v - cy) * Z / f_real`, with
  `f_real` named explicitly - the `Done when` property below holds for `f_real`, not `f_assumed`.
  Not from the matrix `tx`/`ty`: they are genuinely metric camera-frame values
  (`corr(X, tx) = 0.991`), but they locate the model origin rather than the eye and carry the y-up
  flip.
- **The seam**, one small module holding `f_assumed` computed as `(H/2)/tan(31.5 deg)` from the
  frame, `f_real` (defaulting to `f_assumed`), the principal point, `ipd_person`, and `mirrored`.
  Phase 2 replaces it with the real camera config. A placeholder to be deleted, not a config system.
- **Smoothing** in `smoothing.py`, over `create_left_triangle_filter` and `roll_append_smooth` from
  `pose_tools.utils.np_signal` - causal, weights summing to 1, so offline and live behave alike.
  **Not `SignalTracker`**, which despite its name in `00_start.md` is a gesture classifier:
  `update()` returns a thresholded derivative and the smoothed value is only a side attribute.
  Initialise the history from the first sample, or the first frames ramp up from zero and dominate
  the plot.
- **Missing faces.** A frame with no face, or no matrix, yields no position - never an interpolated
  one. The filter holds state across the gap, and the CSV records the gap. All three clips detect on
  every frame, so only tests exercise this path.
- **Landmarker setup**, pinned rather than left to chance: VIDEO running mode, `num_faces=1`,
  `output_facial_transformation_matrixes=True`, timestamps strictly increasing (`detect` passes
  `int(frame.msec)`, and MediaPipe rejects a non-monotonic sequence).
- **Outputs** under `AbyssPaths.cache_fol`, which exists and is gitignored: a CSV of `idx, msec,
  iris_u_px, iris_v_px, ipd_px, depth_m, x/y/z raw and smoothed, has_face`, and a matplotlib figure
  of the three axes over time, raw against smoothed.
- **Where the code goes**: `src/abyss/viewer/` - `eye_position.py`, `smoothing.py`, and the driving
  script in `scripts/`. `tests/` mirrors `src/abyss/` without repeating the package name, so tests
  land in `tests/viewer/`, beside the existing `tests/params/`.
- **Tests**, none of which may read a clip or a model, since neither is in git:
  - a committed fixture of ~20 rows of `(u, v, 4x4 matrix)` extracted once from `face01`, so the
    whole matrix-to-eye-position conversion is covered offline
  - the pinhole conversion against hand-computed values, including the y-up flip and the sign for a
    landmark left of the principal point
  - `f_assumed` equals `(H/2)/tan(31.5 deg)` at several frame heights
  - changing `f_real` scales depth and leaves lateral untouched; changing `ipd_person` scales all
    three
  - a missing matrix yields no position, and the filter survives the gap
  - `mirrored=True` flips the sign of X and nothing else

## Out of scope

- Real camera intrinsics and screen geometry: phase 2. This phase gets a placeholder holding five
  numbers, not a config system.
- Camera calibration. Q2 deferred it. The identity factor is not camera calibration - it is one
  number about the person, and it is what makes the output metric at all.
- Head orientation as an output. `R` is used internally for the eye offset and the yaw gate, but no
  Euler angles are exported: that conversion is a general utility, and by the boundary rule it
  belongs upstream in pose-tools if anything ever needs it.
- Live capture. Q5 says offline; the frame source is a seam.

## Done when

- The script processes all three clips and writes both artefacts for each.
- On `face03_zoom`, reported depth tracks the known ramp to **within 3% mean error** - the
  measurement above reaches +1.97%. This is the criterion that means something: the earlier draft
  asked only that depth be "stable to a few percent", which a constant would pass.
- The identity factor brings `face01` and `face02_portrait` onto one scale: with `ipd_person` set
  equal, their implied head sizes agree to a few percent rather than 13%.
- Changing `f_real` scales depth only; changing `ipd_person` scales all three axes. Both asserted.
- Smoothed output is visibly less jittery than raw in the plot, with the filter width recorded in
  the log rather than tuned silently.
- `make check` is green, and the suite passes with no clip and no model present.
