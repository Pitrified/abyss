---
status: done
---

# Phase 3 - off-axis projection

## Overview

Eye position plus screen rectangle to a projection matrix. This is the payoff: the point where
"where is the viewer" and "where is the screen" become "what should be drawn".

Context: [`00_start.md`](00_start.md). Consumes phase 1's
[`01_viewer_position.md`](01_viewer_position.md) eye position and phase 2's
[`02_camera_screen_model.md`](02_camera_screen_model.md) `ScreenConfig`. Phase 4 draws through
what this produces.

Pure maths, no camera, no clip, no model download. Every test runs headless and offline, which is
why this phase is worth doing carefully: it is the last one where correctness is cheap to
establish. A sign error here is invisible in a unit test that only checks shapes and obvious in
phase 5 as a scene that slides the wrong way.

Both inputs are now measured rather than guessed, which is new since this phase was sketched:
`g7_webcam` has `focal_px=945.0`, and `g7_internal` has a real panel and a real camera offset.

## The frustum is not the risk

Given the eye at `(ex, ey, ez)` relative to the screen centre, the screen `w` by `h`, and a near
plane at `n`, the asymmetric frustum is four divisions:

    left   = (-w/2 - ex) * n / ez
    right  = ( w/2 - ex) * n / ez
    bottom = (-h/2 - ey) * n / ez
    top    = ( h/2 - ey) * n / ez

and the projection matrix is the standard `glFrustum` form built from those. That is well-trodden
and will work the first time.

**The risk is the coordinate handoff**, and it is worth being explicit about because three frames
meet here and two of them disagree about which way is up.

| Frame | Convention | Where it comes from |
| ----- | ---------- | ------------------- |
| camera | `+X` image right, `+Y` image **down**, `+Z` away from the camera, metres | phase 1, `eye_position_m` |
| screen | origin at panel centre, `+X` viewer's right, `+Y` **up**, `+Z` out of the screen toward the viewer | this phase, new |
| clip | OpenGL normalised device coordinates, `+Y` up, `-Z` forward | the projection matrix output |

So the phase has exactly one interesting function and it is not the frustum: it is the one that
takes an eye position in the camera frame and a `ScreenConfig`, and returns the eye in the screen
frame. Subtract `camera_to_centre_m`, then flip the axes that disagree.

Worked out rather than assumed, **both X and Y flip and Z does not**:

    x_screen = -(x_camera - offset_x)
    y_screen = -(y_camera - offset_y)
    z_screen = +(z_camera - offset_z)

- **Y flips** because the camera frame points `+Y` down while the screen frame points `+Y` up.
  `camera_to_centre_m` is expressed in the camera frame, which is why g7's offset is a *positive*
  0.1005 for a camera sitting *above* the panel. Getting this wrong puts the eye below the screen
  when it is above it.
- **X flips too**, which is the less obvious half. The camera looks *at* the viewer, so the
  viewer's right hand appears on the **left** of an unmirrored image, the same way a person facing
  you has their right hand on your left. Image left is negative camera X, so the viewer's right is
  camera `-X`. This is exactly why video call apps mirror the self view.
- **Z does not flip.** Camera `+Z` points away from the lens, which is toward the viewer, and the
  screen frame's `+Z` points out of the panel, also toward the viewer.

Together that is a 180 degree rotation about Z, which is the right shape for two frames that face
each other, and a proper rotation rather than a reflection.

One further trap: **the X sign also depends on `mirrored`, and that is already handled upstream.**
`eye_position_m` flips X when `FrameGeometry.mirrored`, so this phase must not flip it a second
time. `g7_webcam` is not mirrored; `pixel7pro_front` is. The bug would be invisible on g7 and wrong
on the phone, so a test should pin that this phase treats the incoming X as already correct.

## What gets built

`src/abyss/render/frustum.py`, a new subpackage, since this is neither viewer tracking nor config.

| Piece | Role |
| ----- | ---- |
| `eye_in_screen_frame(eye_camera_m, screen)` | the coordinate handoff above, returns a 3-vector |
| `Frustum` | frozen dataclass, `left/right/bottom/top/near/far` in metres at the near plane |
| `frustum_for_eye(screen, eye_camera_m, near_m, far_m)` | the four divisions |
| `projection_matrix(frustum)` | the 4x4, `glFrustum` convention |
| `view_projection_matrix(screen, eye_camera_m, near_m, far_m)` | the above plus the eye translation, so callers stay in the screen frame |

Plus one convenience for phase 4, which does not have OpenGL and will be drawing with numpy:

| Piece | Role |
| ----- | ---- |
| `project_points(matrix, points_m, width_px, height_px)` | world points to pixel coordinates, doing the perspective divide and the viewport transform |

The 4x4 is the real deliverable and `project_points` is a thin wrapper over it, not a parallel
implementation. Phase 4 calls the wrapper; the matrix is what gets tested.

## Decisions

- **No screen rotation.** `ScreenConfig` deliberately does not model it, for the reason recorded
  there: a laptop camera sits in the lid bezel, so it does not move relative to the panel when the
  lid tilts, and one offset holds for every lid angle. The axis-aligned frustum above is therefore
  exact here, not an approximation.
  The general form is Kooima's generalised perspective projection, which takes three screen corners
  and derives an orthonormal screen basis, handling a tilted or off-angle display for free. That is
  the named replacement if an external monitor ever appears, so this is a seam in the sense
  `00_start.md` requires: one implementation now, a second one we can name.
- **Near and far are arguments, not config.** They are properties of the scene being drawn, not of
  the screen, and nothing has a scene yet. Defaults of 0.05 m and 100 m, chosen so the near plane
  sits well inside any plausible eye distance.
- **The near plane is not the screen plane.** A common confusion worth stating once: the screen
  sits at distance `ez`, the near plane at `n`, and the frustum extents are the screen rectangle
  scaled by `n / ez`. Setting `n = ez` is legal but not required.

## Tests

The three the sketch named, plus the one that actually matters.

- **Centred eye gives a symmetric frustum.** `ex = ey = 0` implies `left = -right` and
  `bottom = -top`, and `right / top` equals the screen's aspect ratio.
- **Moving the eye shifts the frustum the right way.** Eye to the viewer's right moves `left` and
  `right` together in the direction that keeps the screen fixed in the world. Pinned with an
  explicit expected sign, not just "it changed".
- **Corners map where they should.** For a centred eye the screen corners land on the viewport
  corners.
- **The invariant that defines head-coupled perspective**: the four screen corners map to the four
  viewport corners **for every eye position**, not only the centred one. That is what makes the
  screen behave like a window rather than a picture.

**Corrected after implementing, by deliberately breaking each sign and watching what went red.**
The plan originally claimed the corner invariant would catch the frame conversion too, and that if
only one test survived it should be that one. That is wrong, and wrong in an instructive way.

The corner invariant is a **self-consistency** check. It verifies that `frustum_for_eye`,
`projection_matrix` and `project_points` agree with each other. If the frame conversion is wrong,
the frustum is built for the wrong eye position and the corners *still* fill the image perfectly,
because the error is upstream of the part being checked. Measured: flipping the X or Y sign in the
conversion leaves all 45 swept cases green.

So two independent families are needed, and neither substitutes for the other:

| Family | Catches | Blind to |
| ------ | ------- | -------- |
| corner invariant, swept | near-plane scaling, the matrix, the viewport transform, the eye translation | the camera to screen conversion |
| directional and parallax | the conversion signs, which way the world moves | an internally consistent pipeline that is uniformly wrong |

The directional tests are the ones that pin physics: moving to the viewer's right shifts the
frustum left, moving up shifts it down, and a point deep behind the window drifts across the image
the way a distant tree does. Exactly three tests fail when an axis sign is flipped, and all three
are of that family.

Plus the error cases, since the project raises named exceptions rather than producing quiet
nonsense:

- eye in or behind the screen plane (`ez <= 0`) raises, rather than dividing by zero or silently
  inverting the image
- `near <= 0`, or `far <= near`, raises
- a degenerate screen cannot occur: `ScreenConfig` already validates `width_m` and `height_m` as
  positive, so this phase must not re-validate them

One integration check with the real numbers: `g7_internal` at a plausible 0.5 m eye distance should
give a horizontal field of view of roughly 38 degrees, which is a sanity figure to assert loosely
rather than a measurement to pin exactly.

## Out of scope

- Anything drawn. Phase 4 owns the scene, and this phase's output is checked numerically.
- The sink config, still deferred to phase 4 with its first caller.
- Stereo, one frustum per eye. Ruled out in `00_start.md`; the single cyclopean eye stands.
- Smoothing. Phase 1's `PositionSmoother` already runs upstream, and smoothing a matrix rather than
  a position would be the wrong place.
- Any OpenGL. The matrix follows the `glFrustum` convention because it is the standard everyone
  documents, not because anything binds a GL context.

## Open questions

- Q18: **Does the viewport transform belong here or in phase 4?** `project_points` is the only
  piece that needs to know pixels, and pixels are arguably a sink concern.
  ANS: **Here.** Phase 4 would otherwise reimplement the perspective divide, which is exactly where
  sign errors breed, and the point of this phase is that the sign work happens once in a place with
  tests around it.
- Q19: **Should the eye position carry its frame in the type?** Three frames meet in this phase and
  they are all bare 3-vectors of floats, so nothing stops a camera-frame vector being passed where
  a screen-frame one is wanted.
  ANS: **Document it, do not type it, and type it the moment it travels.** The rule being applied:
  contained within one place means prose is enough; passed along means it earns a type.
  That makes a design constraint rather than just a note, because the screen-frame vector is
  currently produced by one function and consumed by the next. To keep it contained, **every public
  entry point takes the eye in the camera frame**, and the screen-frame vector never crosses out of
  this module. `eye_in_screen_frame` stays public because it is worth testing and debugging
  directly, but nothing outside `abyss.render` is expected to hold its result.
  The trigger to revisit is explicit: if any caller outside this module ever needs a screen-frame
  eye position, it is being passed along, and it gets a `NewType` then.

## Done when

- `abyss.render.frustum` exists with the pieces above, and nothing in it reads config from a
  registry: a `ScreenConfig` is passed in, per the phase 2 rule that configs are passed rather than
  looked up.
- The corner invariant holds across a swept grid of eye positions, not just the centred case.
- The real `g7_internal` and a real eye position from a clip produce a frustum whose numbers are
  plausible, checked once and recorded in the log rather than asserted tightly.
- `make check` is green, and the suite still passes with no clip, no camera and no model present.
- The regression CSVs are untouched: this phase adds a consumer and changes nothing upstream, so
  `scripts/viewer_position.py` output must stay byte-identical.
