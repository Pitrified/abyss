---
status: planned
---

# Phase 4 - a minimal scene through the window

## Overview

The first phase that produces something worth looking at, and deliberately no more than that.

Context: [`00_start.md`](00_start.md). Consumes phase 3's
[`03_off_axis_projection.md`](03_off_axis_projection.md) matrix and phase 1's
[`01_viewer_position.md`](01_viewer_position.md) eye track. Phase 5 replaces the offline loop with a
live one. A real renderer is [`../02_scene_rendering/`](../02_scene_rendering/) and nothing here
waits on it.

Everything so far is numbers. Phase 3's own tests establish that the matrix is right, but nobody has
*seen* the effect, and a sign error that survives a green suite shows up here as a scene that slides
the wrong way. So this phase is as much a verification of phase 3 as it is new work.

It also owns `SinkConfig`, deferred from phase 2 to the phase where its first caller appears.

## Where the seam actually goes

Q4 asked for the simplest scene behind a seam, and the obvious reading is wrong, so it is worth
stating before any code.

The tempting seam is the scene: an interface that yields geometry, with a wireframe box today and a
loaded 3D model later. That seam is in the wrong place. The named second implementation is
`02_scene_rendering`, an OpenGL renderer, and **it does not produce geometry, it produces pixels**.
A geometry-shaped interface would exclude the very implementation it exists for, which is the test
`00_start.md` sets: a seam only counts when the second implementation can be named, and it has to be
able to plug into the thing that is being called a seam.

So the seam is one step up:

    Renderer.render(view_projection, width_px, height_px) -> BGR image

A numpy wireframe satisfies it today. A GL context that binds the same 4x4 and reads the framebuffer
back satisfies it later, and so does a splat renderer, and so does a captured-content reprojection.
The scene is then an implementation detail of the wireframe renderer, not a shared contract, which is
also why it does not need a Protocol of its own (Q22).

The sink is a genuine second seam, on Q13's argument: it acts on a frame that already exists.

| Seam | Now | Named second |
| ---- | --- | ------------ |
| renderer | `WireframeRenderer`, numpy and `cv.line` | GL, `02_scene_rendering` |
| sink | `PngSink`, frames to `cache/render/<run>/` | `WindowSink`, phase 5 |
| eye source | a phase 1 CSV, or a synthetic sweep | a live tracker, phase 5 |

## What the scene is

One primitive: a 3D line segment in the screen frame. A box, a grid and a floating object are all
segments, so there is one thing to project and one thing to draw.

**A box whose mouth is the screen.** Front face exactly the panel rectangle at `z = 0`, back face at
`z = -depth`, default 0.6 m. The mouth is the window, so it must land on the image border for every
eye position, which is phase 3's corner invariant made visible: if the border ever moves, the
projection is wrong, and you can see it without reading a number.

Contents, each earning its place:

| Piece | Why |
| ----- | --- |
| 12 box edges | the room, and the border self-check |
| a grid on the back wall | parallax needs a reference to slide against |
| one mid-line per side wall | tells the eye the walls recede, cheaply |
| a small cube floating mid-depth, ~0.06 m at `z = -0.25 m`, off-centre | parallax is most legible as relative motion between two things at different depths, so the cube moving against the back grid *is* the effect |

Around 40 segments. Depth-faded colour: near edges bright, far edges dim, the cheapest depth cue
there is. No filled surfaces, so the scene stays one primitive and a wireframe throughout (Q21).

**The scene lives entirely behind the window**, `z <= 0`, and that is load-bearing rather than
incidental. It means no point is ever at or behind the eye, so `project_points` cannot raise and
phase 4 needs no clipper at all. The named upgrade is homogeneous-space segment clipping against the
near plane, needed the first time something is meant to poke out through the window. Recorded so that
the absence of a clipper reads as a decision rather than an oversight.

Painter ordering: sort segments back to front by midpoint depth and draw in that order, so nearer
lines land on top. One argsort, no z-buffer. Wireframe has no occlusion, so this is cosmetic, and
saying so keeps anyone from mistaking it for hidden-line removal.

## What gets built

| Piece | Where | Role |
| ----- | ----- | ---- |
| `SinkConfig` | `src/abyss/config/sink.py` | the fourth config model, finally with a caller |
| `Sink`, `PngSink` | `src/abyss/sink.py` | Protocol carrying `size`, plus today's implementation |
| `Scene`, `window_box(...)` | `src/abyss/render/scene.py` | segments and the builder |
| `Renderer`, `WireframeRenderer` | `src/abyss/render/renderer.py` | the seam and the numpy implementation |
| `render_scene.py` | `scripts/` | the offline loop, CSV or synthetic sweep |

A module `src/abyss/sink.py`, not a package: there is one implementation. It becomes a package in
phase 5 when the window sink arrives, which is a rename, not a redesign.

Nothing new in `AbyssPaths`. Output goes under the existing `cache_fol`, per the phase 2 rule that
path entries get added when something needs them.

## The aspect trap

Phase 3 maps the panel corners to the viewport corners *whatever the viewport is*. Render a 1.78
panel into a 1.33 image and it fills it perfectly, stretched, with nothing raising and nothing
looking obviously broken. So the check has to be explicit and it has to live where both numbers meet,
which is the render entry point, not either config.

It cannot be an equality check. `g7_internal` is 344 x 193 mm, aspect **1.7824**, while 1280x720 is
**1.7778**: a real panel is not exactly 16:9, and an exact test would reject the actual device on day
one. So `AspectMismatchError` above a 2% tolerance, which passes g7's 0.26% and catches any genuine
mix-up.

## Decisions

- **The eye track comes from phase 1's CSV, not from re-running the tracker.** Deterministic, fast,
  and it needs no model download or clip present, so the phase stays testable on g4. The per-frame
  work goes in a function taking one eye position, and the script is a thin loop over it, so phase 5
  swaps the source without touching the render path.
- **A synthetic sweep is a first-class mode, not a test fixture.** A left-to-right, near-to-far eye
  path with no clip involved is the only artefact that can be produced on a box with no camera and no
  data, and it is what a reviewer looks at. It doubles as the regression input.
- **A contact sheet is written alongside the frames**: nine sweep positions tiled into one PNG. Over
  ssh, one image that shows the parallax beats ninety that have to be flicked through.
- **The sink owns the output resolution, and the `Sink` protocol exposes it as `size`.** Pixels are
  what a sink writes; metres are what the screen is. The render call takes width and height as
  arguments and the loop feeds it `sink.size`, so `PngSink` can take its size from `SinkConfig` while
  phase 5's `WindowSink` takes its own from the window. See Q20.
- **The clip cameras are still unmeasured, so the depths are wrong**, and the rendering will be
  correct *for the eye positions it is given*. Not a defect of this phase, and the artefacts should
  say so rather than implying the metres are trustworthy.

## Tests

The phase 3 lesson applies directly and is the reason this list is split. A test built from the
system's own outputs cannot catch an error upstream of both sides of the comparison, so each test
below is labelled with which kind it is.

Self-consistency, which is cheap and worth having anyway:

- the box mouth lands on the image border for a swept grid of eye positions, now through the renderer
  rather than through `project_points` directly
- the same eye position renders byte-identical images twice
- an empty scene renders the background and does not crash

Physics, which is what would actually catch a regression in the chain:

- moving the eye to the viewer's right moves the back wall's drawn centroid **right** in the image,
  the same direction phase 3's parallax test pins, now measured off pixels
- the floating cube and the back wall move by **different amounts** for the same eye motion, since
  equal motion would mean the depth is not reaching the projection at all
- a frame is not blank: some minimum fraction of pixels differ from the background, which is the
  guard against the classic silent failure where everything projects off-screen and the suite still
  passes

Plus the boundaries:

- `AspectMismatchError` above tolerance, and no error at g7's real 0.26%
- `PngSink` writes the filenames it claims, zero-padded so a glob sorts correctly
- `SinkConfig` rejects a non-positive size, as the other three models reject theirs

## Out of scope

- A real renderer, textures, shading, models, GL. That is `02_scene_rendering` and this phase must
  not grow into it.
- Occlusion or hidden-line removal. Wireframe, painter ordering, stated as cosmetic.
- Near-plane clipping. Ruled out by keeping the scene behind the window, with the upgrade named.
- A live window or interactive rate. Phase 5.
- Compositing the camera image next to the render. It sounds useful for debugging and it belongs to
  phase 5, where a camera is actually in the loop.

## Open questions

- Q20: **Does the output resolution belong on `SinkConfig`, or is it an argument to the render call?**
  On the sink it travels with "what happens to a finished frame", which is what Q13 says a sink is,
  and one object then describes the whole output side. As an argument it stays with the caller who
  chose it, and the sink is purely a destination. The tie-breaker is probably phase 5: a window sink's
  size is the window's, not something a config picked.
  ANS: **Both, because the question had a false exclusive in it.** Asked as "where is the number
  stored" it needs phase 5 to settle; asked as "who owns the number" it settles now. The owner is the
  sink, so the `Sink` protocol carries a `size` property and the loop is
  `renderer.render(matrix, *sink.size)`. `PngSink` takes its size from `SinkConfig`, phase 5's
  `WindowSink` takes it from the window it opened, and the render path never learns which. So the
  resolution is a `SinkConfig` field *and* a render argument, with the sink reconciling them.
  No hack and no deferred refactor. Even had it gone the other way the cost was bounded: the number
  is not persisted in any format, so being wrong is a one-line change at two call sites, not a
  migration. Worth recording that the bounded-cost check is what made this safe to decide early,
  rather than confidence in the answer.
- Q21: **Is a depth-faded wireframe enough to read as a box, or does the back wall need to be
  filled?** A wireframe cube is Necker-ambiguous and can flip inside-out perceptually, which would
  make the parallax read backwards to the eye even though the maths is right. Depth fading may be
  enough. Filling the back wall with `cv.fillPoly` is a few lines and removes the ambiguity, at the
  cost of the scene no longer being a single primitive.
  ANS: **Start with no fill.** Depth-faded wireframe only, and the scene stays one primitive. If the
  box reads ambiguous when there is finally something to look at, the fill is added then. This costs
  nothing to defer because it is entirely inside `WireframeRenderer`: no interface, no config and no
  test outside that module changes, which is the difference between deferring a decision and
  deferring a design.
- Q22: **Should `Renderer` be a Protocol now, or after the second implementation exists?** The seam
  argument above says the second implementation is real and named, which is the project's stated bar.
  The counter is the project's other stated bar, that abstraction needs a case that demands it today,
  and one implementation does not. Cheap either way, so it is about which rule wins when they point
  opposite ways.
  ANS: **Protocol now, for both `Renderer` and `Sink`**, and the apparent conflict was a misreading
  rather than a real tension. "A case that demands it today" is about *speculative* futures, and
  these are not speculative: `02_scene_rendering` is a written-down initiative and phase 5's window
  sink is the next phase. Second and third implementations can both be named and are clearly in
  scope, they simply arrive tomorrow rather than today.
  The correct reading of the simplicity rule, recorded so it stops being applied too strictly: it
  bans abstraction built for futures that cannot be named, not abstraction built for work that is
  already planned. Scheduled is not speculative.

## Done when

- The sweep produces frames and a contact sheet in which the parallax is visible to the eye, and the
  box mouth stays welded to the image border throughout.
- A phase 1 CSV drives the same path and produces frames for a real clip.
- `SinkConfig` exists and is passed in, never looked up, per the phase 2 rule.
- `make check` is green, and the suite still passes with no clip, no camera and no model present.
- The regression CSVs are untouched: this phase adds a consumer and changes nothing upstream, so
  `scripts/viewer_position.py` output must stay byte-identical.
