---
status: done
---

# Phase 1 - a public splat on the Pixel, rate tiers

## Overview

Answers the first capability question: can the Pixel 7 Pro draw a splat in the page, through our
frustum, and at which rate tier. Uses a public pre-trained splat, so it does not wait on phase 2.
Context: [`00_start.md`](00_start.md) Q1, Q3, Q8, Q9, Q11, Q12.

## Goals

1. The three.js camera at the eye, the projection holding only the frustum, with the existing
   checks against Python still green.
2. Spark vendored and loading a splat from `~/data/splats/` through the server.
3. A synthetic splat checked headless at Python-projected pixels.
4. The Pixel's rate tier for several splat counts, on both delegates.

## Plan

- **Matrix split (Q1).** `frustum.js` returns the projection and the eye translation separately,
  `viewProjectionMatrix` stays as their product for the checks. `scene.js` sets the projection on
  the camera and a translation to the eye on its world matrix (and the inverse on
  `matrixWorldInverse`). Test first: a check that the camera's world position equals the eye, plus
  the existing frame readback.
- **Vendor Spark.** `@sparkjsdev/spark` 2.4.0 in `ui/vendor.toml` (`dist/spark.module.js`), and
  three's `examples/jsm/postprocessing/Pass.js` added to the three entry. An import map in
  `index.html` for `"three"` and `"three/addons/"`, and `scene.js` switched to the bare specifier,
  so the page and Spark share one three.js instance.
- **Serve splats (Q8).** An `AbyssPaths` entry for `~/data/splats/`, and a server route for it
  with the same ETag and gzip handling as the vendor files. `.spz` is already compressed.
- **Scene choice.** A `?splat=<name>` query parameter selects a splat; without it the page draws
  the wireframe as today. A hand-set transform per splat (Q10) in a small JSON file beside it.
- **Synthetic splat test (Q11).** A handful of Gaussians at known screen-frame points, generated
  at test time, read back at Python-projected pixels, the way the cube is checked. It must fail
  on a mirrored or vertically flipped frame.
- **Public splat.** Spark's sample `butterfly.spz` (4.0 MB) into `~/data/splats/`, licence checked
  first; a larger public scene for the sweep, found during the phase.
- **Rate sweep (Q9).** One scene subsampled to about 100k, 500k, 1M and 2M splats, each placed in
  a tier (30, 10, 1 fps) with tracking running, on the CPU and the GPU delegate. Read from
  `window.abyss.timing` and the log button, as in `03_phone_webapp`.
- **g7 render (Q12).** Run the page in Chromium on g7 once with a GPU flag (`--use-angle=vulkan`
  or similar). If it does not just work, record that and stop.

## Out of scope

- Training (phase 2) and the Pixel capture (phase 3).
- Optimising the tracking; the rate is recorded, not tuned.
- glTF scenes (Q13).

## Pitfalls

- Spark spawns workers from `blob:` URLs, which the CSP allows (`worker-src 'self' blob:`);
  whether it needs anything else the CSP refuses is seen on the first load.
- Spark's sort may run asynchronously in a worker, so a frame read back right after a camera move
  can show the previous sort. The synthetic test should wait for a settled frame.
- A large splat over Tailscale's relay was measured at 70 to 140 kB/s. Check the path is `direct`
  before timing a load.

## Done when

- `make check` and `make e2e` pass, with the synthetic splat test and the matrix split checks among them.
- The Pixel draws the public splat through the frustum, and the effect is seen by the user.
- The rate tier per splat count and delegate is logged in `tracking.md`.
- The g7 render result is logged, either way.

## What the implementation found

- Spark sorts from `camera.matrixWorld`, so the camera has to sit at the eye; a mutation that left
  it at the origin failed 7 tests.
- The inline import map needs the CSP to allow it by hash; the server computes the hash per page.
- `SplatMesh({ splats })` draws a loaded `PackedSplats`; `packedSplats:` drew nothing.
- Headless SwiftShader takes minutes per frame of a real splat, so the tests use a five-point
  synthetic one. On g7, `--use-angle=vulkan` puts Chromium on the Quadro: 2M splats at 40 fps (Q12).
- Pixel tiers, tracking running: 100k is tier 30 on the GPU delegate, 500k to 1M tier 10, 2M
  tier 1; the CPU delegate is tier 10 from 100k. The splat cost shows in the landmarker time, not
  in `draw`. Table in `tracking.md`.
- Placement is what makes a captured room work: scaled down to fit behind the glass it reads as a
  dollhouse seen from outside. At scale 1 with the splat's capture point at a typical eye, the
  viewer is inside the room and the panel behaves as a window.
