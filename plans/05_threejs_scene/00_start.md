---
status: in progress
priority: 1
description: |
  A Gaussian splat in the phone page, drawn with three.js (Spark) through abyss's frustum,
  trained on g7 with gsplat. Settles first whether the Pixel can draw one and g7 can train one.
---

# three.js scene - a splat on the phone

Picked 2026-11-10 from the ranking in
[`../02_scene_rendering/01_research_renderers.md`](../02_scene_rendering/01_research_renderers.md), section "On the phone":
three.js in the existing phone page, which leads both the phone port and the beauty lists once a
Gaussian splat counts as content rather than an engine.
`02_scene_rendering` stays the generic renderer folder; this one is the three.js route only.

This folder first settles what is technically possible, before any scene is chosen:

1. **Pixel render.** Can the Pixel 7 Pro draw a splat in the page, through our frustum, at which rate tier (30, 10 or 1 fps)?
2. **g7 training.** Can g7 train a splat from a capture?
3. **g7 render.** Only if it is nearly free. Otherwise this feature is the phone.

## Current state

- The page draws with three.js 0.186.1 (`ui/vendor.toml`), wireframe only (`ui/phone_web/js/scene.js`).
- The camera is a bare `THREE.Camera` at the origin. The eye translation is folded into the
  projection matrix (`viewProjectionMatrix` in `ui/phone_web/js/frustum.js`), so three.js believes
  the camera is at the origin. A wireframe does not care, but anything that depends on the camera
  position does: splat depth sorting, and the specular light of glTF materials. See Q1.
- On the Pixel the landmarker takes 28 ms of the 33 ms camera frame (GPU delegate), so the tracking
  alone runs at 30 fps. A splat shares that GPU.
- g7, checked 2026-11-10: Quadro RTX 3000 Max-Q, 6 GB, 332 MB used at idle, driver 595.99.02
  (the repo instructions still say 580). No CUDA toolkit (`nvcc` absent), no COLMAP, ffmpeg present.
  30 GB RAM, 828 GB free disk. `gsplat 1.5.3` and `torch 2.15.0` resolve for Python 3.14 and 3.12
  (`uv pip compile`, nothing installed). torch 2.15.0 is built against CUDA 13.2.
- Ubuntu 26.04's own archive has `cuda-toolkit` 13.1.1 and `colmap` 3.12.6 (CPU-only).
  A simulated install pulls 83 packages and leaves the driver untouched. gcc is 15.2.0.
- Installed 2026-11-10. `nvcc` 13.1.115 at `/usr/local/cuda-13.1/bin`, off PATH, with `/usr/local/cuda`
  linked to it; a one-kernel test built with gcc 15 for `sm_75` ran correctly on the GPU.
  Driver still 595.99.02. `colmap` installed but failed to start: the package links `libPoseLib.so`
  without depending on `libposelib`. With `libposelib` 2.0.5 installed, `colmap -h` reports
  `COLMAP 3.12.6 ... without CUDA`.

## Decisions

- **three.js**, in the existing page. Unity or Godot only if the browser runs out.
- **Capabilities before content.** Each probe answers one of the three questions above, at the rate tier,
  without millisecond precision.
- **Training is a content tool, not an abyss dependency** (for now): it produces a file the page
  serves. See Q6.
- **Split the view from the projection** (Q1): the three.js camera moves to the eye, the
  projection holds the frustum only.
- **Install what is needed on g7, no workarounds** (user, 2026-11-10). System installs are fine,
  handed to the user as one script to run from Termux, with a box-level plan note
  (`~/repos/plans/2026-11-10_00_cuda_colmap.md` for CUDA and COLMAP).
  Options that existed only to avoid an install were dropped: Brush as the trainer (Q4), because
  it avoided the CUDA toolkit, and `pycolmap` for the poses (Q5), because it avoided apt.

## Open questions

Numbering is local to this folder.

- Q1: **Split the matrix into view and projection?** Today the eye translation is inside the
  projection matrix and the three.js camera sits at the origin. Splat renderers sort by distance
  from the camera position, so they would sort from the wrong point, and glTF specular highlights
  would come from the wrong place.
  - a. Split in the page: the projection from the frustum only, the camera's world matrix a
    translation to the eye. The product is unchanged, so the existing checks against Python still hold.
  - b. Keep it folded, and accept wrong sorting.
  - Recommended: a, because it is small and every non-wireframe scene needs it.
  - ANS: a, "ok we solve it".
- Q2: **Which probe first?**
  - a. Pixel render first, with a public pre-trained splat. Training follows, or runs in parallel.
  - b. g7 training first, then render our own splat.
  - Recommended: a, because the two are independent, and the render is closest to what the user sees.
  - ANS: a, as recommended (user, 2026-11-10).
- Q3: **Which web splat library?**
  - a. Spark (<https://sparkjs.dev/>), built for three.js.
  - b. GaussianSplats3D (<https://github.com/mkkellogg/GaussianSplats3D>). Its maintenance status needs checking.
  - c. Both, compared on the same splat.
  - Recommended: a, with b as the fallback if a does not take our camera.
  - ANS: a, as recommended (user, 2026-11-10).
- Q4: **Which trainer on g7?**
  - a. gsplat's `simple_trainer`, built against the CUDA toolkit installed on g7.
  - b. Nerfstudio `splatfacto` (gsplat underneath, a much larger environment, plus its own capture tooling).
  - Recommended: a, because it is the smaller of the two and reported lean on VRAM, which matters at 6 GB.
  - Dropped: Brush (Rust on wgpu), whose only edge here was avoiding the CUDA toolkit install.
  - ANS: a, as recommended (user, 2026-11-10).
- Q5: **Camera poses for training.** Every trainer needs the capture's camera poses, which come from COLMAP.
  - a. `pycolmap` from PyPI, CPU, no system install.
  - b. COLMAP from apt, a system install with a plan note.
  - Recommended: a, unless it lacks a step b has.
  - ANS: b. "we can install whatever we need", so a's only advantage is gone.
    COLMAP 3.12.6 from apt, CPU-only; a CUDA build from source only if extraction is too slow.
- Q6: **Where do the training tools live?**
  - a. Their own environment outside abyss (own uv project or binary), plus a script in abyss that calls them.
  - b. A dependency group in abyss's `pyproject.toml`.
  - Recommended: a, because torch, CUDA and pinned Python versions should not touch abyss's 3.14 environment.
  - ANS: a, as recommended (user, 2026-11-10).
- Q7: **First training data.**
  - a. A small public scene (from the Mip-NeRF 360 set), to prove the pipeline. Then a Pixel capture.
  - b. A Pixel capture straight away.
  - Recommended: a, because a failure then points at the pipeline, not at the capture.
  - ANS: a, as recommended (user, 2026-11-10).
- Q8: **Where do splats live, and in which format?**
  The Pixel has fetched from g7 at 70 to 140 kB/s over Tailscale's relay; a raw `.ply` splat runs to hundreds of MB.
  - a. `~/data/splats/` outside the repo (a path entry in `AbyssPaths` when it is real), served by
    the existing server, in a compressed format (`.spz`, or the library's own).
  - b. Raw `.ply`.
  - Recommended: a.
  - ANS: a, as recommended (user, 2026-11-10).
- Q9: **How is the Pixel's rate measured?** One scene subsampled to several splat counts
  (about 100k, 500k, 1M, 2M), each placed in a tier (30, 10, 1 fps), with tracking running, on both
  delegates, since the GPU delegate shares the GPU with the splat.
  - ANS: as proposed (user, 2026-11-10).
- Q10: **Placing the scene.** A trained splat is in COLMAP's frame, at an arbitrary scale with no
  metric units. It has to be placed and scaled against the panel, and that placement is what makes or
  misses the depth "wow".
  - a. A hand-set transform per scene, in a small file next to the splat.
  - b. Fitted automatically (a ground plane, a chosen point at the panel).
  - Recommended: a for now; b once there is more than one scene.
  - ANS: a, as recommended (user, 2026-11-10).
- Q11: **Headless test for the splat path.** SwiftShader in `make e2e` will be slow on a real splat.
  - a. A synthetic splat of a few Gaussians at known screen-frame points, read back at
    Python-projected pixels, the way the cube is checked today. This also catches a mirrored or flipped frame.
  - b. No headless check; device only.
  - Recommended: a.
  - ANS: a, as recommended (user, 2026-11-10).
- Q12: **g7 render.** The cheapest route is the same page in Chromium on g7, which the harness already
  drives, but on SwiftShader. Asking it for the GPU (`--use-angle=vulkan` or similar) is one flag to try.
  A Python render through gsplat comes free only if gsplat ends up installed for training.
  - a. Try the GPU flag once. If it does not just work, drop g7 render from this feature.
  - b. Drop it now.
  - Recommended: a.
  - ANS: a, as recommended (user, 2026-11-10).
- Q13: **A glTF scene with real lighting.** It needs no capability probe: three.js loads glTF, and Q1 covers the camera.
  - a. Out of this feature, unless a splat fails.
  - b. Also a phase here, as the fallback scene.
  - Recommended: a.
  - ANS: a, as recommended (user, 2026-11-10).

Batch 2, for phase 3, raised 2026-11-10 with the first capture (a metro carriage) already in hand.
The capture guide they refer to is in [`03_pixel_capture.md`](03_pixel_capture.md).

- Q14: **Motion for a capture.** The panel is a window: the viewer's eye moves over a patch roughly
  30 cm wide and 20 cm tall, 15 to 40 cm from the glass, always looking through it. A splat is only
  sharp from viewpoints near where it was captured, and depth only comes from the camera moving,
  not turning. The metro video swayed sideways, then turned in place.
  - a. Sweep a plane: the phone faces the scene the whole time and slides left-right at three or
    four heights, then forward and back once, covering a patch a bit larger than the eye's.
  - b. Orbit a subject, as Mip-NeRF 360 does: good for walking around an object, wasted on a
    window that is only looked through from one side.
  - c. Free-hand, as the metro video.
  - Recommended: a, because it captures exactly the viewpoints the window will be seen from, plus a
    margin; turning in place adds no depth.
  - NEW_ANS:
- Q15: **Camera settings.** Motion blur and per-frame changes are what spoil a splat: exposure,
  focus and white balance shifting, the lens switching, stabilisation warping the frame.
  - a. 4K at 30 fps, 1x lens, exposure and focus locked (long press), stabilisation off if the
    Pixel camera allows it.
  - b. 1080p at 60 fps, same locks: a shorter shutter, less blur, half the detail.
  - c. Default settings, as the metro video (1080p30, stabilised).
  - Recommended: a in good light, b in dim light; frames are downscaled for training anyway.
  - NEW_ANS:
- Q16: **Length and frames.** Phase 2 trained 312 images in 32 minutes; the frame count changes the
  matching and posing time, not the training time, which is set by the step count.
  - a. 30 to 60 s of slow movement, about 200 to 300 frames taken from it.
  - b. 10 to 15 s, as the metro video, about 150 frames.
  - Recommended: a, because a slow sweep at three heights needs that long.
  - NEW_ANS:
- Q17: **Placement of a capture.** The room placed the eye at one capture camera at scale 1.
  A phone capture has a real scale in metres only up to COLMAP's unknown factor.
  - a. Eye at the capture's centre camera, scale set by eye so near objects sit at the glass.
  - b. Measure one known distance in the scene (a door width) and scale to metres.
  - Recommended: a for now; b if objects look the wrong size.
  - NEW_ANS:
- Q18: **The splat budget for a capture.** Phase 1 and 2 tiers: about 500k splats for about 15 fps,
  1.5M for 10 fps, with tracking.
  - a. Train at the defaults and thin on the phone with `keep=`.
  - b. Cap the count in training (gsplat's MCMC strategy takes a maximum).
  - Recommended: a first, b if `keep=` visibly degrades the scene.
  - NEW_ANS:

## Phases

The plan of record is [`tracking.md`](tracking.md); first sketch:

0. CUDA toolkit and COLMAP on g7, by the user's script; verified by building one CUDA kernel.
1. Matrix split (Q1), and a splat in the page on the Pixel with a public splat: rate tiers.
2. Training on g7 with a public scene, the result shown on the Pixel.
3. A Pixel capture, trained on g7, placed against the panel.
