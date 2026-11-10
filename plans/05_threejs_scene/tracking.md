# implementation tracking

A Gaussian splat in the phone page, drawn with three.js through abyss's frustum, and trained on g7.
This folder settles what is technically possible first: whether the Pixel can draw a splat and at
what rate, and whether g7 can train one. Analysis and decisions in [`00_start.md`](00_start.md).

## Key decisions

- **three.js in the existing page**, with Spark 2.4.0 for the splats (Q3). GaussianSplats3D is
  the fallback; its README says it is no longer in active development and points to Spark.
- **The view is split from the projection** (Q1): the three.js camera sits at the eye.
  Spark reads the camera's pose (`camera.matrixWorld`) to sort, so a camera left at the origin
  would sort from the wrong point.
- **Render on the Pixel first, training second** (Q2). Phase 1 uses a public pre-trained splat.
- **Training is gsplat** (Q4) with COLMAP for the poses (Q5), in its own environment outside
  abyss (Q6), on a public scene first (Q7).
- **Splats live in `~/data/splats/`**, compressed, served by the existing server (Q8).
- **Rate in tiers**: 30, 10 or 1 fps, from a splat-count sweep on both delegates (Q9).
- **Install what is needed on g7**, handed to the user as a script with a box-level note.
- Headless check with a synthetic splat (Q11); g7 render is one GPU flag tried once (Q12);
  glTF stays out unless a splat fails (Q13).

## Phases

| #  | Phase                                      | Plan | Status  |
| -- | ------------------------------------------ | ---- | ------- |
| 0  | CUDA toolkit and COLMAP on g7              | box note `~/repos/plans/2026-11-10_00_cuda_colmap.md` | done |
| 1  | a public splat on the Pixel, rate tiers    | [`01_pixel_splat.md`](01_pixel_splat.md) | done |
| 2  | training on g7, a public scene             | [`02_g7_training.md`](02_g7_training.md) | in progress |
| 3  | a Pixel capture, placed against the panel  | [`03_pixel_capture.md`](03_pixel_capture.md) | draft |

Status values: draft / planned / in progress / done / superseded / discarded.

Phases 1 and 2 are independent; 3 needs both.

## Log

Append-only. Newest at the bottom.

- 2026-11-10 : bootstrapped from the ranking in `02_scene_rendering/01_research_renderers.md`;
  branch `feat/05_threejs_scene`. Q1 to Q13 answered, Q1 and Q5 by the user's own words, the rest
  as recommended. Brush and `pycolmap` dropped: each existed only to avoid an install.
- 2026-11-10 : **phase 0 done.** The user ran the box script. `cuda-toolkit` 13.1.1 from Ubuntu's
  archive: `nvcc` 13.1.115 at `/usr/local/cuda-13.1/bin`, off PATH, `/usr/local/cuda` linked to it.
  A one-kernel test built with gcc 15.2 for `sm_75` ran correctly on the GPU; driver still 595.99.02.
  `colmap` 3.12.6-4 failed to start: it links `libPoseLib.so` but does not depend on `libposelib`.
  Found by `ldd`, checked by running it against the unpacked package, fixed by installing
  `libposelib` 2.0.5. `colmap -h` now reports 3.12.6 without CUDA.
- 2026-11-10 : checked for phase 1: Spark 2.4.0 on npm, MIT, released 2026-10-20, peer
  `three >=0.180.0` (the page has 0.186.1). It imports the bare specifier `"three"` and
  `"three/addons/postprocessing/Pass.js"`, so the page needs an import map; the page imports three
  by path today. Spark's own sample `butterfly.spz` is 4.0 MB. Mip-NeRF 360 is one 12.5 GB zip.
- 2026-11-10 : phase plans written and reviewed. Claim checked: Spark's module reads
  `camera.matrixWorld`, which is why Q1 matters. `plans.py check` flags the missing `priority` and
  `description` frontmatter, which this repo does not use (`plans/README.md`); left as is.
- 2026-11-10 : **phase 1 started.** Reviewed first: the plan holds, with two corrections found
  while building. The page's CSP refuses inline scripts, so the import map Spark needs is allowed by
  its SHA-256, computed by the server for each HTML page it sends (and on its 304s, which a browser
  stores headers from). Spark's `SplatMesh({ packedSplats })` leaves `numSplats` at 0 for a source
  that is already loaded; `SplatMesh({ splats })` sets it.
- 2026-11-10 : **matrix split done.** `eyeView` in `frustum.js` returns the eye and the projection
  apart, and their product; the three.js camera sits at the eye. Existing readback tests green
  unchanged; with the camera put back at the origin on purpose, 7 tests fail, the splat and
  wireframe readbacks among them.
- 2026-11-10 : **splats in the page.** Spark 2.4.0 vendored, imported only when `splat=` is given.
  `/splats/` serves `~/data/splats/`; a placement file per splat (position, quaternion, scale).
  `keep=` thins a splat evenly in the page, through Spark's `forEachSplat` and `pushSplat`, so one
  file gives every count. A five-point PLY written by `abyss.ui.splat` is read back in `make e2e` at
  Python-projected pixels, within 2 px, for three eyes. `make check` 310, `make e2e` 53.
- 2026-11-10 : public splats in `~/data/splats/`, from Spark's examples (licence not stated; used
  locally, not redistributed): `butterfly` 177k splats, `painted-bedroom` 500k,
  `greyscale-bedroom` 2M. All are y-down, z-forward; `[1, 0, 0, 0]` stands them up. Placements set by
  hand: butterfly at scale 0.04, the bedrooms at 0.025, all a few cm behind the panel. Headless
  frames show the butterfly upright and shifting with the eye; the bedroom at 0.05 filled the panel
  with one wall, at 0.025 it reads as a room.
- 2026-11-10 : SwiftShader draws real splats at about 2 s a frame (177k) to 5 s (500k), so they stay
  out of `make e2e`. **g7 render (Q12): yes, one flag.** `--use-angle=vulkan` (or `--use-angle=gl-egl`)
  gives headless Chromium the Quadro; the default and `--use-angle=gl` stay on SwiftShader.
  On the GPU: butterfly 60 fps (the frame cap), 2M splats at 40 fps, at 1082x2342.
- 2026-11-10 : `sweep=` added for the Pixel: one page load draws each share in turn with tracking
  running, 3 s to settle and 7 s measured, logs a tier per count (20 fps or more is 30, 5 or more is
  10, else 1) and a table to copy with the log button. Runbook section 6 written.
- 2026-11-10 : **Pixel run blocked**: port 8000 on g7 is held by `lang-tutor`'s uvicorn (since
  2026-11-08), and `tailscale serve` proxies `https://g7.tail5dff8b.ts.net/` to it, so the URL
  serves lang-tutor. Left running; how to serve abyss is for the user.
- 2026-11-10 : phase 1 committed. abyss moved to port 8001 (`DEFAULT_PORT`), served on the
  tailnet as `https://g7.tail5dff8b.ts.net:8443/` through a second `tailscale serve` entry
  (box note `2026-11-10_01_tailscale_serve_abyss_8443`); lang-tutor keeps 443 -> 8000.
- 2026-11-10 : **Pixel rate sweep**, `greyscale-bedroom` thinned with `keep=`, 366x793 css at
  2.625, tracking running. The two sweeps ran one after the other, the CPU one second.

  | splats | GPU delegate     | CPU delegate     |
  | ------ | ---------------- | ---------------- |
  | 100k   | 29.9 fps, tier 30 | 15.0 fps, tier 10 |
  | 500k   | 15.0 fps, tier 10 | 12.0 fps, tier 10 |
  | 1M     | 8.6 fps, tier 10  | 7.5 fps, tier 10  |
  | 2M     | 4.0 fps, tier 1   | 4.1 fps, tier 1   |

  Look runs on GPU: `butterfly` (177k) 20 to 24 fps, `painted-bedroom` (500k) 12 fps.
  The splat cost does not show in `draw` (0.5 to 0.9 ms throughout, the call only queues work);
  it shows in the landmarker time, which grows with the count on both delegates (GPU 36 to 236 ms,
  CPU 56 to 234 ms): the landmarker waits on the GPU the splat keeps busy, and Spark's sort worker
  shares the CPU. At 2M the GPU-run rate kept falling after the sweep, 4.0 to 2.9 fps over 45 s,
  likely heat. Tier 30 holds only up to about 100k to 200k splats; 500k and 1M are tier 10.
  The CPU run's camera took 133 s to start, not explained by the log.
- 2026-11-10 : **Pixel look**: butterfly upright, a little behind the glass. The bedroom showed less
  ceiling as the face lowered, which looked like a vertical flip. It is not one: rendered on g7's
  GPU from its capture point the room is upright (bed and floor below, beams above). At scale
  0.025 the room was about 7 cm tall with its ceiling near eye height, a dollhouse seen from
  outside, so a lower eye saw its roof edge-on. New placement for both bedrooms: scale 1, the
  splat's origin (its capture point) at a typical eye, `[0, 0.044, 0.25]` in the screen frame.
  Headless frames from a high, level and low eye now show floor, wall and ceiling in turn.
- 2026-11-10 : **phase 1 done.** With the new placement the painted bedroom on the Pixel reads as
  a room behind the glass, at 15 fps on the GPU delegate; the user: "It works". All "Done when"
  items met: checks green, effect seen, tiers and the g7 render logged.
- 2026-11-10 : phase 2 started; review before any install. Corrected: torch 2.15.0 with gsplat
  1.5.3 was a guess. gsplat's `main` examples pin torch 2.9.1 and torchvision 0.24.1, which have
  cu130 wheels for Python 3.12; torch 2.15.0 has only cu132 wheels, and the local nvcc is 13.1.
  The scene is Mip-NeRF 360 `room`, pulled out of the 12.5 GB zip by range requests (listing the
  zip's members that way worked). Found: the driver is now 595 (CUDA 13.2), not the 580 the repo
  instructions state, likely moved by phase 0's `cuda-toolkit` install.
- 2026-11-10 : **data and COLMAP.** `room` pulled from the remote zip by range requests: 396 MB
  (`images_4`, `images_2`, `sparse`) in 395 s, about 1 MB/s, into `~/data/splat_train/360_v2/room/`.
  **COLMAP from scratch on the CPU** (3.12.6, built without CUDA) on the 312 quarter-size images
  (779x519), one shared camera: features 22 s, exhaustive matching 776 s, mapper 230 s, 17 min in
  all. One model, 311 images registered (the shipped poses also have 311), 40.6k points,
  reprojection error 0.59 px (shipped: 112.6k points at full size, 0.86 px). Matching is the cost
  and grows with the square of the image count; a sequential or vocabulary-tree matcher is the
  lever for a larger capture in phase 3.
- 2026-11-10 : **environment.** gsplat cloned at `512d366b` into `~/repos/gsplat`, uv venv on
  Python 3.12. torch 2.9.1+cu130 and torchvision 0.24.1 installed in 915 s, nearly all download;
  the first try timed out on `pypi.nvidia.com` at uv's default HTTP timeout, `UV_HTTP_TIMEOUT=600`
  fixed it. torch sees the Quadro, compute capability 7.5.
- 2026-11-10 : **gsplat built.** `pip install -e .` with `TORCH_CUDA_ARCH_LIST=7.5`: 2315 s, while
  COLMAP held the CPU for the first part; gsplat reports 1.6.0. `examples/requirements.txt` (with
  fused-ssim, fused-bilagrid, ppisp built from git) took 129 s more. A 1000-Gaussian rasterization
  ran on the GPU.
- 2026-11-10 : **training.** gsplat's loader needs the full-size `images/` and re-derives the
  downscaled ones from it; it also scales the intrinsics to the images it reads. So `room_q/` links
  `images` to `images_4` and trains at factor 1, instead of pulling 969 MB more. AlexNet (233 MB)
  downloads on the first run, for LPIPS. Default strategy, 779x519:

  | steps | wall time | peak VRAM | Gaussians | PSNR | SSIM | LPIPS |
  | ----- | --------- | --------- | --------- | ---- | ---- | ----- |
  | 7k    | 3:56 train, 300 s all | 2.2 GB | 778k | 30.04 | 0.918 | 0.111 |
  | 30k   | 31:21 train, 1901 s all | 3.9 GB | 1.53M | 32.19 | 0.942 | 0.075 |

  6 GB is not the limit at this size: half-size images (`images_2`, already pulled) are the next
  step up if quality asks for it. The 30k peak includes headless page runs on the same GPU.
- 2026-11-10 : **convert and place.** Spark exports `transcodeSpz`, so the PLY becomes SPZ in
  headless Chromium through the served page: 360 MB to 31.9 MB in 20 s (7k: 184 to 14.1 MB).
  `abyss.ui.spz` holds it, with an e2e test that the test splat's SPZ loads with all its splats.
  gsplat's training frame is z-up and centred on the cameras (Mip-NeRF 360 room: cameras on a ring
  of radius 1.0 at height 0.6). `placement_at_camera` maps z to the screen's up and puts a capture
  camera at the typical eye, facing into the screen; `pick_camera` takes the one far out and facing
  the middle. Its pick, camera 179, looks at an armchair and the garden window; the camera picked by
  hand, 116, mostly at a dark TV. `scripts/train_splat.py` chains the three steps. `make check` 317
  passed, `make e2e` 54 passed.
