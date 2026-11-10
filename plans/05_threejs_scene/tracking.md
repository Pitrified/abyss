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
| 1  | a public splat on the Pixel, rate tiers    | [`01_pixel_splat.md`](01_pixel_splat.md) | planned |
| 2  | training on g7, a public scene             | [`02_g7_training.md`](02_g7_training.md) | planned |
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
