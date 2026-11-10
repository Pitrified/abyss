---
status: planned
---

# Phase 2 - training on g7, a public scene

## Overview

Answers the second capability question: can g7 train a splat in 6 GB of GPU memory.
A public scene with known-good images first, so a failure points at the pipeline, not a capture.
Context: [`00_start.md`](00_start.md) Q4 to Q8. Needs phase 0's CUDA toolkit and COLMAP.

## Goals

1. gsplat installed in its own environment outside abyss, its CUDA kernels built on g7.
2. One public scene trained to a splat, with VRAM and wall time recorded.
3. The result converted to `.spz` in `~/data/splats/` and drawn by the page (on the Pixel once phase 1 is done).

## Plan

- **Environment (Q6).** A uv project outside the repo (location chosen at the start of the phase),
  with torch 2.15.0 and gsplat 1.5.3 and gsplat's example requirements. `CUDA_HOME` set if torch
  does not find `/usr/local/cuda` by itself. First run builds the kernels; time it.
- **Data (Q7).** One Mip-NeRF 360 scene, which comes with COLMAP poses already. The full set is
  one 12.5 GB zip; look for a single-scene source first.
- **Train.** gsplat's `simple_trainer` at its defaults, with the image downscale factor raised
  until it fits in 6 GB. Record peak VRAM (`nvidia-smi`), wall time, and the Gaussian count.
- **COLMAP (Q5).** Run once on the same scene's images from scratch, without the shipped poses,
  to prove the CPU path and time it; phase 3 depends on it.
- **Convert (Q8).** The trained `.ply` to `.spz`, with whatever tool Spark or gsplat provides.
- **Script.** A script in abyss `scripts/` that calls the environment, so the run is repeatable.

## Out of scope

- A Pixel capture (phase 3).
- Tuning training quality beyond fitting in memory.

## Pitfalls

- A training run is long machine time. Each run is proposed first, with its expected duration.
- 6 GB may force a heavy downscale; record the quality cost rather than chase it here.

## Done when

- One public scene trained on g7, numbers logged in `tracking.md`.
- COLMAP's CPU run timed on the same images.
- The trained splat drawn by the page.
