---
status: draft
---

# Scene rendering - a real renderer behind the scene seam

Spun off from [`../01_abyss_expansion/00_start.md`](../01_abyss_expansion/00_start.md) when the
phases there were reassessed (Q14). That initiative keeps the *minimal* scene: whatever is cheapest
to draw that makes the head-coupled effect visible, so the projection maths can be seen to work.
This folder is everything past that point.

## Why it is separate

Nothing in the expansion waits on it. The off-axis frustum is verified by unit tests and by the
minimal scene; a better renderer changes what is drawn, not whether the projection is right. It also
carries its own dependencies and its own machine constraint, neither of which the expansion has:

- a GPU context, so it is a g7 target, not g4
- a rendering library (`moderngl`, `pyglet`, `pyrender`) that abyss does not depend on today
- assets, and a place for them outside the repo

## What it inherits

The scene seam from the expansion: the renderer is a swappable component, receiving a projection
matrix and a screen config, returning a frame. Anything here has to fit behind that interface.

Two candidates were parked in the expansion's "Tools suggested, not evaluated" section and belong
here rather than there:

- **OpenGL** via a Python binding. The natural fit, since the phase 3 projection matrix is exactly
  what a GL pipeline consumes. Open question whether offscreen rendering (EGL / OSMesa) works well
  enough to develop any of it on the headless box.
- **Gaussian splatting** (or NeRF, which it has largely displaced for real-time work) to render
  novel views of a captured real place. Heavier: training needs the GPU, and only splatting renders
  fast enough for an interactive loop.

## Open questions

Numbering is local to this folder.

- Q1: **Which renderer?** Depends on what the minimal scene turns out to be, and on whether the
  answer has to run headless for development.
  ANS: ...
- Q2: **Where do assets live?** The deleted `utils/data.py` knew a `~/data/3d_models` folder. Data
  lives outside the repo by convention, so this is a path entry in `AbyssPaths` when it is real.
  ANS: ...
