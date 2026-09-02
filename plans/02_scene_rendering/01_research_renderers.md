---
status: draft
---

# Renderer research

Survey of what could sit behind the `Renderer` seam, gathered before choosing.
No decision here: this is the material Q1 of [`00_start.md`](00_start.md) is answered from.

## What is being chosen for

The seam is already fixed by phase 4 of the expansion (`src/abyss/render/renderer.py`):

```python
class Renderer(Protocol):
    def render(
        self, view_projection: np.ndarray, width_px: int, height_px: int
    ) -> np.ndarray: ...
```

A 4x4 view projection matrix and a pixel size in, a BGR `uint8` frame out.
The geometry behind that matrix is written up in
[`../../docs/library/off_axis_projection.md`](../../docs/library/off_axis_projection.md).
The matrix is built by `view_projection_matrix` from the viewer's eye position and the panel rectangle;
it is an off-axis asymmetric frustum, the `glFrustum` form with `left != -right` and `bottom != -top`.

Five constraints. They are **not equally weighted**, and the weighting below comes from the user's
notes of 2026-09-02 rather than from the survey; it is recorded here because it changes which
candidates are worth effort, not which ones exist.

- **C1 matrix in.** The library must accept a projection matrix that abyss computed,
  not just a field of view and an aspect ratio.
  A library that only exposes `set_fov()` cannot express an off-axis frustum, and reconstructing one
  through its camera parameters means reimplementing the maths the repo already has and tests.
  **Hard.** Without it the projection this whole repo exists to compute is not the one being drawn.
- **C4 seam fit.** Whether the renderer can be swapped without the loop knowing which one it has,
  and whether two of them can coexist in one process. **The deciding constraint**, weighted above
  everything else here: the point of the seam is to hold several renderers over time, so a candidate
  that fits it badly costs more later than it saves now. Broken out in full in "C4 in detail" below,
  which is where the assessment lives.
- **C3 rate.** Noted, not blocking. **The floor is 15 fps**, which is the point at which the effect
  is still live rather than a slideshow, and not the 30 fps the camera happens to supply.
  That is a much larger budget than the wireframe's 2.52 ms suggests: 15 fps is 66.7 ms per frame,
  the tracker takes 11.55 ms and the window sink 5.4 ms, so a renderer has on the order of **40 ms**
  even counting a full camera interval against it, and closer to 48 ms if it does not
  (at 15 fps the 30 fps camera always has a frame buffered, so the read stops costing 8 ms).
  Roughly sixteen times the current render stage. A candidate is only excluded by C3 if it is
  offline, which in this document means Blender alone.
- **C2 headless.** Noted, not blocking. **g4 is not mandatory** for this phase. The seam holding
  several renderers is itself the answer: a headless-capable one can stay as the testable path
  while a g7-only one is developed beside it. Still worth recording per candidate, because a
  renderer that runs on g4 keeps the file-output regression tests meaningful, and one that cannot
  makes this phase the repo's first stated exception to "write for the weaker machine".
- **C5 packaging.** FOSS, and installable under **Python 3.14** through uv.
  Partly relaxed: **new libraries, system packages and drivers may be installed on g7**, so a
  candidate needing a Vulkan loader, an EGL library or a newer driver is not thereby excluded.
  What that does not relax is PyPI: installing system packages does not produce a 3.14 wheel, and
  3.14 is recent enough that wheel availability is a real filter. It is **unverified for every entry
  below**; treat the packaging column as a claim to check, not a finding.

Hardware, verified on g7 today rather than quoted from the plan:
Quadro RTX 3000 Max-Q, 6144 MiB, driver 580.173.02, OpenGL 4.6 core, direct rendering on `:1`.
`libEGL_nvidia.so.580.173.02` is present alongside `libEGL_mesa.so.0`, so a GPU EGL context without
X11 is plausible on g7. g4 has no such device and would fall back to Mesa `llvmpipe` software rendering.

## OpenGL and GPU bindings

The layer that maps onto the seam most directly, because the seam was designed around it.

### ModernGL

<https://github.com/moderngl/moderngl>, <https://moderngl.readthedocs.io>

Thin wrapper over OpenGL 3.3+ core profile. Buffers, shaders, VAOs, framebuffers as Python objects,
no scene graph, no window. Windowing is left to a separate package (`moderngl-window`) or to nothing at all.

- Pro: no camera abstraction to fight. A projection matrix is a uniform, so C1 is free.
- Pro: standalone contexts are a first-class feature, and EGL is a documented backend:
  `moderngl.create_context(standalone=True, backend="egl")`. That is the C2 answer,
  with `LIBGL_ALWAYS_SOFTWARE=1` and Mesa as the fallback where there is no GPU.
- Pro: framebuffer objects render at any size independently of any window, which suits a
  seam that is handed a size.
- Pro: smallest new concept count. The repo already owns the projection maths; this adds a context,
  a shader and a buffer.
- Con: GLSL by hand. Loading a model, lighting it and depth sorting are all code to write,
  not features to call.
- Con: OpenGL is in maintenance across the industry; new work upstream is on Vulkan and WebGPU.
  Not a problem on a 4.6 driver, but it is the direction of travel.
- Note: the ModernGL headless doc's Ubuntu recipe starts an `Xvfb` display. That is for the X11
  backend; the EGL path is what avoids needing one, and the doc is not clear about the distinction.
  Worth verifying on g4 rather than believing either reading.

### PyOpenGL

<https://pyopengl.sourceforge.net/>

Raw ctypes bindings to the C API. The oldest option, and what most other libraries sit on.

- Pro: complete API surface, including EGL and every extension.
- Pro: no abstraction can get in the way of C1.
- Con: C-shaped API in Python. Every call is a ctypes crossing, which is both verbose and slow
  per call; a renderer written against it is drawing calls in a loop.
- Con: maintenance has been intermittent. Usually reached through something else rather than directly.

### wgpu-py

<https://github.com/pygfx/wgpu-py>

Python bindings to `wgpu-native`, the Rust implementation of WebGPU, which targets Vulkan, Metal and D3D12.
The modern replacement for the PyOpenGL layer.

- Pro: current API rather than a maintained legacy one, and the same shader language (WGSL) that
  the browser uses, which is the one point of contact with `../03_phone_webapp/`.
- Pro: offscreen rendering to a texture is the normal way to use it; no display server involved.
- Pro: explicit API, so a projection matrix is a uniform buffer. C1 free.
- Con: explicit in the Vulkan sense. Bind group layouts, pipelines and command encoders before
  a triangle appears; considerably more setup code than ModernGL for the same result.
- Con: NVIDIA driver 580 supports Vulkan, but this is another driver path to be verified on both
  machines rather than assumed.

### VisPy

<https://vispy.org/>

High-performance 2D/3D visualisation on OpenGL, aimed at scientific plotting of large datasets.
Sits between raw GL and a plotting library: a `gloo` layer that is close to the hardware,
and scene graph layers above it.

- Pro: the `gloo` layer is a reasonable ModernGL alternative, with more batteries.
- Pro: multiple backends, and offscreen rendering is supported.
- Con: the value it adds over ModernGL is visualisation primitives (markers, lines, volumes,
  colormaps) that a head-coupled scene has little use for.
- Con: the scene graph's cameras are the usual fov/aspect kind, so C1 lives in `gloo`, below them.

### pyglet

<https://pyglet.org/>

Pure Python windowing, input, audio and OpenGL context creation. Frequently paired with ModernGL
rather than competing with it.

- Pro: no compiled dependencies, and it is the window layer several of these assume.
- Pro: owns the event loop, which is the thing `cv.waitKey` is currently doing badly.
- Con: not a renderer by itself for this purpose; its own graphics API is 2D-leaning.
- Con: requires a display, so it is a g7-only component whichever renderer it wraps.

## Scene graphs and engines

Higher level: they own the camera, the scene tree and the render loop.
The recurring risk in this whole category is C1, since a camera abstraction is exactly what an
off-axis frustum has to bypass.

### Panda3D

<https://www.panda3d.org/>

Mature open-source engine (BSD), C++ core with Python as the primary language.
Used for real games, not only demos.

- Pro: **`MatrixLens` takes an explicit 4x4 projection matrix**, which is precisely C1 and
  unusually rare in this category. There is also an `FC_off_axis` frustum flag, and the stereo
  support already builds asymmetric frusta for the two eyes.
- Pro: complete engine. Model loading, materials, shaders, scene graph, culling, animation.
- Pro: offscreen buffers are supported for render-to-texture.
- Con: large dependency and a large framework. It wants to own the application, and abyss already
  has a loop with capture and tracking in it.
- Con: getting a frame back as an ndarray each tick goes through its texture readback path,
  which is a supported but not a fast path.
- Con: the coordinate convention is Z-up with Y forward, which is not the convention the repo's
  screen and camera frames use. A conversion, and a place for a sign error.

### Ursina

<https://www.ursinaengine.org/>

A friendly API layer over Panda3D.

- Pro: the shortest path from nothing to a lit spinning cube.
- Con: the layer's purpose is hiding exactly what abyss needs to reach. Anything non-standard means
  going through it to Panda3D, so it adds a dependency without removing the one under it.
- Con: not a serious candidate for this seam. Listed so it is not raised again.

### pygfx

<https://pygfx.org/>, <https://github.com/pygfx/pygfx>

A render engine on wgpu-py, API shaped after three.js, with a lean towards scientific visualisation.
Same team as wgpu-py.

- Pro: a genuine scene graph and material system without an engine's application framework.
- Pro: WebGPU underneath, so the concept model transfers to a browser implementation of the
  same scene if `../03_phone_webapp/` ever happens.
- Pro: offscreen rendering to a texture, which suits both the seam and headless work.
- Con: C1 is the open question. It has camera objects; whether an arbitrary projection matrix can be
  set on one, or whether `NDCCamera` plus a manual transform is the workaround, needs checking in
  the source before this can be shortlisted.
- Con: younger and smaller than Panda3D. The scientific focus shows in what is polished.

### raylib, via raylib-python-cffi

<https://www.raylib.com/>, <https://github.com/electronstudio/raylib-python-cffi>

C game library with a deliberately small API, wrapped with CFFI as the `raylib` and `pyray` modules.

- Pro: small and readable API, immediate-mode style, quick to get 3D on screen.
- Pro: `SetMatrixProjection` exists in `rlgl`, so C1 is reachable, though it means dropping below
  the `Camera3D` type the rest of the API is built around.
- Con: assumes it owns the window and the frame loop. Headless is not a supported mode.
- Con: readback through `LoadImageFromScreen` is a screenshot path, not a pipeline.

### Open3D

<https://www.open3d.org/>

Point cloud and mesh processing library with a visualiser attached.

- Pro: strong at the data side: meshes, point clouds, registration, and an offscreen renderer
  (`OffscreenRenderer`) that returns images.
- Con: the camera is set through intrinsics and extrinsics. That is closer to C1 than a fov call
  is, since an intrinsic matrix does carry a principal point offset, which is the axis shift.
  Whether it covers the general asymmetric frustum is unverified.
- Con: heavy install, and rendering is not the part of it that gets the attention.

### PyVista and VTK

<https://pyvista.org/>

Pythonic wrapper over VTK, the scientific visualisation toolkit.

- Pro: VTK's camera does expose window centre and a user-supplied transform, so off-axis is expressible.
  VTK has been used for CAVE-style displays.
- Pro: offscreen rendering is well trodden, including on headless servers with OSMesa builds.
- Con: VTK's abstractions are aimed at scientific datasets; a scene of models is possible but
  against the grain.
- Con: very large dependency for what would be used.

## Offscreen mesh renderers

Narrower: load a mesh, render it to an image. No window, no loop.

### pyrender

<https://github.com/mmatl/pyrender>, <https://pyrender.readthedocs.io>

Physically based renderer for glTF 2.0 scenes, built on trimesh, with both a viewer and an
`OffscreenRenderer`.

- Pro: **offscreen is the primary use case**, with three backends selected by `PYOPENGL_PLATFORM`:
  pyglet, `osmesa` (software) and `egl` (GPU without a display manager). This is exactly the
  g7/g4 split, chosen by an environment variable.
- Pro: returns colour and depth as numpy arrays, which is the seam's return type without adaptation.
- Pro: sensible lighting and materials out of the box, so a scene looks like something without
  writing GLSL.
- Con: **C1 is the problem.** The camera classes are `PerspectiveCamera` (yfov, aspect) and
  `IntrinsicsCamera` (fx, fy, cx, cy). `IntrinsicsCamera` can express an off-axis frustum through
  the principal point, so the projection is reachable, but it means converting the frustum to
  intrinsics rather than passing the matrix. That conversion is a place for a bug, and the repo
  has tests for the matrix, not for its intrinsics equivalent.
- Con: maintenance has been thin; the last release is old and the EGL device selection issue
  (multi-GPU) is a known long-standing one.
- Con: per-frame cost is unmeasured here and it is not written for a 30 fps interactive loop.

### trimesh

<https://trimesh.org/>

Mesh loading and geometry processing. Rendering is delegated (to pyglet for a viewer, or to pyrender).

- Pro: the right tool for **loading and preparing** models whatever renders them. Likely to be used
  regardless of the C1 decision.
- Con: not a renderer. Listed as a component, not a candidate.

### simple-3dviz

<https://simple-3dviz.com/>

Small mesh and point cloud viewer on moderngl, with offscreen rendering built in.

- Pro: much lighter than Open3D or PyVista for "show me these meshes".
- Con: small project, thin camera model. Same C1 question as the others in this section,
  with less code behind it if the answer is no.

## Radiance fields

Rendering a captured real place instead of authored geometry. A different project, and it is worth
being explicit that this is a content decision that drags a pipeline behind it, not a renderer swap.

### 3D Gaussian Splatting, reference implementation

<https://github.com/graphdeco-inria/gaussian-splatting> (INRIA)

The 2023 paper's code. Explicit scene representation (anisotropic 3D Gaussians) rasterised
differentiably, rather than a neural field ray-marched.

- Pro: real-time rendering is the paper's headline claim, and rasterisation is what makes it so.
- Pro: a rasteriser takes a view and a projection, so C1 is structurally fine.
- Con: research code with a non-commercial licence on the original release. **Licence must be
  checked before use**, unlike everything else in this document.
- Con: CUDA extension compiled against a specific toolkit and torch version. Under Python 3.14
  this is the least likely thing in this file to install cleanly.

### gsplat

<https://github.com/nerfstudio-project/gsplat>

The Nerfstudio team's CUDA rasterisation backend, pip-installable, Apache 2.0.

- Pro: permissive licence, packaged as a library, actively developed. The reference
  implementation's problems in one package.
- Pro: reported to use materially less VRAM than the original, which matters at 6 GB.
- Con: still a CUDA extension with a torch dependency. C2 (g4) is out entirely, which the relaxed
  weighting now permits, but C5 is a real risk on 3.14 and is the harder problem.
- Con: a rasteriser, not a scene: it needs a trained model, which needs captures and a training run.

### Nerfstudio

<https://docs.nerf.studio/>

Framework around the whole pipeline: capture, COLMAP poses, training (`splatfacto` for splats),
and a web viewer.

- Pro: the practical route to producing a splat at all, as opposed to rendering one.
- Con: large environment, and it is a research framework rather than a dependency.
- Con: training is a GPU job. 6 GB is workable for small scenes and is the ceiling.

### LichtFeld Studio

<https://lichtfeld.io/>

C++23 and CUDA application for training, editing and rendering splats locally, with Python plugins.

- Pro: an application rather than a research repo; editing and rendering are treated as features.
- Con: external application, so it is a content tool that would produce an asset for abyss to render,
  not a renderer abyss calls.

### taichi_3d_gaussian_splatting

<https://github.com/wanmeihuali/taichi_3d_gaussian_splatting>

Unofficial reimplementation in Taichi, which JIT compiles Python to GPU kernels.

- Pro: no CUDA toolchain to match; Taichi handles the backend and also has a CPU one.
- Pro: readable, since the kernels are Python.
- Con: unofficial and slower than the CUDA original.

### NeRF, and instant-ngp

<https://github.com/NVlabs/instant-ngp>

The predecessor family: an implicit neural field ray-marched per pixel.

- Con: **superseded for this use.** Splatting rasterises where NeRF marches rays, which is the
  whole speed difference, and the geometry it produces is at least comparable.
  Listed so the option is recorded as considered and dropped rather than missed.
- Note: NeRF stores far less than a splat scene does, which is the one axis where it still wins.
  Irrelevant when the scene sits on the same disk as the renderer.

## External FOSS applications

Not libraries. Either a content tool feeding abyss, or a different host for the whole thing.

### Blender

<https://www.blender.org/>

GPL. Modelling, and two renderers (EEVEE rasteriser, Cycles path tracer), fully scriptable through `bpy`.

- Pro: the obvious **asset source**, whatever renders at runtime. glTF export into trimesh or pyrender
  is a standard path.
- Pro: `bpy` is on PyPI, so a headless render can be driven from Python.
- Pro: the camera has `shift_x` and `shift_y`, which is film-back offset, which is an off-axis
  frustum. The technique has been applied in Blender before.
- Con: **not a real-time path**, and the only candidate here that C3 excludes even at the relaxed
  15 fps floor. EEVEE headless has historically needed a display or `xvfb-run`, and per-frame cost
  is far above 66 ms for anything worth looking at. This is a content tool, not a runtime.
- Con: `bpy` pins its own Python version, which is likely to conflict with 3.14 outright.

### Godot

<https://godotengine.org/>

MIT game engine, GDScript or C#, with GDExtension for native code.

- Pro: a complete real-time engine with a permissive licence and a small runtime.
- Pro: `Camera3D` has `set_frustum_offset`, and the projection can be overridden, so C1 is reachable.
- Con: the language boundary. Driving it from the existing Python loop means either an IPC channel
  or rewriting capture and tracking in the engine, and MediaPipe lives in Python.
- Con: inverts the architecture. The engine owns the loop; abyss becomes a pose source feeding it.
  Defensible, but it is a different project shape and should be recognised as one.

### Xvfb, Mesa, EGL

<https://docs.mesa3d.org/>

Infrastructure rather than a candidate: the pieces that decide how C2 is answered.
`llvmpipe` gives software GL where there is no device, `Xvfb` gives a display where a library insists
on one, EGL avoids needing either. C2 is no longer blocking, so this stops being a gate and becomes
the choice of *which* renderer stays testable on g4 while another is developed on g7.

## C4 in detail: the seam

The constraint the notes weight above the others, so it gets the space.

### What C4 actually asks

Not "can it draw a scene". The seam exists so that `run_loop` can hold a renderer without knowing
which one it is, and so that **more than one can exist at a time**: a wireframe for tests, a GL scene
on g7, a splat renderer later. That decomposes into four questions, and they are not the same question.

**C4a, target ownership.** Can it render to an offscreen target of a size the *caller* chooses,
with no window in existence? The seam is handed a size, because that size comes from `ScreenConfig`,
which describes a physical panel the frustum is built from. A library whose render target is its own
window has taken that decision away: the output size becomes the library's, and every measurement in
the screen config then describes a rectangle that is not what got drawn. This is the same failure
`WindowSink` documents for a floating window, one layer down.

**C4b, loop ownership.** Is it called once per frame, or does it call you? `run_loop` already owns
pacing, capture, tracking, the controls and the stats. An engine with an `app.run()` at the centre
inverts that: capture and tracking become callbacks inside someone else's frame. That is not fatal
and it is arguably the better architecture for a game, but it is a rewrite of `loop.py` rather than
a renderer swap, and it cannot coexist with a second renderer that expects to be called.
This is the criterion that separates the libraries from the engines.

**C4c, pixel handover.** OpenGL framebuffers come back bottom-up and RGB or RGBA; the sinks want
top-down, BGR, contiguous `uint8`. So every GL candidate owes a vertical flip and a channel swap per
frame. The cost is small and can be made zero (swizzle in the fragment shader, or negate the Y of
the projection), but it is per-candidate work with a specific hazard: **a vertically mirrored render
is the failure a head-coupled scene hides best.** The image still moves with the head, still has
parallax, still looks alive; it is just inverted about the horizontal. Whatever is chosen needs a
fixed-matrix, known-image regression test that a mirror would fail, because watching it will not
catch this.

**C4d, context lifetime and coexistence.** An OpenGL context is thread-affine and current-per-thread:
created once, made current on the thread that renders, and two GL renderers in one process means
managing whose context is current. Today's loop is single-threaded so this is free, but it is a
standing constraint on ever moving capture into its own thread. WebGPU has no current-context
concept at all, which is a real point in wgpu-py's favour and the kind of thing that only shows up
under a coexistence requirement. Some candidates are worse than awkward here and are outright
**single-instance**: `pyrender` selects its backend through the `PYOPENGL_PLATFORM` environment
variable read at import, so the choice is process-global; `bpy` is a singleton module wrapping a
whole application; Godot is a separate process.

### The readback question

Sitting under C4a, and the largest single thing the survey turned up.

`Renderer.render` returns an ndarray. For a GPU renderer that means `glReadPixels` (or the
equivalent) of a 1920x1080 framebuffer every frame, roughly 6 MB across PCIe, after which
`WindowSink` hands it to `cv.imshow`, which uploads it to the GPU again to draw it. The pixels make
a round trip for no reason.

Three ways out, in increasing order of disruption:

1. **Accept it.** Readback is on the order of a millisecond or two at this size, plus a pipeline
   stall. Against the 40 ms that C3's 15 fps floor actually allows, this is noise. It also keeps
   the property that makes the phase testable: the same renderer still writes PNGs and video, on
   either machine, through the sinks that already exist.
2. **A second, optional method.** The GL renderer keeps `render` for the headless and file paths,
   and adds a `present` that draws to its own window and swaps buffers, used when a window sink is
   in play. Two paths to keep correct, and the loop has to know which it has, which is a small hole
   in the abstraction the seam exists to provide.
3. **Move the seam.** The renderer owns the window, and the sink concept collapses into it for the
   live case. Cleanest at runtime, and the worst fit for a seam meant to hold several renderers at
   once: it is the arrangement that cannot be verified without a display, and it makes the window a
   renderer's private property rather than a swappable output.

Option 1 unless measurement says otherwise, and the measurement is cheap to make once any GL context
exists. The relaxed C3 makes it stronger than it was: the round trip was only worth arguing about
against a 16.85 ms budget.

### How the candidates rank on C4

- **Strong.** ModernGL, wgpu-py, pygfx, VisPy through `gloo`. All four are libraries that render to
  a framebuffer or texture the caller sized, are called rather than calling, and construct a context
  as an object rather than as a process mode. gsplat belongs here too on structure alone: it is a
  function taking camera parameters and returning a tensor, which is the seam's shape exactly, and
  its problems are C5 and content rather than C4.
- **Workable, with a caveat.** Panda3D and Open3D and PyVista/VTK all support offscreen buffers and
  can be driven a frame at a time, but each was written expecting to own the loop, and the
  frame-at-a-time path is the less travelled one. pyrender fits C4a and C4b well and fails C4d:
  its backend is a process-global environment variable, so it cannot sit beside a second GL renderer
  that wants a different one.
- **Poor.** raylib and Ursina both assume the window and the frame loop are theirs.
- **Not in-process at all.** Blender and Godot. Whatever their merits, they are a separate program,
  so "swappable renderer" would mean an IPC protocol, which is a different project.

## Summary

Ordered by the weighting in "What is being chosen for": **C4 first, C1 second**, C3 and C2 noted,
C5 last. `?` is unverified.

| Candidate | C4 seam | C1 matrix | C3 rate | C2 headless | C5 on 3.14 | Shape |
| --------- | ------- | --------- | ------- | ----------- | ---------- | ----- |
| ModernGL | strong | yes | yes | yes, EGL | ? | library, no window |
| wgpu-py | strong, no current context | yes | yes | yes | ? | library, modern API |
| pygfx | strong | ? | yes | yes | ? | scene graph |
| VisPy | strong via `gloo` | via `gloo` | yes | yes | ? | visualisation |
| PyOpenGL | strong | yes | yes | yes | ? | raw bindings |
| gsplat | strong | yes | yes | no, CUDA | risk | splat rasteriser |
| Panda3D | workable, wants the loop | yes, `MatrixLens` | yes | buffers | ? | engine |
| Open3D | workable | ? intrinsics | ? | yes | ? | data + viewer |
| PyVista / VTK | workable | yes | ? | yes, OSMesa | ? | scientific |
| simple-3dviz | workable | ? | ? | yes | ? | thin viewer |
| pyrender | fails C4d, global backend | via intrinsics | ? | yes, EGL/OSMesa | ? | offscreen only |
| raylib-python | poor, owns the window | via `rlgl` | yes | no | ? | game library |
| Ursina | poor, owns the loop | through Panda3D | yes | no | ? | wrapper |
| Nerfstudio | n/a, a pipeline | n/a | n/a | no | risk | training pipeline |
| Blender / `bpy` | not in-process | yes, `shift_x` | no | with xvfb | conflict | content tool |
| Godot | not in-process | yes | yes | no | n/a | different host |

Four things the table makes visible once C4 leads it:

- **C1 is less selective than expected.** Off-axis projection is what stereo needs, so most 3D
  libraries have a route to it. It filters almost nothing on its own.
- **C4 is what actually sorts the field**, and it sorts it by *shape* rather than by capability:
  libraries that are called cluster at the top, frameworks that call you cluster below, separate
  programs fall out entirely. That ordering is stable regardless of which scene gets drawn, which is
  what makes C4 the right thing to weight.
- **Relaxing C2 changed less than expected.** Every strong-C4 candidate except gsplat is headless-capable
  anyway, because rendering to a caller-sized offscreen target and rendering without a display are
  nearly the same property. The relaxation buys the splat branch and little else.
- **Relaxing C3 to 15 fps excludes nothing that C4 had not already excluded**, Blender aside.
  Its main effect is on the readback question, which stops being a trade-off at a 40 ms budget.

## Questions this raises

To be folded into [`00_start.md`](00_start.md) as numbered questions rather than answered here.

- Authored geometry or captured place. This is prior to the renderer choice and the survey does not
  settle it; the radiance field section only exists because Q1 left it open.
- Whether the seam should be proven with **two** renderers before one is chosen. The C4 weighting
  implies it: the wireframe plus one GL renderer running behind the same protocol is the only thing
  that actually demonstrates swappability, and it is a cheaper test of the seam than a good scene is.
- Which of the three readback options, and whether to measure first. The relaxed C3 points at
  option 1 without measuring.
- Whether the mirror hazard in C4c gets a fixed-matrix regression test as part of the phase's
  exit criteria. Recommended, because it is the one defect the live demo cannot reveal.
- Python 3.14 wheel availability for the shortlist. One `uv pip install --dry-run` per candidate
  answers the whole C5 column, and should happen before any of them is chosen.
