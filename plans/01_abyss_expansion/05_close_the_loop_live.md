---
status: planned
---

# Phase 5 - close the loop, live

## Overview

Camera to tracker to renderer to screen, on the machine holding both. The exit criterion of the
initiative, and the only phase that cannot be finished on a headless box.

Context: [`00_start.md`](00_start.md). Wires together phase 1's
[`01_viewer_position.md`](01_viewer_position.md) tracker, phase 2's
[`02_camera_screen_model.md`](02_camera_screen_model.md) config, phase 3's
[`03_off_axis_projection.md`](03_off_axis_projection.md) projection and phase 4's
[`04_minimal_scene.md`](04_minimal_scene.md) renderer. Nothing after it belongs to this initiative:
a real renderer is [`../02_scene_rendering/`](../02_scene_rendering/) and the phone is
[`../03_phone_webapp/`](../03_phone_webapp/).

**Almost no new maths.** Every piece exists; this phase makes them a loop. That is the reason to be
suspicious of it rather than relaxed: the work that remains is exactly the work the previous four
phases were able to defer, which is everything that only exists in real time. Latency, a scale
estimate with no future frames to draw on, a camera that can fail while claiming success, and a
window that has to be the panel rather than a rectangle on it.

## What is actually new

| Piece | Why it could not exist before |
| ----- | ----------------------------- |
| `WindowSink` | the named second sink, and the first one whose size it does not choose |
| a live head scale | `estimate_head_scale` consumes a whole clip; live there is no whole clip |
| capture health | a dead camera returns `ok=True` and black frames, which reads as "no face" |
| a latency budget | nothing so far had to finish inside a frame |
| the loop itself | the first code that owns capture, inference, projection and display together |

## g7 can now be checked against a tape measure

Worth stating before the plan, because it changes what "working" means for this phase.

Substituting the scale estimate into the position, the reported depth is

    depth = depth_m * scale
          = depth_m * ipd_real / median(ipd_px * depth_m / focal)
          -> focal * ipd_real / ipd_px

so **MediaPipe's own depth cancels out**, up to the difference between a per-frame value and the
median it was normalised by. The depth that reaches the frustum is the pinhole formula over the
measured focal length and the viewer's real interpupillary distance, nothing more.

Two consequences. First, the depth is only as good as those two numbers, and on g7 both are now real:
945 px at 720 tall, measured by ChArUco, and an interpupillary distance that is currently the 63 mm
population mean rather than this viewer's. Second, and this is what makes the phase checkable, a
prediction can be written down before running anything:

| apparent iris separation | predicted depth |
| ------------------------ | --------------- |
| 60 px | 0.99 m |
| 80 px | 0.74 m |
| 100 px | 0.60 m |
| 120 px | 0.50 m |

Sit at a tape-measured 0.50 m and the loop should say so. Every previous phase could only check
internal consistency, because the sample clips have no known camera. This one can be wrong against
the world, which is a much better test.

## The loop, and where it is allowed to be simple

    capture -> landmark -> eye sample -> scale -> smooth -> frustum -> render -> window

One thread, one loop, synchronous, until measurement says otherwise. The named upgrade is
MediaPipe's `LIVE_STREAM` running mode, which delivers results through a callback and decouples
inference from capture, and a capture thread that always holds the newest frame. Both are real
answers to a latency problem that has not been measured yet, and picking them first would be
designing against a guess.

So **step one is measurement, not code**. Everything about the loop's shape follows from how long a
face landmarker call takes, and it is a morning's difference whether that is 15 ms or 90 ms.

## Step one: a benchmark that outlives this phase

`scripts/benchmark_landmarker.py`, and it is deliberately not part of the loop. It times the stages
against a **recorded clip**, so it needs no camera, no display and nobody sitting still, which is
what lets the same script run on any machine in the fleet and produce comparable numbers.

**The question it answers has already changed once, which is the argument for building it.** It was
going to be "is the GPU delegate faster". A twenty-line probe run before any of this settled that:
the delegate is disabled in the wheel's build flags, and CPU inference is 11.2 ms median at
1920x1080 regardless. So inference is not where the time goes, and the benchmark's job is now to
find out where it does.

| Axis | Values |
| ---- | ------ |
| stage | capture, landmark, eye position, projection, render, sink |
| frame size | 1280x720, and 640x480 as the cheap fallback |

Output is one row per configuration - median, p95 and the achievable frame rate - written to
`cache/benchmark/` and pasted into the log per machine. Per machine matters: g4 is an old integrated
GPU and g7 a Quadro, and while neither runs the inference on its GPU, the CPUs differ and the
interesting result is the shape of the gap rather than either figure alone.

The delegate axis is **dropped, not skipped**: `delegate=GPU` raises
`GPU processing is disabled in build flags` on this wheel, so a GPU row would only ever record the
same failure. Recorded here so the next person does not spend an afternoon installing drivers that
are already installed. Reopening it means building MediaPipe from source, and at an 89 fps ceiling
there is nothing to buy.

This also corrects a repo-wide claim rather than only informing this phase, so the fix goes with it:
`.github/copilot-instructions.md` states "CPU only. No Nvidia GPU here" and "Headless. No display",
which are g4's constraints presented as the environment's. On g7 there is a Quadro RTX 3000 with
6 GB, OpenGL 4.6 and an X11 session on `:1`. Both machines' constraints stay written down, attached
to the machine they belong to.

## Capture, which has bitten this repo before

Three findings from the calibration work in `tracking.md` apply directly, and all three are cheaper
to build in than to rediscover:

- **Pin the mode.** The Chicony webcam's YUYV default silently clamps to 640x480. MJPG 1280x720 is
  the mode the focal length was measured at, and `focal_px_for_height` would rescale to 480 without
  knowing the aspect ratio changed. Set the fourcc and the size, then verify what the camera
  actually gave back rather than assuming it obeyed.
- **Set `CAP_PROP_BUFFERSIZE` to 1.** The queue was measured four frames deep. Offline that produced
  four identical calibration views; live it is 160 ms of pure latency between the viewer moving and
  the camera admitting it.
- **A dead camera claims success.** When the session was locked, `read()` returned `ok=True` with
  black frames. Downstream that is indistinguishable from "no face", so the loop would sit there
  rendering a held position and looking merely boring rather than broken. The loop must check frame
  statistics, not the return flag, and say `CaptureIsBlackError` rather than shrugging.

## Fullscreen is a geometric requirement

`ScreenConfig` describes the whole panel: 344 by 193 mm, with the camera 100.5 mm above its centre.
The frustum is built from that rectangle, so the render only means anything if it covers exactly that
rectangle. A 1280x720 window floating on a 1920x1080 desktop is a different, smaller, differently
placed window onto the world, and every number in the config would then be describing something that
is not on screen.

So fullscreen is not a presentation choice, it is what makes the geometry true. The window opens
fullscreen or the phase has not been done. Modelling a windowed rectangle is possible - it is another
`ScreenConfig` with a smaller size and an offset - but it is a different thing to build and there is
no case for it.

Render at the panel's native 1920x1080 rather than at 720 and letting OpenCV upscale. The scene is
about forty lines; the cost is nothing and the result is crisp.

## What gets built

| Piece | Where | Role |
| ----- | ----- | ---- |
| `benchmark_landmarker.py` | `scripts/` | step one, and portable across the fleet |
| `WindowSink` | `src/abyss/sink.py` | fullscreen `cv.imshow`, reporting the panel's size as its own |
| `CaptureIsBlackError` and friends | `src/abyss/video/capture.py` | opening a camera in a known mode, and noticing when it dies |
| `LiveScale` | `src/abyss/viewer/eye_position.py` or beside it | the bootstrap-and-freeze scale estimator |
| the loop | `src/abyss/loop.py` | capture to sink, source and sink both injected |
| `scripts/live.py` | `scripts/` | the manual entry point, the only piece that needs a display |

`src/abyss/sink.py` becomes `src/abyss/sink/` here, as phase 4 said it would: a rename, not a
redesign.

**The loop takes its frame source and its sink as arguments**, which is what keeps this phase
testable at all. The same loop run over a clip with a `PngSink` is phase 4's output and needs no
camera and no display; run over device 0 with a `WindowSink` it is the live effect. If the loop can
only be exercised through a window, the phase has been built wrong.

## Tests

The live path cannot be tested automatically here, and pretending otherwise would be the anti-pattern
this repo keeps catching. What *can* be tested is everything that is not the window:

- the loop over a recorded clip with a `PngSink` produces the same frames as phase 4's track mode,
  which pins that the live wiring did not quietly change the offline behaviour
- `LiveScale` freezes after its bootstrap: the same samples in a different order give the same
  factor, and a later outlier does not move it
- `LiveScale` before bootstrap reports that it is not ready, rather than returning 1.0 and silently
  rendering at the wrong scale
- a black frame raises `CaptureIsBlackError`, and a frame with content does not
- a capture that comes back at 640x480 when 1280x720 was asked for raises, rather than rescaling the
  focal length to a resolution it was never measured at
- `WindowSink` satisfies the `Sink` protocol, checked without opening a window

And one manual check, written down because it is the phase's real exit criterion: sit at a
tape-measured distance and compare the reported depth against the table above.

## Out of scope

- A real renderer. The wireframe room is what gets displayed, unchanged from phase 4.
- The phone, the webapp, and serving frames anywhere. `../03_phone_webapp/`.
- Multiple viewers, or choosing between them. There is one person and one `ViewerConfig`.
- Stereo. Ruled out in `00_start.md` and still ruled out.
- Recording the live session to disk. `VideoSink` already exists and could be attached, but a run
  that both displays and records is a feature, not part of closing the loop.

## Open questions

- Q23: **How is the head scale estimated with no future frames?** `estimate_head_scale` takes the
  whole clip and returns one constant, which live is not available.
  a. Bootstrap and freeze: collect front-facing samples until N of them, take the median, never
     change it. A key re-runs the bootstrap.
  b. Rolling median over the last N front-facing samples, updated every frame.
  c. Skip it, leave the scale at 1.0 and accept the per-person error, which was measured at 13%
     between two subjects.
  Recommended: a, because b makes the scale a slowly moving target and the whole scene breathes
  when it moves, which is worse than being consistently a few percent off. Freezing also matches
  what the offline path does, so the two agree.
  ANS: **a, bootstrap and freeze.**
- Q24: **`VIDEO` or `LIVE_STREAM` running mode for the landmarker?** `VIDEO` is synchronous and is
  what phase 1 already uses, so the offline and live paths stay identical. `LIVE_STREAM` delivers
  results through a callback and lets capture run ahead of inference, at the cost of the loop no
  longer being a loop.
  Recommended: start on `VIDEO`, measure, and move only if the measurement says so. Named upgrade,
  not a starting point.
  ANS: **`VIDEO`, and the GPU question is closed by measurement rather than deferred.** Measured on
  g7 on 2026-08-17, before writing any of the loop:
  - **The GPU delegate cannot be used, and no install fixes it.** `delegate=GPU` fails with
    `ImageCloneCalculator: GPU processing is disabled in build flags`. The pip wheel is compiled
    without GPU support, so the Quadro, the driver, EGL and GLESv2 being present is irrelevant. The
    only route is building MediaPipe from source with GPU flags.
  - **It would not be worth it anyway.** CPU inference over `face01.mp4` at 1920x1080 runs at a
    median of **11.2 ms**, p95 11.7 ms, a face found in all 60 frames. That is an 89 fps ceiling
    from inference alone, at a larger frame than the loop will use.
  So inference is not the bottleneck, `VIDEO` stays, `LIVE_STREAM` is not needed, and the latency
  budget is spent somewhere else - capture queue depth and display, which is where the loop should
  look.
- Q25: **What happens to the smoother when frames are not evenly spaced?** `PositionSmoother` uses a
  left-triangle filter over the last five samples and assumes even spacing, which a clip guarantees
  and a live loop does not. Five taps at 25 fps is also 0.2 s of lag, which is visible in a
  head-coupled display in a way it never was in a plot.
  a. Leave it and retune the tap count once the real frame rate is known.
  b. Make it time-aware, weighting by elapsed time rather than by sample count.
  Recommended: a, because the tap count is one number and the real frame rate is not known yet.
  ANS: **a, retune the tap count once the rate is measured.**
- Q26: **Does the loop own the timing, or does the sink?** A display sink has a natural pace and a
  PNG sink does not, so "run at 30 fps" is a property of neither the renderer nor the tracker.
  Recommended: the loop, running as fast as the source allows and reporting what it achieved. Frame
  pacing is a problem only if the loop turns out to be faster than the display, which would be a
  good problem.
  ANS: **The loop owns it**, running as fast as the source allows and reporting what it achieved.
- Q27: **Is the viewer's own interpupillary distance worth measuring?** The depth is now
  `focal * ipd / ipd_px`, so the 63 mm population mean maps directly into a depth error: a viewer at
  60 mm would be read 5% too far away. Measuring it is a manual step, and there is a viewer registry
  waiting for the entry.
  Recommended: yes, once the loop runs, since it is the one remaining unmeasured number in the whole
  chain and the tape-measure check will show it as a constant offset.
  ANS: **Yes, measure it**, once the loop runs and the tape-measure check can show it as the
  constant offset it would be.

- Q28: **Is there a packaging fix for the GPU delegate, such as `mediapipe[cuda]`?** Raised because
  the situation looks absurd from outside: a 6 GB Quadro, driver 580, EGL, GLESv2 and OpenCL all
  present, and the library declines to use any of it. If a wheel variant exists, it is a one-line
  change to `pyproject.toml`.
  What is established, and how:
  - **There is no such extra.** `mediapipe` 1.0.0 declares no extras at all, checked in the installed
    metadata, and PyPI reports the same for the current 1.0.1. `mediapipe[cuda]` would install plain
    mediapipe and emit a "does not provide the extra" warning, which is the worst outcome of the
    three: it looks like it worked.
  - **CUDA is the wrong axis.** MediaPipe's GPU inference is the TFLite GPU delegate over OpenGL ES
    compute shaders, reached through EGL on Linux, with Metal on Apple. There is no CUDA backend to
    enable, so the NVIDIA-ness of the card is not the missing piece. Confidence: high on the
    architecture, and it is consistent with the error naming build flags rather than a missing
    runtime.
  - **The failure is a build-time switch, not a missing dependency.** `GPU processing is disabled in
    build flags` is the graph refusing to instantiate a calculator that was compiled out. No package
    installed alongside it can add one back.
  **Corrected on 2026-08-17 after actually searching, having first concluded this needed a source
  build.** It does not, and the real story is duller and more useful:
  - **GPU in the Python Tasks API is an officially supported feature**, and the docs are specific
    about where: "GPU support is currently limited to Ubuntu platforms". g7 runs Ubuntu 22.04, so
    this is the supported combination, not an exotic one.
  - **What we hit is a packaging regression, not a design.** The delegate worked on the Linux wheel
    up to **0.10.31** and broke in **0.10.32**, whose `manylinux_2_28_x86_64` wheel was built without
    the GPU flags, giving exactly our error. Reported upstream as
    [issue #6231](https://github.com/google-ai-edge/mediapipe/issues/6231) on 2026-02-03. Our 1.0.0
    inherits it.
  Routes, cheapest first:
  a. **Leave it, and re-probe on the next mediapipe bump.** The check is the twenty-line script that
     found this, so the cost of noticing a fix is a minute.
  b. **Pin 0.10.31.** Not one line, despite appearances: `pose-tools` requires `mediapipe>=1.0`, so
     this is an upstream change and a tag bump before abyss can express it, and it trades a current
     library for a six-month-old one across three repos.
  c. **Build from source with bazel.** Last resort rather than the answer, which is how it was
     wrongly written here first.
  Recommended: **a**, and now for three independent reasons rather than one.
  - The camera hands over 30 frames a second, so the frame budget is 33 ms and inference already fits
    in a third of it. Nothing the viewer can see improves.
  - **The GPU may not even be faster.** Users report no noticeable CPU/GPU difference in recent
    versions where 0.10.20 had one
    ([issue #6216](https://github.com/google-ai-edge/mediapipe/issues/6216)), which is plausible: the
    CPU path runs XNNPACK, and at this model size the transfer overhead can eat the win. Chasing it
    would mean paying b's cost for an unmeasured gain.
  - It is upstream's bug and upstream's to fix, and the workaround costs more than the wait.
  Worth naming why this looked absurd: MediaPipe is marketed on running well on every device, and
  that claim is about mobile and web, where the GPU path is the normal one. Python on desktop Linux
  is the thinnest-served corner of it, narrow enough that a routine wheel build could drop GPU
  support and ship. Not a stale claim so much as one that was never about this target.

## Done when

- The loop runs live on g7, fullscreen, and the scene moves with the viewer's head the way a window
  would.
- The measured end-to-end rate and latency are written down in the log, whatever they turn out to be.
  A slow loop that is honestly measured closes this phase; an unmeasured fast one does not.
- The benchmark has been run on g7, and on g4 if it is reachable, with both delegates, and the
  numbers are in the log next to the machine they came from.
- The reported depth agrees with a tape measure to within a stated tolerance, and any constant offset
  is explained rather than tuned away.
- The same loop, given a clip and a `PngSink`, reproduces phase 4's offline output.
- A dead or blocked camera fails loudly, and holding a position through a missing face is visible on
  the frame.
- `make check` is green, and the suite still passes with no camera, no display and no model present.
- The phase 1 regression CSVs are untouched.
