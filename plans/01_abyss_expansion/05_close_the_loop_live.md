---
status: in progress
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

**Two size axes, not one, and no capture stage.** Both corrections came out of writing it:

| Axis | Values |
| ---- | ------ |
| tracker stage | decode, landmark, eye position - scale with the **capture** size |
| render stage | projection, render, sink - scale with the **output** size |
| capture size | 1280x720, and 640x480 as the cheap fallback |
| output size | 1920x1080, and 1280x720 |

The single frame-size axis does not survive contact. The two halves scale with different numbers -
1280x720 in and 1920x1080 out in the live loop - and worse, 640x480 is 4:3, so rendering into it
raises `AspectMismatchError` against a 16:9 panel. Splitting the axis is what makes both halves
measurable.

There is no capture stage here either. Timing an mp4 decode and calling it capture would be a proxy,
and a bad one: a V4L2 MJPG read costs queue latency and JPEG decode that a seek-free file read does
not. The decode is timed and named `decode`. Real capture timing arrives with `video/capture.py` in
step two, where it can use the real opener rather than a second copy of it.

Output is one row per configuration - median, p95 and the achievable frame rate - written to
`cache/benchmark/` and pasted into the log per machine. Per machine matters: g4 is an old integrated
GPU and g7 a Quadro, and while neither runs the inference on its GPU, the CPUs differ and the
interesting result is the shape of the gap rather than either figure alone.

The budget table **excludes the sink and prints the excluded figure beside it**. Only `PngSink`
exists to time, and encoding a PNG to disk is not what the window sink will do, so folding it in
would report a budget for a loop nobody is going to run. What is left is what both loops pay, so the
remaining headroom is what the window sink has to fit into.

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

## What step one measured

g7, `face01.mp4`, 120 timed frames past a 5 frame warm-up, medians in milliseconds. Two runs at 60
and 120 frames agree to about 1 ms on every stage.

| path | stage | 1280x720 | 640x480 | | 1920x1080 | 1280x720 |
| ---- | ----- | -------- | ------- | --- | --------- | -------- |
| tracker | decode | 1.60 | 1.60 | | | |
| tracker | landmark | **11.55** | **12.77** | | | |
| tracker | eye position | 0.08 | 0.09 | | | |
| render | projection | | | | 0.04 | 0.04 |
| render | render | | | | **9.12** | 4.26 |
| render | sink (`PngSink`) | | | | 14.41 | 6.88 |

The render row is what step two then fixed: it reads **2.52** and **1.31** ms now.

Budget, sink excluded, against the camera's 33.3 ms, before and after step two:

| capture | output | loop, step one | loop, step two | headroom for the sink |
| ------- | ------ | -------------- | -------------- | --------------------- |
| 1280x720 | 1920x1080 | 22.39 ms, 44.7 fps | **16.85 ms, 59.4 fps** | +16.49 ms |
| 1280x720 | 1280x720 | 17.52 ms, 57.1 fps | 15.63 ms, 64.0 fps | +17.70 ms |

Three findings, each of which contradicts something that was written down before it was measured.

**Shrinking the capture buys nothing, and 640x480 is not a cheap fallback.** MediaPipe resizes to a
fixed model input, so the capture size does not change what the network sees. Over six paired runs
the smaller frame was slower in five, median 12.6 ms against 11.9 - but the run-to-run spread,
11.5 to 12.8 ms at 720, is as wide as the gap, so **"640x480 is slower" is not a claim this data
supports**; "it is not faster" is. The first write-up said slower on the strength of two runs and a
third contradicted it, which is what prompted counting properly.
Either way the axis is settled, and on the stronger half: pin 1280x720, which is also the mode the
focal length was measured at. 640x480 is not an escape hatch, because there is nothing to escape to.

**The render stage is 9.1 ms at 1080p, and 8.2 ms of that is the background fill.** Decomposing it:
the projection is 0.02 ms, drawing 36 anti-aliased lines is 0.51 ms, and
`np.full((h, w, 3), (16, 16, 16), np.uint8)` is 8.2 ms. Passing a **3-tuple** rather than a scalar
takes numpy off its memset path and onto a broadcast assignment, and at 1920x1080 that is a factor
of **41**: the identical array from `np.full(..., 16)` costs 0.20 ms. `BACKGROUND_BGR` is grey, so
the scalar is exactly equivalent today; for a background that is not grey, assigning per channel
into `np.empty` costs 1.79 ms and is still 5x better. Raised as Q29 rather than fixed in passing.

**Inference is confirmed as not the bottleneck, and the whole loop fits.** 11.6 ms at 1280x720, in
line with the 11.2 ms the earlier probe measured at 1920x1080. With step two applied the loop costs
**16.9 ms rendering native, 59.4 fps**, leaving 16.5 ms for the window sink - the one stage still
unmeasured, because it does not exist yet. Inference is now 75% of the loop, and everything else
together is 5 ms.

## Step two: the background fill, 8 ms for nothing

Q29 answered **a**: its own numbered step, before the loop is written. It is phase 4's code and a
one-line defect, but it is a quarter of the live frame budget, and the budget is what the rest of
this phase is designed against - so it goes in before anything depends on the wrong number, and it
goes in as a step rather than folded silently into the loop, which would leave no record that it
existed.

Nothing about the *output* changes. This is a pure cost fix: for the grey background actually in
use, every fill considered below produces a byte-identical frame. Phase 4's regression fixture is
projected coordinates rather than pixels, so it is untouched by construction, and the nine existing
renderer tests keep pinning the behaviour.

### The fill to use, and the two that were rejected

| | cost at 1920x1080 | correct for any background |
| --- | --- | --- |
| `np.full(shape, (16, 16, 16))` - today | 8.24 ms | yes |
| `np.full(shape, 16)` - scalar | 0.20 ms | **no** |
| per-channel assign into `np.empty` | 1.79 ms | yes |

**Per-channel assign is what gets built**, single path, no branch:

    frame = np.empty((height_px, width_px, 3), dtype=np.uint8)
    for channel, value in enumerate(self.background):
        frame[:, :, channel] = value

The scalar is four times cheaper again and is exactly equivalent *while the background is grey*,
which is a precondition on a constructor argument that callers are free to change. A non-grey
background would then render silently in the wrong colour, which is a worse failure than 1.6 ms:
wrong output that looks like output. Taking the scalar means either dropping the `background`
parameter or asserting greyness, and neither is worth 1.6 ms of a 33 ms budget. The branch that
picks the scalar when all three channels agree is the obvious third way and is **not** built: it is
two code paths and a test matrix for a saving nothing needs yet. If the window sink turns out to eat
the headroom, that branch is the named upgrade and it is three lines.

**Reusing one buffer across frames is rejected outright**, not deferred. It would take the fill to
zero, and it is wrong here: `render` returns the frame and the caller keeps it. Phase 4's
`render_run` holds selected frames for the contact sheet while writing the same frame to two sinks,
so a reused buffer would rewrite frames that had already been handed over, and the contact sheet
would come out as nine copies of the last frame. A renderer that mutates what it already returned is
not a renderer this seam can have.

### Done when

- `WireframeRenderer.render` no longer fills with a tuple, and the frame for the default background
  is unchanged.
- A test pins that a **non-grey** background is rendered in that colour. This is the test that fails
  if someone later takes the scalar shortcut, which is exactly why it is worth writing for a
  parameter nothing currently varies.
- A test pins that two successive `render` calls return independent arrays, guarding the rejected
  buffer reuse against being reintroduced as an optimisation.
- The benchmark is re-run and the render stage at 1920x1080 has dropped from 9.1 ms to about 1.5 ms,
  with the new loop total in the log. The benchmark is the instrument here: **no timing assertion
  goes in the test suite**, since a wall-clock threshold on a shared machine is a flaky test that
  measures the load average.
- `make check` is green and phase 4's coordinate fixture is untouched.

**Done.** The render stage is **2.52 ms** at 1920x1080 and 1.31 at 1280x720, so the loop is 16.85 ms
and 59.4 fps rendering native. The 1.5 ms estimate above was optimistic and stayed in the plan
rather than being quietly adjusted: it counted the 1.79 ms fill and forgot the 0.51 ms of lines it
is added to, so 2.3 ms was the number to predict. Both new tests were checked by mutation. The
scalar shortcut fails exactly one test, the intended one. Buffer reuse fails three: the intended one
plus both parallax tests, which hold two frames at once and so were already covering it
incidentally - the explicit test earns its place by naming the reason rather than by being the only
thing that catches it.

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

**Done as step three**, in `src/abyss/video/capture.py` with 10 tests and no camera involved. Three
things it settled that the plan had left implicit.

**"Check frame statistics" needed a second condition to be safe.** The measured dead frames were
mean 10.7 and standard deviation 2, and rejecting on darkness alone would reject a viewer in a badly
lit room - the failure mode being a live loop that refuses to run in the evening. Dark alone is a dim
room, flat alone is a blank wall in good light, and a dead capture is both at once, so both are
required together. Two tests pin exactly that, and flipping the `and` to an `or` fails both.

**Reading every pixel would have cost 7.3 ms**, a fifth of the frame budget, to answer "is anything
there". The check samples every 8th pixel instead and costs 0.26 ms. Measured, not assumed, and it
is the same shape of mistake as the background fill Q29 removed - the second time in two days that
the obvious spelling of a trivial operation was a fifth of a frame.

**`ok=False` and a black frame get different errors.** Collapsing them would lose the distinction
the module exists to draw: one means the capture stopped, the other means it did not stop and that is
the whole problem.

`open_camera` is deliberately not unit tested. It is three `set` calls and a readback against real
hardware, and a mock of it would only assert that the lines were written in the order they were
written in. The checks are free functions over a frame precisely so that everything else can be.

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

Render at the panel's native 1920x1080 rather than at 720 and letting OpenCV upscale.

**"The cost is nothing" was wrong and the benchmark said so**: the render stage was 9.1 ms at
1920x1080, a quarter of the frame budget, of which 8.2 ms was one avoidable line. Step two removed
it, and native rendering now costs **2.5 ms** against 1.3 ms at 720. That is close enough to nothing
to make the claim true, but it was not true when it was written and it was not free to fix.

## What gets built

| Piece | Where | Role |
| ----- | ----- | ---- |
| `benchmark_landmarker.py` | `scripts/` | step one, and portable across the fleet |
| the background fill | `src/abyss/render/renderer.py` | step two, returning 8 ms of the frame budget |
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

## Steps four and five: the loop, and what only a person can do

Built: `LiveScale` in `viewer/eye_position.py`, the `sink/` package with `WindowSink`,
`src/abyss/loop.py` and `scripts/live.py`. 245 tests, and **none of them needs a camera, a display
or a model file**.

Three things came out of writing it.

**The loop takes its tracker as an argument too**, not only its source and sink. The plan named two
seams; a third was needed for the same reason one level down. A landmarker built inside the loop
would make every test need a model file, and the loop's job is orchestration - what happens in what
order, and what to do when a step declines to produce anything. `track_with_landmarker` builds the
real one, and a stub tracker in a test exercises every path the live run takes.

**There are three states per frame, not two.** The plan had a face and no face. There is also *not
yet calibrated*, before the head scale has bootstrapped, and it cannot be folded into either: with
no scale there is no correct depth to render at, and Q23 says explicitly that returning 1.0 and
carrying on is the failure to avoid. So the loop shows what it is waiting for and counts those
frames separately. The distinction between the other two is preserved on the frame itself: a held
position is marked, because live there is nobody reading a log.

**The plan's offline equivalence test cannot hold as written**, and this is a correction rather than
an omission. It asked that the loop over a clip produce "the same frames as phase 4's track mode".
It cannot: `LiveScale` bootstraps from the first 30 front-facing samples while `estimate_head_scale`
uses the whole clip, **by design**. Measured on `face01`, the two give **0.939 and 0.941**, 0.2%
apart - the entire cost of bootstrap-and-freeze, against the 16% per-identity spread it corrects.
What replaces the test is a `scale=` argument that starts `LiveScale` frozen at a known value, so an
offline run can be compared with a live one without the bootstrap being the difference between them,
plus the clip mode in the runbook's pre-flight.

## The manual half

`docs/guides/phase5_live_runbook.md` holds everything a machine cannot do for itself: pre-flight on
a clip, measuring the viewer's interpupillary distance, the live run, the tape measure check with
its prediction table, the seven known failure modes with the error each one prints, and the template
for the log entry.

It is a separate document rather than a section here because it outlives the phase. The plan records
why the loop has its shape; the runbook is what someone follows at the desk, including the next time
the camera moves or a different person sits down.

## Tests

The live path cannot be tested automatically here, and pretending otherwise would be the anti-pattern
this repo keeps catching. What *can* be tested is everything that is not the window:

- ~~the loop over a recorded clip with a `PngSink` produces the same frames as phase 4's track
  mode~~. **Struck: it cannot hold, and the reason is the design rather than a defect.** `LiveScale`
  bootstraps from the first 30 front-facing samples while `estimate_head_scale` uses the whole clip
  (Q23), so the two runs differ by exactly the scale difference - 0.2% on `face01`. Replaced by
  `LiveScale(scale=...)` for a controlled comparison, and by clip mode as the runbook's pre-flight.
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

- Q29: **Where does the background fill fix go?** `WireframeRenderer.render` builds its frame with
  `np.full(shape, self.background, dtype=np.uint8)` where `background` is a 3-tuple, costing 8.2 ms
  at 1920x1080 against 0.20 ms for the scalar form. It is phase 4's code and a real defect rather
  than a preference: it is a quarter of the live frame budget, spent on nothing.
  a. Its own numbered step in this phase, before the loop is written, since the loop's budget is
     what made it visible and the fix changes that budget.
  b. Fold it into phase 4 as a correction, since that is whose code it is.
  c. Leave it. 22.4 ms already fits inside 33 ms.
  Recommended: a, because the skill's rule is that a real defect gets its own step rather than being
  fixed in passing, and because the headroom it returns is what the window sink has to fit into.
  Note the general fix and the minimal one differ: a scalar `np.full` is exactly equivalent only
  while the background is grey, so either the constant's greyness becomes a stated precondition or
  the fill assigns per channel at 1.79 ms.
  ANS: **a, its own numbered step, planned above as step two.** The fill assigns per channel, on the
  argument that the scalar's 1.6 ms is not worth a correctness precondition on a constructor
  argument callers can change: a non-grey background would render silently in the wrong colour.
  The scalar-when-grey branch is named as the upgrade if the window sink turns out to need the
  headroom, and buffer reuse is rejected outright rather than deferred, since `render` returns the
  frame and phase 4's `render_run` holds it.

## Done when

- The loop runs live on g7, fullscreen, and the scene moves with the viewer's head the way a window
  would.
- The measured end-to-end rate and latency are written down in the log, whatever they turn out to be.
  A slow loop that is honestly measured closes this phase; an unmeasured fast one does not.
- The benchmark has been run on g7, and on g4 if it is reachable, and the numbers are in the log next
  to the machine they came from. CPU only: the delegate axis was dropped in Q24/Q28, since a GPU row
  can only ever record the same build-flags failure.
- The reported depth agrees with a tape measure to within a stated tolerance, and any constant offset
  is explained rather than tuned away.
- The same loop, given a clip and a `PngSink`, runs the whole chain with no camera and no display.
  **Not** "reproduces phase 4's offline output": that was struck as impossible by design, since
  `LiveScale` bootstraps from the first 30 front-facing samples while `estimate_head_scale` uses the
  whole clip (Q23). The two differ by 0.2% on `face01`, which is the cost of freezing.
- A dead or blocked camera fails loudly, and holding a position through a missing face is visible on
  the frame.
- `make check` is green, and the suite still passes with no camera, no display and no model present.
- The phase 1 regression CSVs are untouched.

## Where this stands

Verified:

- The loop runs live on g7, fullscreen, and the effect works. The tape measure agreed at 0.50, 0.70
  and 1.00 m, with the residual attributed to positioning rather than to the model.
- The clip path runs the whole chain with no camera and no display.
- `make check` green at 259 tests, none of which needs a camera, a display or a model.
- The phase 1 regression CSVs regenerate byte-identical against `~/abyss-baselines/g7-d7bd614/`,
  checked by sha256 after `eye_position.py` gained `LiveScale`.
- A dead camera and a held position both fail visibly, covered by tests rather than by a live
  demonstration - locking the screen mid-run has not been tried.

Outstanding, and none of it is code:

- **The final rate is unmeasured.** Every live figure in the log predates the buffer fix: 8.3, then
  14.8 with the exposure control cleared, then the fix the probe says is worth 2x. One run closes
  this, and it is the criterion that says an unmeasured fast loop does not count.
- **The tape measure numbers were never written down.** The check passed and the machine was too
  laggy to copy the terminal, so the record has "pretty close" where it should have three pairs and
  a stated tolerance. Worth one more pass now that the depth readout is on the frame.
- **The benchmark has not run on g4.** It was built to be portable for exactly this and g4 has not
  been touched since. Not blocking: the interesting comparison is the shape of the gap, and it can
  be taken whenever g4 is next in use.
