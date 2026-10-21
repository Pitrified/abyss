# The geometry, end to end

What abyss computes between a camera frame arriving and a pixel being drawn, and why each step is
the shape it is. This is the index for the two detailed pages:

- [`viewer_position.md`](viewer_position.md): from pixels to a metric eye position.
  The pinhole model, MediaPipe's assumed intrinsics, and the head scale calibration.
- [`off_axis_projection.md`](off_axis_projection.md): from an eye position to a projection matrix.
  The screen frame, the asymmetric frustum, and the two invariants that define it.

The measurement procedures are in [`../guides/manual_measurements.md`](../guides/manual_measurements.md),
and why the calibration methods work is in [`camera_calibration.md`](camera_calibration.md).
This page does not repeat either.

## A note on the mathematics

Formulae here are written as LaTeX in markdown. GitHub and the VS Code markdown preview render it;
a plain text editor shows the source. That is a limitation worth knowing about but not one worth
working around, because the notation is short enough to read either way.

What markdown genuinely cannot carry is the **figures**. Six or so diagrams would do more work than
any paragraph here: two coordinate frames facing each other, the similar-triangles construction of
the frustum extents, an off-axis frustum beside a toed-in one, the parallax gain. Those are deferred
to [`../../plans/04_docs_site/00_start.md`](../../plans/04_docs_site/00_start.md), which is the
spin-off for building docs that can hold them.

## The claim

A screen showing a fixed perspective is a picture. A screen showing a perspective rebuilt from where
the viewer's eye actually is behaves like a **window**: the scene behind it stays put in the room
while the viewer moves, and the panel frames a different part of it from every position.

This is head-coupled perspective, also called fish tank VR in the literature.
The whole of abyss is the machinery for one sentence: *find the eye, rebuild the projection, draw*.

## The chain

Six steps, each with the frame its output lives in.

| # | Step | Input | Output | Frame |
| - | ---- | ----- | ------ | ----- |
| 1 | Capture | camera | BGR frame, 1280x720 | image, pixels |
| 2 | Landmark | frame | 478 landmarks + a 4x4 head pose | image + MediaPipe model |
| 3 | Lift | landmarks, pose, intrinsics | eye position, metres | camera |
| 4 | Smooth | eye position | eye position | camera |
| 5 | Project | eye position, panel rectangle | 4x4 view projection matrix | screen to clip |
| 6 | Draw | matrix, scene | BGR frame, 1920x1080 | pixels |

Steps 1 to 4 are [`viewer_position.md`](viewer_position.md), steps 5 and 6 are
[`off_axis_projection.md`](off_axis_projection.md).
Step 2 is MediaPipe's, reached through `pose_tools`, and abyss does not reimplement it.

## Three frames, and where they disagree

The chain crosses three coordinate systems, and two pairs of them disagree about direction.
Nearly every sign error possible in this codebase is at one of these two boundaries.

**Image frame.** Pixels. Origin at the top left corner, $u$ right, $v$ **down**.
The convention every image library uses.

**Camera frame.** Metres. Origin at the optical centre,
$+X$ to the right of the image, $+Y$ **down** the image, $+Z$ away from the lens.
This is the OpenCV convention, and it is what `abyss.viewer.eye_position` produces.
Note that MediaPipe does not use it: its output is centimetres and y-up, converted once on the way
out of `extract_eye_sample` and never carried further.

**Screen frame.** Metres. Origin at the centre of the panel,
$+X$ the **viewer's** right, $+Y$ up, $+Z$ out of the panel towards the viewer.
The scene sits at $z \le 0$, behind the window.
Used inside `abyss.render.frustum` and, deliberately, nowhere else.

The two disagreements:

- **Camera to screen** flips $X$ and $Y$, keeps $Z$. Detailed in
  [`off_axis_projection.md`](off_axis_projection.md#the-camera-to-screen-conversion).
- **NDC to pixels** flips $Y$ back, because normalised device coordinates are y-up and images are
  y-down.

Counted across the whole chain, $Y$ is flipped twice and $X$ once.
An odd number of flips on a single axis is a mirror, and a mirrored head-coupled render is the one
defect that still looks alive: it moves with the head, it has parallax, and it is inside out.
That hazard is why both flips are pinned by tests rather than left to inspection.

## The four calibrations

Nothing in the chain works without physical numbers, and there are exactly four sources of them.
Each is measured once and stored as a literal in `src/abyss/params/abyss_devices.py`.

| # | Quantity | Method | Value on g7 | Quality |
| - | -------- | ------ | ----------- | ------- |
| 1 | Panel size | the display's own EDID | 344 x 193 mm | exact |
| 2 | Camera to panel centre | ruler | (0, 100.5, 0) mm | +/- a millimetre |
| 3 | Camera focal length | ChArUco, Zhang's method | 945 px at 720 high | rms 0.263 px |
| 4 | Viewer interpupillary distance | mirror and ruler | 60 mm | best effort |

They are not equally good and the difference matters when a result looks wrong.
1 is exact and can be ruled out first.
3 is a real calibration with a residual attached, and was reproduced to 0.5% by a second run.
2 and 4 are hand measurements: a ruler against a bezel and a tape held against a face.
Treat 4 in particular as a stated input rather than a measurement, and do not fit anything to it.

A fifth number, the **head scale**, is derived rather than measured: see
[`viewer_position.md`](viewer_position.md#the-head-scale-calibration).

## What was verified, and to what

The exit criterion for the live phase was a tape measure against the reported depth,
sitting still at three marked distances:

| set | reported | error |
| --- | -------- | ----- |
| 0.50 m | 0.536 m | +7.2% |
| 0.75 m | 0.745 m | -0.7% |
| 1.00 m | 0.973 m | -2.7% |

Mean absolute error 3.5%. Three hand-held points is enough to say the chain is calibrated and not
enough to characterise its error, so nothing further is fitted to them.
The loop ran at 29.6 fps against a 30 fps camera on g7.

The properties of the projection are checked by unit test rather than by eye, because the failures
that matter are the ones a demo hides. Those are the two theorems in
[`off_axis_projection.md`](off_axis_projection.md#two-invariants).
