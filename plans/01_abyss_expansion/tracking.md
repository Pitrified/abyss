# implementation tracking

Building what abyss is actually for: head-coupled perspective, where the screen behaves like a
window onto a scene rather than a flat picture. Analysis and open questions in
[`00_start.md`](00_start.md).

## Key decisions

- **pose-tools stays the general library.** Anything `climbing-wire` would also want goes upstream
  and comes back as a tag bump. A face landmarker wrapper is the likely first instance of this.
- **Verifiable headless, first.** This box has no display and no GPU. Every phase has to be
  checkable on a recorded clip with file output, or it cannot be developed here at all. Live camera
  and live screen are the last mile, not the first.
- **The projection maths is not the risk.** Off-axis frustums are well understood. Metric eye
  position from a webcam, and knowing where the screen sits relative to the camera, are.
- **Cheapest implementation, behind a named seam.** Q2-Q5 all resolved the same way: recorded clips
  not a live camera, nominal config not calibration, the simplest scene, frames to disk not a
  window. Each is swappable, and a seam only counts when the second implementation can be named.
- **Face landmarker, single eye.** Q1 confirms phase 0 in pose-tools; both landmarkers if they wire
  together easily, face alone if not. Q6 rules out stereo and accepts interpupillary distance as the
  scale reference, configured like every other device number.
- **Config travels with the data, not the process.** Not per-machine: the Pixel records frames that
  a different machine processes, so the host says nothing about the camera. Four objects are passed
  to the components instead - camera (intrinsics), stream (clip or live capture, since one camera
  feeds both), screen (geometry the frustum is built from) and sink (what happens to a finished
  frame). No env var, no hostname lookup. The screen/sink line is a stage boundary: geometry is an
  input to rendering, the sink acts after it.
- **Spun off, not phased.** A real renderer ([`../02_scene_rendering/`](../02_scene_rendering/)) and
  the phone webapp ([`../03_phone_webapp/`](../03_phone_webapp/)) are separate initiatives: own
  dependencies, own questions, nothing here waits on them. The face landmarker is a genuine
  cross-repo prerequisite and stays in the table.
- **Pydantic for config.** Models fix the shape, params supply the values, pydantic validates. abyss
  takes the dependency back although pose-tools just dropped it: pose-tools has no config surface,
  abyss does. Values stay plain Python literals in params (Q10); a loader arrives only when
  something outside the repo needs to write config.

## Phases

Q1-Q14 are answered and the scope is settled. No sub-plan files exist yet - the sketches below are
all there is until each phase is written up. Phase 0 shipped as pose-tools v0.4.0 and is pinned
here, so phase 1 is unblocked.

| #  | Phase                          | Plan | Status |
| -- | ------------------------------ | ---- | ------ |
| 0  | face landmarker in pose-tools  | tracked in pose-tools | done, shipped as v0.4.0 |
| 1  | viewer position from a clip    | [`01_viewer_position.md`](01_viewer_position.md) | planned |
| 2  | camera and screen model        | -    | planned |
| 3  | off-axis projection            | -    | planned |
| 4  | minimal scene through it       | -    | planned, real renderer spun off |
| 5  | close the loop, live           | -    | planned, runs on g7 not here |

Status values: draft / planned / in progress / done / superseded / discarded.

Sketch of each, to be replaced by real sub-plans as they are picked up:

- **0 - face landmarker in pose-tools.** `landmark/face.py` over `BaseLandmarkerFrame`, plus a
  `face_landmarker` entry in `ModelManager.MODEL_FILENAMES`, released as a tag and pinned here.
  Confirmed by Q1. Done: tracked in `pose-tools/scratch_space/04_face_landmarker/`, shipped as
  `v0.4.0` and pinned here. Face runs alone - the two landmarkers were never wired together, since
  face alone answers the question. It stays listed as the cross-repo prerequisite it was.
- **1 - viewer position from a clip.** Recorded video in, a per-frame 3D eye position out, smoothed
  with the `utils.np_signal` filters, not `SignalTracker` - see the sub-plan. Output is a plot and a
  CSV, both checkable headless. This is where the scale problem gets solved or explicitly deferred.
- **2 - camera and screen model.** Nominal published numbers, not a calibration step: focal length
  or FOV for a capture device, width and height in metres plus origin relative to the camera for a
  display device. Four pydantic models - camera, stream, screen, sink - constructed from literals in
  params and passed in rather than looked up.
- **3 - off-axis projection.** Eye position plus screen rectangle to a projection matrix. Pure
  maths, so it gets real unit tests: centred eye reduces to a symmetric frustum, moving the eye
  shifts the frustum the right way, corners map where they should.
- **4 - minimal scene through it.** The first phase that produces something worth looking at, and
  deliberately no more than that: the cheapest thing to draw that makes the effect visible, written
  to files, behind the scene and sink seams. A real renderer is `02_scene_rendering`.
- **5 - close the loop, live.** Camera to tracker to renderer at interactive rate, on a machine
  with a display. The only phase that cannot be finished on this box: g7 with a local window is the
  target. The phone, served over a webapp, comes after and is not part of this initiative.

## Log

Append-only. Newest at the bottom.

- 2026-08-11 : bootstrapped the initiative after the reboot merged to `main`. Surveyed what
  `pose-tools@v0.2.1` provided and what is missing: no face landmarker (MediaPipe's
  `FaceLandmarker` does expose `output_facial_transformation_matrixes`, i.e. head pose relative to
  the camera, which pose-tools does not wrap), no camera model, no renderer. Checked `holo-table`
  for prior art since the name suggested overlap - it is pinch-gesture streaming over a socket,
  nothing to reuse here. Framed the problem as three coordinate frames and wrote Q1-Q6.
- 2026-08-13 : repinned pose-tools to `v0.3.0`, after a cleanup pass over that repo
  (`pose-tools/scratch_space/02_cleanup/`) removed the `geometry.landmark_geometry` shim, the
  `load_env()` import side effect, and the whole template config scaffold. Nothing abyss imports
  was touched: all 7 symbols still resolve, `import pose_tools` no longer reads
  `~/cred/pose-tools/.env`, and `make check` is green. Established as a rule that abyss imports
  pose-tools symbols from where they are defined, never through a re-export module.
- 2026-08-14 : folded the answers to Q1-Q6 into `00_start.md`. Phase 0 is confirmed (face
  landmarker upstream); stereo is ruled out and interpupillary distance accepted as the scale
  reference. Q2-Q5 all came back as "cheapest version now, behind a swappable seam", so that is
  recorded as one decision with the four seams named rather than four separate ones. Parked OpenGL
  and Gaussian splatting / NeRF as suggested tools, neither evaluated - OpenGL needs a GPU context
  so it is a g7 target unless offscreen EGL works. Raised Q7-Q9: the per-machine config the answers
  now require reopens the params layer the reboot stripped down, and phase 2 waits on them.
- 2026-08-14 : folded Q7-Q9. Q7 corrected the axis the Q2 answer had been written on: config is not
  per-machine but per capture device and per display device, passed in as an ingestion config and a
  render config, because the Pixel will record frames that another machine processes. Env var and
  hostname selection both rejected. Q8 picks pydantic, so abyss takes back the dependency
  pose-tools dropped in v0.3.0 - deliberate, not drift. Q9 confirms the phone is config on both
  ends, reached over a webapp later, which is what forces the capture / compute / display split.
  Raised Q10-Q12: values as params literals or a loaded file, whether the ingestion config carries
  the input path, and whether phase 5 is a g7 window or a webapp served to the phone.
- 2026-08-14 : folded Q10-Q12, which unblocks every phase. Config values stay Python literals in
  params, no loader until something outside the repo writes config. The ingestion side splits into a
  camera config and a stream config, since one webcam serves both a recorded clip and a live capture
  and its intrinsics must not move when the source does. Phase 5 stays a local window on g7; the
  phone webapp is later and outside this initiative. Phases 2 and 5 moved to planned. Raised
  Q13-Q14: whether the output side splits display geometry from output sink the way the input side
  split, and whether the webapp is a phase here or a sibling plan folder.
- 2026-08-14 : folded Q13-Q14. The output side splits into a screen config and a sink config, on the
  argument that screen geometry is an input to rendering while the sink acts after it - a stage
  boundary, not symmetry with the input side. Four config models total. Reassessed every sketched
  phase against "own dependencies, own questions, executable cold": phase 4 splits, keeping the
  minimal scene here (phases 3 and 5 cannot be seen to work without something drawn) and spinning
  the real renderer out to `02_scene_rendering`; the webapp becomes `03_phone_webapp`; phase 0 stays
  listed as an upstream prerequisite because it is a different repo but phase 1 blocks on its tag.
  Both spin-off folders created at draft with their own local questions.
- 2026-08-14 : repinned pose-tools to `v0.4.0`, which delivers phase 0. Upstream now has
  `FaceLandmarkerFrame`, `draw_face_landmarks`, the face result helpers including
  `get_facial_transformation_matrix()` and the named iris indices, and
  `ModelManager.ensure_model()` so a machine with no `.task` files can fetch them. Verified from
  here: every new symbol imports, the face model resolves, and `make check` is green. Also fetched
  `~/data/pose/face01.mp4` while working upstream - `yoga01.mp4` contains no face MediaPipe can
  detect, so it was useless for this half of the work. Phase 1 has a clip, a model and a landmarker.
- 2026-08-14 : planned phase 1, measuring against `face01.mp4` before writing rather than after.
  Findings that shaped it. MediaPipe's transformation-matrix depth is self-consistent with a pinhole
  model - `ipd_px * depth` holds to 2.1%, `iris_px * depth` to 4.1% - but the implied focal length is
  ~1000 px on a 1920-wide frame, about 88 degrees, which is MediaPipe's default camera rather than
  this one. Interpupillary distance turns out to be the wrong depth cue: it correlates -0.76 with
  absolute yaw, so it collapses exactly when the viewer turns their head, while the iris diameter
  correlates only -0.21 and the matrix accounts for pose outright.
  Two review corrections, both worth recording because both were my errors. `SignalTracker` is a
  gesture *classifier*, not a smoother: `update()` returns a thresholded derivative and the smoothed
  value is a side attribute, so phase 1 uses the `utils.np_signal` primitives it is built on. That
  claim had been sitting in `00_start.md` and `tracking.md` since the bootstrap survey, uninspected;
  both are now corrected. And the scale analysis in my first draft was wrong: a wrong focal length
  scales depth *only*, because `f` cancels in the lateral conversion, while a wrong head-size
  reference scales all three axes. Verified numerically. It matters because a wrong FOV stretches
  depth against lateral rather than scaling the trajectory uniformly - a distortion of the frustum,
  not a change of units.
- 2026-08-15 : had the phase 1 plan reviewed by a second agent with fresh context, read-only. It
  found real errors and I re-derived its three biggest claims before accepting any of them.
  **My 984 px focal was circular** - it is `ipd_px * tz / 63 mm`, the camera assumption multiplied
  by a head-size assumption, so it measures neither. Fitting by reprojection gives ~900 px on
  `face01`, and the law is `f = (H/2)/tan(31.5 deg)`: fitted/predicted is **1.021 on both a 1080-
  and a 1920-tall clip**, so MediaPipe assumes a 63 degree vertical FOV and the focal follows frame
  height. Reprojection rms is 2.5 px at the fitted focal against 24 px at 984. Padding a frame to
  1920x1920 with identical content moves depth from -50.7 to -88.1 cm.
  **`tz` is the head origin, not the eye** - offset -2.69 cm, swinging 0.92 cm with `corr(yaw)
  -0.66`, which is the same yaw coupling I had rejected the IPD cue for. **The matrix is y-up**
  against the pixel convention's y-down, `corr(pinhole Y, ty) = -0.967`, a sign trap the draft never
  mentioned. Its claim that `tx`/`ty` are not camera-frame metric was itself wrong (`corr = 0.991`),
  but the conclusion not to use them stands for better reasons.
  Chasing a depth-varying clip took three dead ends worth recording: a "grizzly bear GoPro selfie"
  that contains a bear and no human face; a selfie clip that read 45-80 cm only because I had not
  passed `num_faces=1` and was tracking two different people - it is really 45-50 cm; and two
  minutes of a speech where the camera is too wide for any detection at all. Concluded that **no
  real clip can validate depth anyway**, since none carries a measured distance. Built
  `face03_zoom.mp4` instead: a known 1.00x-1.60x ffmpeg zoom ramp over `face01`, where ground truth
  is exact. MediaPipe tracks it to +1.97% mean, 5.83% worst, over a 1.56x depth range. That is now
  the phase's acceptance criterion, replacing "depth stable to a few percent", which a constant
  would have passed.
  **The scale problem is per-identity**, which neither the plan nor the review had right: with the
  focal law fixed, the implied interpupillary distance is 70.4 mm on `face01`, 71.7 mm on its zoomed
  copy, and 62.2 mm on a different subject - 13% apart. So the fix is a per-session identity factor
  estimated from front-facing frames, not a single constant.
  Answers folded in: metres everywhere with cm confined to the function that reads the matrix;
  mirroring becomes a camera config field; `smoothing.py` stays its own module against the review's
  advice. Dropped the synthetic invariant test (it tests its own generator) and `yaw_deg` from the
  CSV (the Euler conversion is a pose-tools utility by our own boundary rule). Kept: a committed
  fixture so the conversion is testable with no clip and no model, filter warm-up from the first
  sample, an explicit no-face policy, and the principal point named among the assumptions.
