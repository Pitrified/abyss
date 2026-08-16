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
  frame). Phase 2 builds three of them plus a `ViewerConfig`; the sink model waits for phase 4, where
  its first caller is.
  No env var, no hostname lookup. The screen/sink line is a stage boundary: geometry is an
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

Q1-Q17 are answered and the scope is settled. Phases 1 and 2 have sub-plans; for 3 to 5 the
sketches below are all there is until each is written up. Phase 0 shipped as pose-tools v0.4.0 and
is pinned here.

| #  | Phase                          | Plan | Status |
| -- | ------------------------------ | ---- | ------ |
| 0  | face landmarker in pose-tools  | tracked in pose-tools | done, shipped as v0.4.0 |
| 1  | viewer position from a clip    | [`01_viewer_position.md`](01_viewer_position.md) | done    |
| 2  | camera and screen model        | [`02_camera_screen_model.md`](02_camera_screen_model.md) | done |
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
  display device. Pydantic models constructed from literals in params and passed in rather than
  looked up: camera, stream, screen and viewer here, sink in phase 4 with its first caller.
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
- 2026-08-15 : phase 1 implemented on `feat/viewer-position`. `src/abyss/viewer/` holds `camera.py`
  (the five-number placeholder phase 2 replaces), `eye_position.py` and `smoothing.py`, driven by
  `scripts/viewer_position.py`, with 43 new tests - 51 total, none needing a clip or a model.
  **The acceptance test passes at +0.35% mean error** against the known zoom ramp (sd 0.45%, worst
  2.31%) over a 1.61x depth range. Better than the +1.97% the raw matrix gave, because the
  eye-offset correction removes part of the bias too.
  Measured, not assumed: the head-scale estimator reports an implied interpupillary distance of
  66.9 mm for `face01`'s subject and 57.7 mm for `face02`'s, a 16% spread, corrected to the
  configured 63 mm by factors of 0.941 and 1.092. The per-identity problem is real and now handled.
  Filter width is **5 taps** (0.2 s at 25 fps), recorded here rather than tuned silently. Measured
  frame-to-frame jitter on the near-static `face01`: x 1.43 -> 1.27 mm, y 1.35 -> 1.08 mm,
  z 2.05 -> 1.25 mm. Depth benefits most, which is where the jitter was worst.
  `face02_portrait` turned out to have one frame with no face out of 240, so the gap path ran for
  real rather than only in tests: the smoother held 0.4815 m across it and the CSV records the gap.
  Two things the type checker caught that the plan had not: MediaPipe types landmark coordinates as
  optional, so a landmark without them now yields no sample rather than a crash, and `hold()`
  returns `None` before the first sample, which every caller has to handle.
- 2026-08-15 : planned phase 2. Two discoveries while measuring, both of which change what it should
  build. **g4 has a webcam**: `HP HD Camera` on uvcvideo at `/dev/video0`, which fails to open only
  because the user is not in the `video` group - a permissions fix, not a hardware limit, so
  `.github/copilot-instructions.md` claiming this box has no camera is wrong. **g4's panel reports
  its own geometry**: EDID on `card1-eDP-1` gives 309x173 mm at 1920x1080, so screen size is
  machine-readable rather than measured with a ruler, and the same read works on g7. Four of the
  five EDID nodes are zero bytes (disconnected ports), so the reader has to find the connected panel
  rather than read a fixed path, and the detailed timing descriptor is the source, not the
  centimetre-rounded basic block.
  Also worked out that a laptop's screen-to-camera transform is lid-independent, since the camera
  rides in the bezel - one offset covers every lid angle, which is not true of a separate webcam on
  an external monitor. Rotation is left unmodelled for now: it is the identity on a laptop and phase
  3 only needs the corners.
  Raised Q15-Q17: where the viewer's interpupillary distance lives now that it is clearly not a
  camera property, whether device entries carry published or measured values, and whether to fix the
  `video` group on g4 so phases 2-4 can be developed against a live camera.
- 2026-08-15 : folded Q15-Q17. The viewer's interpupillary distance gets its own `ViewerConfig`, so
  five models rather than four - a viewer is not a device, and phase 1 had it on the camera only
  because nothing else existed. Matching a config to a particular person is deferred: one viewer
  today, and the session estimator already derives their scale.
  Device values get measured rather than looked up, but not by calibration: one object of known size
  at a known distance gives `f_px = size_px * distance / real_size`, which is the single number the
  config consumes. A tape measure is good to a percent or two, well inside the 13% per-identity
  error phase 1 already corrects.
  On the webcam finding, the correction is mine to take: g4 is an ssh box, so nobody sits in front
  of it and a live frame shows an empty room. The hardware discovery stands and the repo docs are
  still wrong, but it unblocks nothing - live capture belongs on g7, which has a camera *and* a
  person. Left in the plan as a documentation fix rather than an opportunity.
- 2026-08-15 : review pass on the phase 2 plan by a second agent with fresh context, as was done for
  phase 1. Every factual claim it checked reproduced this time - EDID 309x173 from the detailed
  timing descriptor, the 1.021 focal ratio, pydantic present at pose-tools v0.2.1 and gone at v0.3.0
  and absent from abyss's lock - so no repeat of phase 1's circular number. What it found instead
  were gaps.
  Folded in: the focal length is resolution-dependent, so FOV is the canonical stored value and
  `f_px` is derived from the actual frame height at runtime. This was going to bite immediately -
  `face02_portrait` is 1080x1920 against the other two clips at 1920x1080, so no single literal
  `f_px` could serve one camera across both orientations. Related, published FOV specs are diagonal
  almost without exception, and pasting one into `(H/2)/tan(fov/2)` is about 20% wrong on 16:9, so
  the field is named `fov_vertical_deg` and specs get converted before entry.
  Also folded: an `unknown_clip` camera entry, since the registry named devices and none of them
  shot the three clips; the bezel offset measured rather than left to phase 3, which consumes it;
  the `estimate_head_scale` call sites named explicitly, since `ipd_m` moving to `ViewerConfig`
  changes its signature; `principal_point` stays derived rather than configured.
  `SinkConfig` deferred to phase 4. Nothing constructs or reads it, and configs are passed
  individually rather than bundled, so adding it later widens no existing signature - assessed
  before deciding. The reason to wait is that its fields are a guess until phase 4 knows what it
  draws, and a model pinned to an invented shape is worse than an absent one because it looks
  authoritative. The four-way split and the Q13 argument stand; only the code waits.
  EDID demoted from library code to `scripts/read_edid.py`, run once per machine with the answer
  typed into the params literal, per Q10. The package never reads `/sys`. Two traps recorded in the
  plan for when it runs on g7: sysfs reports `st_size` 0 for every `edid` node including the live
  one, so filtering on size discards the panel, and the `status` file names the connected connector
  outright.
  Corrected in `00_start.md`: the boundary paragraph assigned both landmark smoothing and a
  camera-intrinsics model upstream, and both came back downstream once built. Smoothing because the
  general part (`np_signal`) was already upstream and what abyss added was policy; intrinsics
  because climbing-wire would want the FOV-to-focal relation, three lines of trigonometry, not a
  pydantic registry of abyss's own devices. The rule holds, it just answers per piece of code rather
  than per topic.
  Exit criterion 4 was overstated and is now conditional. "The fallback reproduces phase 1" is
  achievable but not automatic: it needs FOV unset, resolution taken from the frames, and
  `ViewerConfig.ipd_m` defaulting to the same 0.063. Verified by CSV diff against the existing
  phase 1 outputs rather than by eye. Worth keeping: with the head-scale estimator active the focal
  cancels exactly out of the lateral position, so a wrong focal is a pure depth error.
  Per-identity spread corrected throughout from the 13% planning estimate to the 16% actually
  measured (66.9 mm against 57.7 mm).
- 2026-08-15 : phase 2 implemented. `src/abyss/config/` holds `CameraConfig`, `StreamConfig`,
  `ScreenConfig` and `ViewerConfig`, with the device registry and its literals in
  `params/abyss_devices.py`. `viewer/camera.py` is gone and nothing imports it. 97 tests, ruff and
  pyright clean.
  The exit criterion held exactly: all three clips produced CSVs **byte-identical** to the phase 1
  outputs, diffed rather than eyeballed. Head scale still 66.9 mm implied against 63.0 mm
  configured, scale 0.942.
  Three things came out differently from the plan, each recorded because each is a decision rather
  than a detail.
  First, `CameraConfig` has no resolution field. Writing it made the reason obvious: a camera feeds
  several streams at several resolutions, so a stored resolution is a second source of truth that
  can disagree with the frames in hand, and the plan's own rule says the focal follows the actual
  frame height. So the device config carries only what is resolution-independent, and a small
  frozen `FrameGeometry` binds it to a real frame at run time, supplying `focal`,
  `principal_point` and `mirrored`. The only resolution stored anywhere is
  `focal_measured_at_height`, which is what makes a measured focal interpretable at all.
  Second, pydantic swallows exception types. A validator raising `AmbiguousIntrinsicsError` reaches
  the caller as `ValidationError` with the message intact and the type gone, not even as
  `__cause__` - found by a test asserting the type, which failed. The named classes stay because
  they say what went wrong where it went wrong, but the docstring now says plainly that only the
  message survives. Two tests were dropped rather than left asserting something untrue.
  Third, the screen offset is **not measured**, contrary to the plan bullet that said to reach for a
  ruler. g4 is reached over ssh, so there is no hand here to hold one. The entry carries half the
  panel height plus a 10 mm bezel guess, with the word PROVISIONAL in its provenance and a test
  asserting that word is there, so phase 3 cannot consume it believing it was measured. The panel
  size next to it is exact, straight from EDID.
  `scripts/read_edid.py` replaces the planned `from_edid()`; run here it reports 309x173 mm on
  `card1-eDP-1` and disconnected on the other four connectors, matching the literal it produced.
- 2026-08-16 : moved to g7, the machine with a camera and a person in front of it. Nothing was
  committed this session: the work was establishing what g7 actually is, and one new script.
  Two blockers the handoff listed turned out not to exist. The `video` group is **not** needed:
  `id` shows no such group, but systemd-logind grants the active session an ACL, and `getfacl
  /dev/video0` shows `user:pmn:rw-`. Opening the device from Python works with no privileged
  change, so no sudo handoff was required. And `scripts/read_edid.py` ran here unchanged, giving
  `card1-eDP-1` at 344x193 mm, so g7's panel size needs no ruler either.
  A number to be careful with: half of g7's panel height is 96.5 mm, which is numerically identical
  to g4's entire `camera_to_centre_m` Y of 0.0965. That value is g4's 86.5 mm half-height plus its
  10 mm bezel guess. Copying the literal across would look correct and be wrong by exactly the
  bezel, which is the one part still unmeasured.
  Device map, from `/sys/class/video4linux/*/name` rather than `v4l2-ctl`, which is not installed
  and turned out not to be needed. `video0` is the RGB camera and `video2` is an infrared one on a
  separate USB interface, `1-7:1.0` against `1-7:1.2`; `video1` and `video3` are their metadata
  nodes. Measure against `video0`. The 640x360 frames that looked like a second mode in the first
  probe were the IR camera.
  g7's webcam is USB **04f2:b6c8**, where the `g4_internal` provenance string records 04ca:7063.
  Both are labelled "HP HD Camera" and they are different hardware, so no measurement transfers
  between the two registry entries.
  **The pixel format decides the resolution.** OpenCV defaults to YUYV, which caps at 640x480 here
  and silently ignores a request for anything larger, returning 640x480 while reporting success.
  MJPG reaches 1280x720. 1080p does not exist in either. A bare `cv.VideoCapture(0)` therefore gets
  the small mode, and the FOURCC has to be set before the frame size or it is clamped back.
  This matters beyond throughput: 640x480 is 4:3 and 1280x720 is 16:9, so the two modes are not
  self-evidently the same field of view sampled at two densities. `CameraConfig.focal_px_for_height`
  rescales by height alone, which is only valid if the vertical field of view is shared. Whether it
  is has **not** been established. `measure_focal.py compare-modes` exists to settle it and has not
  been run.
  New script `scripts/measure_focal.py`, manual and windowless so it works over ssh. `capture`
  forces MJPG then the size, discards frames until auto-exposure settles, and writes a clean PNG
  plus one with a measuring scale over it; `compare-modes` captures both modes of a fixed scene;
  `solve` turns an apparent size in pixels, a distance and a real size into `focal_px` and prints
  the registry snippet. The focal length itself is still unmeasured, so `g7_webcam` is untouched.
  An episode worth recognising if it recurs. The first captures came back uniformly black: mean
  10.7 of 255, standard deviation 2, flat across 90 consecutive frames, no spatial structure at all.
  That is not an exposure problem, and the controls confirmed it, with auto-exposure on and already
  pushed to a long 312 integration. It coincided with g7 sitting at its login screen and resolved
  after the machine was interacted with, without any code change. The mechanism was never proven,
  so this is recorded as a correlation and not a cause.
  The consequence for phase 5 is real regardless of mechanism: the capture reported `ok=True` and
  handed back black frames, and a face tracker fed black frames reports "no face" rather than an
  error. If this box cuts the camera while the session is locked, the live loop fails silently. The
  live path should check frame statistics, not just the read flag.
  Auto-exposure blew out 54% of the frame against a lit wall, which would leave a white A4 target
  with no measurable edge. `capture` now takes `--exposure` and warns above 15% clipping; 80 brings
  it to 9.4% here. The frame also shows visible barrel distortion toward the edges, so the target
  should be kept near the centre where it is weakest.
  Regression baseline regenerated on g7 at `d7bd614` and copied to `~/abyss-baselines/g7-d7bd614/`
  with sha256sums, before any change. Per-machine on purpose: MediaPipe CPU inference is not
  guaranteed bit-identical across hardware, so g4's CSVs prove nothing here. `make check` is green,
  97 tests.
- 2026-08-16 : added `scripts/calibrate_camera.py`, ChArUco calibration off a screen, which
  supersedes the A4 method as the way to get a focal length. `measure_focal.py` stays as the
  fallback for when there is no second screen.
  The prompt for it was the Kindle and the Pixel being available as targets, with the objection
  that neither the displayed size nor the distance is known. Both objections dissolve. The distance
  is not needed at all: one head-on view of a known-size target is degenerate because `f` and `Z`
  only appear as `f / Z`, which is exactly why the A4 method needs a tape measure, while several
  views at different orientations constrain the intrinsics through the orthonormality of the
  rotation columns and hand back the distance as a result. And the displayed size does not affect
  the focal at all: scaling the board scales the recovered translations and leaves the intrinsics
  untouched. Both are pinned by tests rather than asserted.
  Measured caveat on that second claim: invariance held to 0.001 px between scale 1.0 and 2.0, but
  at scale 0.5 the recovered focal drifted 0.4%, which is conditioning from tiny object
  coordinates rather than a real scale effect. So the true size is still worth supplying, and a
  screen gives it exactly from the pixel pitch with no ruler. `board` prints the diagonal the ppi
  implies as a check against the spec sheet: 6.87 in for the Kindle against an advertised 6.8.
  This method also removes a bias the A4 method cannot detect. A hand-held sheet is never exactly
  fronto-parallel, and tilt by 10 degrees shortens it by `cos 10` and biases the focal low by 1.5%
  with nothing to reveal it. Zhang's method requires tilt instead of suffering from it.
  No new dependency: `opencv-contrib-python` 5.0.0 is already pinned and carries aruco,
  `CharucoBoard` and `CharucoDetector`. `calibrateCameraCharuco` is **gone** in OpenCV 5, so the
  path is `CharucoDetector.detectBoard` then `board.matchImagePoints` then `cv.calibrateCamera`.
  Building OpenCV from source was considered and rejected: the wheel already has FFMPEG, V4L2, Qt5
  and IPP, only GStreamer and the non-free algorithms are absent and neither is wanted, and a
  hand-built install in the uv venv would be silently reverted by `uv sync` exactly as the editable
  pose-tools install is. If the OpenCV 5 API churns further the cheap lever is a 4.x pin, not a
  source build.
  Two pyright findings worth keeping in mind, both from imprecise cv2 stubs against a verified
  runtime: `matchImagePoints` is typed for a sequence of matrices but takes the single `(N, 2)`
  array `detectBoard` returns, and `getChessboardCorners` is typed as a sequence but returns an
  ndarray. One real bug came out of the same pass: `cv.imread` returns `None` on an unreadable
  file and that was being passed straight into detection.
  Detection measured on the generated board: 48 of 48 corners on the full panel canvas, 36 of 48
  after a 4x downscale and a 12 degree rotation. Partial views are fine, which is the reason for
  ChArUco over a plain checkerboard. 103 tests now, ruff and pyright clean.
  Still unmeasured: the focal itself, the mode comparison, and the bezel gap. The tooling is ready
  for all three and none has been run against a real board yet.
- 2026-08-16 : **g7's webcam is measured.** `focal_px=945.0` at height 720, from ChArUco
  calibration off a Kindle Paperwhite 11. Two runs agree to 0.49%, 944.98 from 15 views at 0.26 to
  0.33 m and 940.40 from 8 views at 0.38 to 0.47 m, so they converge from genuinely different
  geometries rather than repeating one setup. Reprojection rms 0.263 and 0.251 px.
  **The vertical field of view is 41.7 degrees, not the 63 that MediaPipe assumes.** That implies a
  587.5 px focal where the real one is 945, a factor of 1.60, so any depth taken on this camera
  through the fallback is 0.625x the truth, 38% too small. That dwarfs the 16% per-identity head
  scale phase 1 corrects, and it is the first hard evidence that the fallback is not merely
  imprecise but wrong on real hardware. The sample clips are untouched: they use `unknown_clip`,
  and the regression CSVs stayed byte-identical through this change, checked by sha256.
  Getting there took six attempts, and none of the failures were what the guidance predicted.
  First the board was backlit against a window, so the camera metered for the window and the board
  went black. Then, with the light moved behind the camera, the opposite: the reader's front light
  and the room together blew the white squares out, and the glare bled over the marker bits. That
  is the failure that misleads, because the board looks perfect to the eye. It showed up as 64 to
  74 candidate quads per frame with zero decoded and 44% to 66% of the pixels inside those quads at
  or above 250. The board was never blurred; it measured 258 Laplacian variance against 60 to 98
  for the background, the sharpest thing in frame.
  The lesson was that the operator cannot be asked to guess an exposure, and the advice to turn the
  front light up or down was reversed twice before that was accepted. `find_exposure` now sweeps a
  ladder at preflight and keeps whatever decodes the most markers. The reader's light being set to
  a mid-sepia warm tone was also a contributor and explains the brown squares in the debug frames,
  which had been misread as camera white balance.
  A real capture bug surfaced only because the user noticed four saved views looked identical. V4L2
  keeps filling its queue while nobody reads, so a read after a four second wait returns four
  second old content. Measured: frame to frame difference of 2.3 to 3.2 across the duplicates
  against 10.7 to 18.4 for every later pair, and probing by idling then reading rapidly put the
  content jump between the fourth and fifth read, confirming queue depth 4. Fixed with
  `CAP_PROP_BUFFERSIZE=1` plus an explicit flush before each kept frame. The next run had 15 of 15
  views distinct, minimum consecutive difference 16.8.
  Coverage stayed near the frame centre, because holding a board at the edge of frame without live
  feedback is guesswork. That is fine and no live preview is needed: the two things centre-only
  coverage degrades are exactly the two the model does not consume. The principal point moved 11 px
  between runs, 635 to 646 against a 640 centre, and `k1` flipped sign between -0.007 and +0.006.
  Both are loosely constrained and both are unused, while the focal, which is consumed, is stable
  to 0.49%. Note this also corrects an earlier guess: the barrel distortion that looked obvious in
  a wide room shot measures near zero.
  `test_no_camera_is_measured_yet` was replaced rather than deleted. It pinned a fact about the
  world that has now changed, so it became tests for the measured focal and its rescaling, and the
  existing clip-camera test gained the reason it now matters more.
  Still unmeasured: the mode comparison at 640x480, and the bezel gap for a `g7_internal` screen.
