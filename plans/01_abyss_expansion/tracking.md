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
  window. Each is swappable, and a seam only counts when the second implementation can be named
  **and can actually plug into it** - phase 4 sharpened that, since a scene-shaped interface would
  have excluded the GL renderer it existed for.
  The companion rule, because it was being read too strictly: "no abstraction without a case today"
  bans building for futures that cannot be named, not for work already written down. Scheduled is not
  speculative, so a Protocol whose second implementation is the next phase earns itself now.
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
- **Open questions live where they apply, which diverges from the skill on purpose.** The
  tracked-development skill puts every `Qn` in `00_start.md`. Here, questions that shape the whole
  initiative stay there (Q1-Q14) and questions scoped to one phase live in that phase's sub-plan
  (Q15-Q17 in phase 2, Q18-Q19 in phase 3, Q20-Q22 in phase 4). Numbering is still global and never
  resets, so a question is unambiguous wherever it sits, and `grep -rn "Q20"` finds it.
  The reason is reading order: whoever picks up a phase reads its sub-plan, and a decision that only
  constrains that phase is evidence they need in front of them, not a cross-reference. `00_start.md`
  stays what it is, the record of why the initiative has its shape.
  The `NEW_ANS:` slot is not used either. Questions are asked in chat and folded in as `ANS:` in the
  same pass, which is the same loop with the file skipped, and it works because the answers arrive in
  conversation rather than as file edits. If answers ever start coming back as edits to the plans,
  the slot earns itself and should come back.
- **Pydantic for config.** Models fix the shape, params supply the values, pydantic validates. abyss
  takes the dependency back although pose-tools just dropped it: pose-tools has no config surface,
  abyss does. Values stay plain Python literals in params (Q10); a loader arrives only when
  something outside the repo needs to write config.

## Phases

Q1-Q27 are answered and the scope is settled. Every phase has a sub-plan. Phase 0 shipped as pose-tools v0.4.0 and
is pinned here.

| #  | Phase                          | Plan | Status |
| -- | ------------------------------ | ---- | ------ |
| 0  | face landmarker in pose-tools  | tracked in pose-tools | done, shipped as v0.4.0 |
| 1  | viewer position from a clip    | [`01_viewer_position.md`](01_viewer_position.md) | done    |
| 2  | camera and screen model        | [`02_camera_screen_model.md`](02_camera_screen_model.md) | done |
| 3  | off-axis projection            | [`03_off_axis_projection.md`](03_off_axis_projection.md) | done |
| 4  | minimal scene through it       | [`04_minimal_scene.md`](04_minimal_scene.md) | done, real renderer spun off |
| 5  | close the loop, live           | [`05_close_the_loop_live.md`](05_close_the_loop_live.md) | in progress, runs on g7 |

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
  `SCREENS["g7_internal"]` is also complete: 344x193 mm from EDID, and the bezel measured with a
  ruler at 4 mm from the top edge of the active area to the centre of the lens, giving a
  `camera_to_centre_m` of `(0.0, 0.1005, 0.0)`, half the panel height plus that gap. Nothing on g7
  is PROVISIONAL any more; g4's entry keeps the word because g4 is still reached over ssh.
  The 640x480 mode comparison is **dropped rather than pending**, and this is a decision not an
  omission. Nothing uses that mode: 1280x720 MJPG is the best the camera has and capture pins it,
  so measuring 480 would be building for a caller that does not exist. The real hazard is that
  `focal_px_for_height` rescales to any height without knowing whether that is meaningful across a
  change of aspect ratio, so the mitigation is to pin the capture mode, which the provenance and
  the guide both now say. Revisit only if something genuinely needs 480.
  Phase 3 is now unblocked on data as well as on maths: it has a real focal length and a real
  screen rectangle to build a frustum from.
- 2026-08-16 : planned phase 3 in [`03_off_axis_projection.md`](03_off_axis_projection.md). Writing
  it moved where the risk sits. The asymmetric frustum itself is four divisions and will work first
  time; what is actually delicate is that three coordinate frames meet in this phase and two of them
  disagree about which way is up. The camera frame points `+Y` down, which is why g7's
  `camera_to_centre_m` is a positive 0.1005 for a camera sitting above the panel, and the screen
  frame points `+Y` up. So the phase has one interesting function, the conversion, and the frustum
  is the easy part after it.
  Recorded as a trap for whoever writes it: the X sign already depends on `mirrored` and
  `eye_position_m` already applies it, so this phase must not flip it a second time.
  The test list gained one the sketch did not have, and it subsumes most of the others. The four
  screen corners must map to the four viewport corners **for every eye position**, not only a
  centred one, because that is the definition of the screen behaving like a window. Swept over a
  grid it catches sign errors, the Y flip, and any confusion between the screen plane and the near
  plane in a single assertion.
  Two questions left open rather than guessed: Q18 whether the pixel viewport transform belongs
  here or in phase 4, and Q19 whether the three frames should be distinct types instead of bare
  3-vectors. Both are better answered while writing the conversion than before it.
- 2026-08-16 : **phase 3 done.** `src/abyss/render/frustum.py`, 63 tests, 169 in the suite, ruff and
  pyright clean, and the regression CSVs byte-identical since this phase only adds a consumer.
  Q18 answered **here**: phase 4 would otherwise reimplement the perspective divide, which is where
  sign errors breed. Q19 answered **document, do not type, and type it the moment it travels**, which
  turned into a design constraint rather than a note: every public function takes the eye in the
  camera frame, so the screen-frame vector never crosses out of the module and stays contained. The
  trigger to revisit is explicit, being any caller outside `abyss.render` wanting one.
  The conversion was worked out rather than assumed and **both X and Y flip, Z does not**. Y because
  the frames disagree about down. X because the camera looks at the viewer, so a viewer's right hand
  lands on the left of an unmirrored image, the same reason video call apps mirror the self view.
  Together a 180 degree rotation about Z, a proper rotation, which is the right shape for two frames
  facing each other.
  **The plan's headline test claim was wrong, and mutation testing is what caught it.** The plan said
  the swept corner invariant would catch sign errors and that if only one test survived it should be
  that one. Breaking each sign in turn showed otherwise: flipping X or Y leaves all 45 swept cases
  **green**, because a wrong eye position builds a frustum that is wrong to match and the corners
  still fill the image perfectly. The invariant is a self-consistency check, not a correctness one.
  So two independent families are needed and neither substitutes for the other. The corner invariant
  covers the near-plane scaling, the matrix and the viewport transform, where all 45 cases fail at
  once for each. The directional and parallax tests cover the conversion, and exactly three of them
  fail when an axis sign flips. Recorded because the general lesson outlives this phase: a test built
  from the system's own outputs cannot catch an error that is upstream of both sides of the
  comparison, however impressive its parametrisation looks.
  Worth noting the tests passed on the first run, which is exactly when they deserve suspicion. The
  mutation pass is the only reason the claim was checked at all.
- 2026-08-16 : planned phase 4 in [`04_minimal_scene.md`](04_minimal_scene.md). Writing it moved the
  seam, which is the one thing in the phase that would have been expensive to get wrong.
  Q4 asks for the simplest scene behind a seam, and the obvious reading puts the seam at the scene:
  an interface yielding geometry, a wireframe today and a model later. That is the wrong place. The
  named second implementation is `02_scene_rendering`, an OpenGL renderer, and it does not produce
  geometry, it produces pixels, so a geometry-shaped interface would exclude the very implementation
  it exists for. The seam is one step up, `render(view_projection, width_px, height_px) -> image`,
  which numpy satisfies today and GL, splatting or a reprojection satisfy later. The scene becomes an
  implementation detail of the wireframe renderer rather than a shared contract. General form of the
  rule: a seam only counts when the second implementation can be named **and can actually plug into
  it**, which is a stronger test than the one `00_start.md` states.
  The scene is a box whose mouth is exactly the screen rectangle, so phase 3's corner invariant
  becomes visible: the border is welded to the image edge or the projection is wrong, and that is
  legible without reading a number. A cube floating mid-depth earns its place because parallax reads
  as relative motion between two depths, not as one thing moving.
  Keeping the scene entirely behind the window is load-bearing, not incidental: no point can then be
  at or behind the eye, so `project_points` cannot raise and the phase needs no clipper. Named
  upgrade is homogeneous-space clipping against the near plane, wanted the first time something is
  meant to poke out through the window.
  One trap found while planning rather than while debugging. Phase 3 maps the panel corners to the
  viewport corners whatever the viewport is, so rendering a 1.78 panel into a 1.33 image fills it
  perfectly, stretched, raising nothing. The check must be explicit and cannot be equality:
  `g7_internal` is 1.7824 while 1280x720 is 1.7778, so a real panel is not exactly 16:9 and an exact
  test would reject the actual device. A 2% tolerance passes g7's 0.26% and catches a genuine mix-up.
  The test list is split by the phase 3 lesson rather than by subject: self-consistency tests on one
  side, and on the other the ones that pin physics, being that the back wall drifts right when the
  eye goes right, that the cube and the wall move by *different* amounts since equal motion would
  mean depth never reached the projection, and that the frame is not blank, which is the guard
  against everything projecting off-screen while the suite stays green.
  Q20-Q22 left open rather than guessed: whether the output resolution belongs on `SinkConfig` or on
  the call, whether a depth-faded wireframe box is Necker-ambiguous enough to need a filled back wall,
  and whether `Renderer` earns a Protocol before its second implementation exists. The last is the
  interesting one, because the project's two stated rules point opposite ways there.
- 2026-08-16 : folded Q20-Q22, so phase 4 is planned rather than draft.
  Q20 turned out to contain a false exclusive. Asked as "where is the resolution stored" it needed
  phase 5 to settle; asked as "who owns it" it settled immediately. The sink owns it, so the `Sink`
  protocol carries a `size` property and the loop is `renderer.render(matrix, *sink.size)`.
  `PngSink` takes its size from `SinkConfig` and phase 5's `WindowSink` from the window it opened,
  and the render path never learns which, so it is a config field *and* a call argument with no
  conflict. No hack now and no refactor later. The bounded-cost check is what made deciding early
  safe rather than confidence in the answer: the number is persisted in no format, so being wrong
  costs one line at two call sites.
  Q21 starts with **no filled back wall**. Depth-faded wireframe only, the scene stays one primitive,
  and the fill gets added if the box reads Necker-ambiguous once there is something to look at. Free
  to defer because it is entirely inside `WireframeRenderer`: no interface, no config, no test
  outside that module. That is the line between deferring a decision and deferring a design.
  Q22 answered **Protocol now, for both `Renderer` and `Sink`**, and the apparent conflict between
  the project's two rules was a misreading rather than a real tension. "A case that demands it today"
  bans abstraction built for futures that **cannot be named**, not for work already written down:
  `02_scene_rendering` is a planned initiative and the window sink is literally the next phase, so
  the second and third implementations are in scope and merely arrive tomorrow. Recorded in the key
  decisions as well, since the strict reading had already come up more than once. Scheduled is not
  speculative.
- 2026-08-16 : reviewed the phase 4 plan with the add/remove pass, five in and three out, and folded
  the accepted eight back into it. The review found one defect rather than only preferences.
  **The box mouth cannot be drawn.** Phase 3's viewport transform puts the panel corners on 1280 and
  720 exactly, not 1279 and 719, which is the correct GL convention and is what
  `test_screen_corners_map_to_viewport_corners` asserts. Those are the outer edges of the last
  pixels, so the right and bottom edges of the mouth fall outside the image and `cv.line` clips them
  silently: the plan's headline visual check would have rendered as two edges out of four. Fixed with
  a frame marker at 98% of the panel rect **in the scene**, not in the drawing, so it goes through
  the identical projection path. A uniform gap to the border is what says the projection is right,
  and a gap opening on one side is easier to see than a line sitting exactly on the edge, so the
  replacement is better than what it replaces.
  The no-face policy turned out to need no code: `viewer_position.py` already calls
  `PositionSmoother.hold()` on faceless frames and writes the held value into the smoothed columns,
  so consuming `*_smooth_m` *is* holding the previous position. Only the leading frames before any
  face are left, where `hold()` returns `None`, and those are skipped. Recorded because it looked
  like new work in the plan and was not.
  `VideoSink` added for a reason about the Protocol rather than about video: **a Protocol with one
  implementation is unvalidated**, and Q22 committed to the interface before its second implementer
  exists. Fifteen lines settle the shape now rather than in phase 5. Its scope is fenced in the plan,
  with a stated tripwire: if it wants an argument beyond fps it has become a feature.
  Removed: painter ordering, since wireframe has no occlusion and depth fading already carries the
  cue; the corner invariant re-tested through the renderer, since phase 3 sweeps it already and it is
  the family the mutation pass proved blind; the determinism and empty-scene tests, one asserting a
  property of numpy rather than of this code and the other having no caller.
  Process note: the add/remove pass is this skill's named convention and it was run from a bare "+5
  -3" without checking what it meant, so it came out one-sided - no pro, con and recommendation per
  item. Asking would have cost one turn.
- 2026-08-16 : recorded where open questions live, as a decision rather than as drift. The
  tracked-development skill puts every `Qn` in `00_start.md`; this initiative has been splitting them
  since Q15, initiative-shaping questions in `00_start.md` and phase-scoped ones in the sub-plan.
  Kept, because whoever picks up a phase reads its sub-plan and a constraint on that phase is
  evidence they need in front of them. Numbering stays global so a question is unambiguous wherever
  it sits. The `NEW_ANS:` slot stays unused for the same practical reason it always was: answers
  arrive in conversation, so asking and folding in happen in one pass. Noted what would bring it
  back, which is answers arriving as edits to the plan files.
- 2026-08-16 : **phase 4 done.** `abyss.render.scene`, `abyss.render.renderer`, `abyss.config.sink`,
  `abyss.sink` and `scripts/render_scene.py`, 39 new tests, 208 in the suite, ruff and pyright clean.
  The phase 1 regression CSVs regenerate byte-identical, checked against
  `~/abyss-baselines/g7-d7bd614/`, since this phase only adds a consumer. The effect is visible: the
  cyan frame marker stays welded to the image border while the room slides and the cube crosses the
  back wall grid.
  **The depth fade test was a proxy, and only mutation exposed it.** Setting `FAR_GAIN` to 1.0, which
  removes fading entirely, left the whole suite green. The test compared whatever was near against
  whatever was far, and the near things happened to be the cyan marker while the far things were the
  grey grid: it passed on base colour alone. Rewritten to compare the *same* base colour at two
  depths. Phase 3's lesson in a new costume - there the test was built from the system's own outputs,
  here from a correlated variable, and both look like coverage until something is deliberately broken.
  The general form: a test that separates two groups must be checked for what else separates them.
  Other mutations behaved. Flattening the cube against the window fails 3 tests, flipping the sweep
  sign 2, disabling the aspect check 2, dropping the zero padding 1, and a renderer that ignores the
  eye translation fails the two directional tests.
  **The clip track renders an extreme view, and it is correct.** `face01_eye.csv` through
  `g7_internal` puts the back wall largely out of frame. Verified by hand rather than by eye: the
  clip's viewer is 0.2435 m above the panel centre at 0.456 m, so the back wall centre is seen
  through the window at y = +0.138 m against a panel top edge of 0.0965 m. It is genuinely outside
  the window. That is the ordinary laptop pose, eyes above the top of the screen, which makes the
  centred sweep the unrealistic one rather than the track. Phase 5 should not tune a demo around a
  viewer who sits where nobody sits.
  The sweep amplitude turned out to be bounded by the scene rather than by taste: at 0.20 m the back
  wall left the frame and took the parallax reference with it, leaving the cube nothing to slide
  against. The bound is about `half_width * distance / depth`, and 0.12 m is now a constant with the
  reason attached.
  Two plan corrections while building. The box contributes 4 back edges and 4 corner connectors, not
  12 edges, since the mouth's own edges cannot be drawn. And phase 4's regression baseline is a
  committed fixture with a real test rather than an out-of-repo checksum: phase 1 went out-of-repo
  because MediaPipe inference is not bit-identical across machines, and projecting a fixed scene
  through a fixed matrix is arithmetic, so that reason does not apply here.
- 2026-08-16 : planned phase 5 in [`05_close_the_loop_live.md`](05_close_the_loop_live.md). Writing
  it turned up one thing that changes what the phase can claim, and it came out of substituting the
  scale estimate into the position rather than from reading the code.
  **After the scale correction, MediaPipe's own depth cancels out.** `estimate_head_scale` divides by
  our focal length, so `depth = depth_m * scale` collapses to `focal * ipd_real / ipd_px`: the
  pinhole formula over the measured focal and the viewer's real interpupillary distance, up to the
  difference between a per-frame value and the median it was normalised by. That means phase 5 is the
  first phase whose output can be **wrong against the world** rather than merely inconsistent with
  itself. A prediction is written into the plan before any code: 120 px of iris separation is 0.50 m,
  100 px is 0.60 m, 60 px is 0.99 m. Sit at a tape-measured distance and check. Every earlier phase
  could only check internal consistency, because the sample clips have no known camera.
  It also promotes the viewer's interpupillary distance from a detail to the last unmeasured number
  in the chain: at the 63 mm population mean, a viewer who is actually 60 mm reads 5% too far away,
  as a constant offset the tape measure will show (Q27).
  Fullscreen turns out to be a geometric requirement rather than a presentation choice. `ScreenConfig`
  describes the whole 344 by 193 mm panel with the camera 100.5 mm above its centre, so a windowed
  render would make every number in the config describe a rectangle that is not on screen.
  Three capture findings from the calibration sessions are folded in as requirements rather than left
  to be rediscovered: pin MJPG 1280x720 since YUYV clamps to 640x480 and the focal was measured at
  720; set `CAP_PROP_BUFFERSIZE` to 1, since the queue measured four frames deep is 160 ms of latency
  live rather than duplicate stills; and check frame statistics rather than the return flag, because
  a camera cut by the lock screen returns `ok=True` with black frames, which downstream is
  indistinguishable from "no face".
  The structural requirement is that the loop takes its source and its sink as arguments, so the same
  loop over a clip with a `PngSink` reproduces phase 4 with no camera and no display. If it can only
  be run through a window, it has been built wrong.
  Q23-Q27 raised: how to estimate head scale with no future frames, `VIDEO` against `LIVE_STREAM`,
  what an evenly-spaced-samples smoother does on a variable frame rate, who owns frame pacing, and
  whether to measure the viewer's interpupillary distance. Each carries a recommendation.
- 2026-08-17 : folded Q23-Q27, so phase 5 is planned. Head scale bootstraps and freezes, the smoother
  keeps its filter and gets its tap count retuned once the real rate is known, the loop owns frame
  pacing, and the viewer's interpupillary distance gets measured once the loop runs and the tape
  measure can show it as a constant offset.
  **Q24 turned out to be bigger than the question asked, because the repo has been wrong about its
  own hardware.** g7 has a Quadro RTX 3000 with 6 GB, driver 580 and OpenGL 4.6, and MediaPipe 1.0.0
  exposes a `GPU` delegate alongside `CPU`. `.github/copilot-instructions.md` has been carrying
  "CPU only. No Nvidia GPU here ... do not write GPU-delegate code paths" and "Headless. No display"
  as repo-wide rules, when both are g4's constraints. Corrected to a per-machine table, keeping the
  useful half of the rule - write for the weaker machine, everything checkable headless on a clip -
  and dropping the false half. `00_start.md` said 4 GB VRAM for g7; it is 6, and OpenGL 4.6 means the
  `02_scene_rendering` spin-off has a real GL context there and does not need offscreen EGL.
  The GPU delegate stays a **measurement, not an assumption**: the enum existing says nothing about
  whether the Linux pip wheel can bind a GPU context for the Tasks API, and the three possible
  outcomes - faster, not faster, fails to initialise - are all useful.
  So step one of the phase became `scripts/benchmark_landmarker.py`, deliberately outside the loop
  and timed against a **recorded clip** so it needs no camera, no display and nobody sitting still.
  That is what makes it portable: the same script runs on g4's old integrated GPU and on g7's Quadro
  and produces comparable rows across delegate, frame size and stage. The interesting result is the
  shape of the gap between the machines rather than either number alone, so results are logged per
  machine.
- 2026-08-17 : measured the landmarker before building anything, prompted by the offer to install
  drivers or libraries on g7. The answer is that **nothing needs installing**, and both halves of
  that are worth recording.
  **The GPU delegate cannot be used and no install would fix it.** `delegate=GPU` fails with
  `ImageCloneCalculator: GPU processing is disabled in build flags`: the pip wheel is compiled
  without GPU support. EGL, GLESv2 and OpenCL are all present and the wheel's `libmediapipe.so` does
  carry GPU calculators, which is exactly the trap - the hardware and the libraries look ready, so
  the failure invites an afternoon of driver installation that cannot work. Only a source build of
  MediaPipe would change it.
  **And it would buy nothing.** CPU inference over `face01.mp4` at 1920x1080 is a median of 11.2 ms,
  p95 11.7, with a face found in all 60 frames: an 89 fps ceiling from inference alone, at a larger
  frame than the loop will use. Q24 is therefore closed by measurement rather than deferred to the
  benchmark: `VIDEO` stays, `LIVE_STREAM` is not needed, and the latency budget must be going
  somewhere other than inference - capture queue depth and display being the candidates.
  The benchmark survives with its question changed, which is a better reason to build it than the
  one it started with. Its delegate axis is dropped rather than skipped, with the failure recorded so
  nobody re-runs it, and capture is added as a stage.
  `.github/copilot-instructions.md` updated again: the GPU bullet now states the measured result
  rather than telling the reader to go and measure it. The GPU remains real and remains relevant to
  OpenGL rendering on g7, which is `02_scene_rendering`, not this.
  One install is still worth making and is not about the GPU: `v4l-utils`, for `v4l2-ctl`. Phase 5
  pins the capture mode, there are four `/dev/video*` nodes and it is not established which is the
  camera, and the `g7_webcam` provenance claims the only other mode is 640x480 - inferred from
  behaviour, never read from the device. Handed to the user to run, per the privileged-command rule.
- 2026-08-17 : `v4l-utils` installed on g7, and reading the device corrected two records and produced
  one number that reshapes phase 5.
  **The camera caps at 30 fps**, in every mode. So the loop's ceiling is 33 ms per frame from
  capture, not the 89 fps that inference alone allows. That reframes yesterday's measurement: 11.2 ms
  of inference inside a 33 ms budget is a third of the frame, with room to spare, and it is the
  reason the GPU question below resolves to "leave it".
  **The `g7_webcam` provenance was wrong** and is corrected. It claimed the only other mode was
  640x480 at a different aspect ratio. MJPG actually offers eight sizes, of which 1280x720, 960x540,
  640x360 and 320x180 are exactly 16:9 and share the aspect the focal was measured at, so rescaling
  within that family is meaningful; 640x480 and 320x240 are 4:3 and 848x480 is 1.767. This also
  promotes an existing test from arithmetic to a claim: `focal_px_for_height(360)` is checked at 360
  because 640x360 is a real mode on this camera.
  Two identification traps recorded so nobody else loses ten minutes. The kernel reports the camera
  as "HP HD Camera", which is the product string on a Chicony module - `g4_internal` carries the same
  product string on different silicon, so the name does not identify the machine; the USB ID
  04f2:b6c8 does. And of the four `/dev/video*` nodes only `video0` is the RGB capture; `video2` is a
  GREY 640x360 15 fps sensor, presumably the infrared one.
  **Correcting yesterday's entry**: it said the wheel's `libmediapipe.so` carries GPU calculators.
  That came from a case-insensitive substring match where "egl" matched inside unrelated words, and a
  narrower search finds no `GlContext` and no `cuda` at all. The strings evidence is unreliable in
  both directions - it does not find the error message the library demonstrably emits either - so the
  runtime failure is the evidence and the strings search should not have been quoted as support.
  The trap is the *system* looking ready, not the wheel.
  Q28 raised and answered from that: there is no `mediapipe[cuda]`, since mediapipe declares no
  extras at all in 1.0.0 locally or 1.0.1 on PyPI, and CUDA is the wrong axis anyway because
  MediaPipe's GPU path is the TFLite GPU delegate over OpenGL ES through EGL. The failure is a
  build-time switch, so the only route to a GPU-capable Python MediaPipe is a bazel source build,
  which trades a pinned public artefact for a local one. Recommended against on the grounds that the
  camera's 30 fps cap makes an 11 ms stage not worth accelerating.
- 2026-08-17 : **correcting today's earlier entry**, which said the only route to a GPU-capable
  Python MediaPipe is a bazel source build. That was reasoning from the error message and from
  memory, and a web search shows it is wrong.
  GPU in the Python Tasks API is **officially supported, and specifically on Ubuntu**: the docs say
  "GPU support is currently limited to Ubuntu platforms", which is what g7 runs. What we hit is a
  **packaging regression**, not a design decision: the delegate worked on the Linux wheel through
  0.10.31 and broke in 0.10.32, whose wheel was built without the GPU flags, producing exactly our
  error. Upstream issue #6231, reported 2026-02-03, and our 1.0.0 inherits it.
  So the routes are cheaper than stated: re-probe after any mediapipe bump, or pin 0.10.31 - though
  that is not one line either, since `pose-tools` requires `mediapipe>=1.0` and would need its own
  change and tag bump first. A source build is the last resort rather than the answer.
  The recommendation does not move, but it now rests on three independent legs instead of one: the
  camera's 30 fps cap makes an 11 ms stage irrelevant, the GPU may not even be faster since the CPU
  path runs XNNPACK and users report no difference in recent versions (issue #6216), and it is
  upstream's bug to fix.
  The lesson worth keeping is about method rather than MediaPipe. The error message named build
  flags, which is true and complete, and reasoning from it produced a confident wrong conclusion
  about what could be done. One search found a version that works. **An error explaining what
  happened is not evidence about what the fix is**, and a library's own issue tracker is a cheaper
  oracle than inference from symptoms.
- 2026-08-17 : **phase 5 step one done**, `scripts/benchmark_landmarker.py` with 7 tests, 215 in the
  suite, ruff and pyright clean. Phase 5 is in progress. Also corrected a stale exit criterion that
  still required running the benchmark "with both delegates", which the Q24/Q28 work had made
  impossible: a GPU row can only record the same build-flags failure.
  Two corrections to the plan came out of writing the script rather than running it.
  **The single frame-size axis does not survive contact.** The tracker stages scale with the capture
  size and the render stages with the output size, and in the live loop those are different numbers,
  1280x720 in and 1920x1080 out. Worse, the plan's cheap fallback of 640x480 is 4:3, so rendering
  into it raises `AspectMismatchError` against a 16:9 panel: half the planned axis was not a legal
  configuration. Split into two axes.
  **And there is no capture stage.** Timing an mp4 decode and calling it capture would be a proxy,
  which is the anti-pattern this repo has now caught three times: a V4L2 MJPG read costs queue
  latency and JPEG decode that a seek-free file read does not. The decode is timed and named
  `decode`. Real capture timing arrives with `video/capture.py` in step two, where it can use the
  real opener rather than a second copy of it. Same reasoning made the budget table exclude the
  sink and print the excluded figure beside it: only `PngSink` exists to time, and a PNG encode is
  not what the window sink will do.
  Measured on g7 over `face01.mp4`, 120 frames, medians. Tracker at 1280x720: decode 1.60, landmark
  11.55, eye position 0.08. Render at 1920x1080: projection 0.04, render 9.12, PngSink 14.41. The
  loop is **22.4 ms rendering native, 44.7 fps**, leaving 10.9 ms of the camera's 33.3 ms for the
  window sink.
  **640x480 is not a cheap fallback: it measured slower**, 12.77 ms against 11.55, in both runs.
  MediaPipe resizes to a fixed model input so the capture size does not change what the network
  sees. Why the smaller frame is consistently worse is not established and the 4:3 squash of a 16:9
  source is the suspect. The axis is settled either way - pin 1280x720, which is also where the
  focal length was measured - and it removes an escape hatch the plan was holding in reserve.
  **The plan's claim that rendering at 1080p "costs nothing" was wrong, and the reason is one line.**
  The render stage is 9.1 ms, of which the projection is 0.02 and drawing 36 anti-aliased lines is
  0.51. The other 8.2 ms is `np.full((h, w, 3), (16, 16, 16), np.uint8)`: passing a **3-tuple**
  rather than a scalar takes numpy off memset and onto broadcast assignment, and the identical array
  from `np.full(..., 16)` costs 0.20 ms. A factor of 41, for a quarter of the live frame budget.
  Worth recording how it was found, because the first decomposition said the opposite. Timing the
  parts by hand gave 0.66 ms total against a 9.12 ms stage, a 14x gap that looked like the benchmark
  being wrong. It was the reproduction that was wrong: the hand-written probe filled with a scalar
  while the renderer fills with a tuple. **A reproduction that is not the real call is evidence about
  the reproduction.** Re-timing `renderer.render` itself confirmed the benchmark to 0.5 ms.
  Raised as Q29 rather than fixed in passing, per the rule that a real defect gets its own step.
  Noting that the minimal and general fixes differ: the scalar is exactly equivalent only while the
  background is grey, and a per-channel assign costs 1.79 ms for an arbitrary colour.
- 2026-08-17 : folded Q29 and planned it as **step two of phase 5**, before the loop is written,
  since the frame budget is what the rest of the phase is designed against.
  The fill assigns per channel at 1.79 ms rather than taking the scalar's 0.20 ms. The scalar is
  four times cheaper again and exactly equivalent *while the background is grey*, which makes
  correctness a precondition on a constructor argument callers are free to change: a non-grey
  background would then render silently in the wrong colour, which is a worse failure than 1.6 ms of
  a 33 ms budget. The branch that picks the scalar when all three channels agree is named as the
  upgrade if the window sink eats the headroom, and deliberately not built - two code paths for a
  saving nothing needs yet.
  **Reusing one buffer across frames is rejected outright rather than deferred**, which is the more
  interesting of the two rejections. It would take the fill to zero and it is wrong at this seam:
  `render` returns the frame and the caller keeps it, so with phase 4's `render_run` holding selected
  frames for the contact sheet while writing the same frame to two sinks, a reused buffer would
  rewrite frames already handed over and the sheet would come out as nine copies of the last one.
  A test that two successive renders return independent arrays goes in to stop it being
  reintroduced as an optimisation later.
  The other test worth naming is a non-grey background rendering in that colour: nothing varies that
  parameter today, and it is precisely the test that fails if someone takes the scalar shortcut.
  No timing assertion goes in the suite - a wall-clock threshold on a shared machine measures the
  load average. The benchmark is the instrument, and re-running it is the exit criterion.
- 2026-08-18 : **phase 5 step two done.** `WireframeRenderer._blank` replaces the tuple `np.full`,
  217 tests, ruff and pyright clean. The render stage at 1920x1080 drops **9.12 to 2.52 ms**, so the
  loop is **16.85 ms and 59.4 fps rendering native**, with 16.5 ms of the camera's 33.3 left for the
  window sink. Inference is now 75% of the loop and everything else together is 5 ms.
  The 1.5 ms predicted in the plan was optimistic, and the reason is worth the line: it counted the
  1.79 ms fill and forgot the 0.51 ms of lines it is added to. Left in the plan next to the measured
  2.52 rather than quietly adjusted.
  Both new tests were checked by mutation rather than trusted. The scalar shortcut fails **exactly
  one** test, the intended one. Buffer reuse fails **three**: the intended one plus both parallax
  tests, which hold two frames at once and were already covering it incidentally. So the explicit
  test earns its place by naming the reason, not by being the only thing that catches it - and the
  parallax tests turn out to have been load-bearing in a way nobody wrote down.
  **Correcting yesterday's entry on 640x480.** It said the smaller capture "measured slower, in both
  runs". A third run reversed the sign, which prompted counting properly: over six paired runs it is
  slower in five, median 12.6 ms against 11.9, but the run-to-run spread at 720 alone is 11.5 to
  12.8 ms, as wide as the gap. So the data supports "not faster" and does not support "slower". The
  decision is unchanged - pin 1280x720, the mode the focal was measured at - but it now rests on
  there being nothing to gain rather than on a penalty. Two runs agreeing looked like a measurement
  and was a coin landing the same way twice.
- 2026-08-19 : **phase 5 step three done.** `src/abyss/video/capture.py` with 10 tests, 227 in the
  suite, ruff and pyright clean, and not one of the new tests needs a camera. The three calibration
  findings are now code rather than log entries: MJPG before the frame size, `CAP_PROP_BUFFERSIZE`
  of 1, and a frame checked for content rather than a return flag trusted.
  Note `src/abyss/video/` existed on disk holding nothing but stale `__pycache__` from the modules
  the 2026 reboot deleted. Untracked, so the package is genuinely new.
  **"Check frame statistics" was underspecified and needed a second condition.** The measured dead
  frames were mean 10.7 and standard deviation 2, and rejecting on darkness alone would reject a
  viewer in a badly lit room - a live loop that refuses to run in the evening. Dark alone is a dim
  room, flat alone is a blank wall in good light, a dead capture is both, so both are required
  together. Mutating the `and` to an `or` fails exactly the two tests that pin it.
  **Reading every pixel to answer "is anything there" would have cost 7.3 ms**, a fifth of the frame
  budget. Sampling every 8th pixel costs 0.26 ms. Caught because the docstring claimed
  "microseconds" and the claim was measured before being left in - it was wrong by two orders of
  magnitude, and the real numbers are now in the file. This is the same shape of mistake as the
  background fill two days ago: the obvious spelling of a trivial whole-frame operation is a fifth
  of a frame, twice running. **Whole-frame numpy in a per-frame path is now a thing to measure on
  sight**, not a thing to reason about.
  `ok=False` and a black frame get distinct errors on purpose: one means the capture stopped, the
  other means it did not stop and that is the whole problem.
  `open_camera` is deliberately not unit tested - three `set` calls and a readback against real
  hardware, where a mock would only assert that the lines were written in the order they were
  written in. The checks are free functions over a frame so that everything else can be tested cold.
- 2026-08-19 : **phase 5 steps four and five done, up to the point a person is required.**
  `LiveScale`, the `sink/` package with `WindowSink`, `src/abyss/loop.py` and `scripts/live.py`.
  245 tests, ruff and pyright clean, and **not one new test needs a camera, a display or a model**.
  The loop ran end to end over `face01.mp4`: 250 frames, 32.4 fps including 1080p PNG encoding,
  250 with a face, 29 calibrating, and the frames were inspected rather than counted - frame 5 is
  the calibrating message, frame 120 carries the cyan marker and the orange cube.
  **The loop takes its tracker as an argument as well as its source and sink.** The plan named two
  seams and a third was needed for the same reason one level down: a landmarker built inside the
  loop would make every test need a model file. `track_with_landmarker` builds the real one.
  **Three states per frame, not two.** The plan had a face and no face; there is also *not yet
  calibrated*, which cannot fold into either, because with no scale there is no correct depth to
  render at and Q23 rules out carrying on at 1.0. The loop shows what it is waiting for and counts
  those frames separately.
  **Correcting the plan's offline equivalence test, which cannot hold as written.** It asked that
  the loop over a clip produce the same frames as phase 4's track mode. It cannot: `LiveScale`
  bootstraps from the first 30 front-facing samples while `estimate_head_scale` uses all 218, by
  design. Measured: **0.939 against 0.941**, 0.2% apart. That is the entire cost of
  bootstrap-and-freeze, against the 16% per-identity spread it corrects, and it is a good number to
  have rather than a problem. Replaced by a `scale=` argument that starts `LiveScale` frozen, so a
  controlled comparison is still possible, plus clip mode as the runbook pre-flight.
  One test caught an off-by-one in its own expectation rather than in the code: the frame that
  completes a bootstrap already renders, so a reset costs one extra calibrating frame and not two.
  Rewritten to compare a run with the reset key against the same run without it, since the absolute
  count was an off-by-one waiting to be asserted wrongly.
  `sink.py` became `sink/` as phase 4 said it would, split as `base` / `file` / `window` rather than
  by convenience: `window.py` is the only module in the repo that needs a display, and having it
  named as one file is the whole reason the package earns its place. Import sites point at the
  defining module; there is no re-export.
  **Written the manual half**: `docs/guides/phase5_live_runbook.md`. Pre-flight on a clip,
  measuring the viewer's interpupillary distance, the live run, the tape measure check with its
  prediction table, seven known failure modes with the error each prints, and a fill-in template for
  the log entry. Separate from the plan because it outlives the phase - it is what someone follows
  at the desk the next time the camera moves or a different person sits down.
  **What remains is the part that needs a person**, and it is only that: sitting in front of g7,
  measuring an interpupillary distance, and holding a tape measure. Nothing else is blocked.
- 2026-09-01 : first user run of the runbook, and it found three defects in one go - all in the
  half that had never been executed rather than in the code paths the tests cover.
  **`--viewer-ipd-mm` after the subcommand was rejected by argparse.** The shared options were
  defined on the top-level parser, so `live.py camera --viewer-ipd-mm 60` failed while
  `live.py --viewer-ipd-mm 60 camera` worked, which is the opposite of how anyone types it. Both
  the script's own docstring and the runbook told the user to type the form that does not parse.
  Fixed with a parent parser so the shared options hang off each subcommand. `tests/scripts/
  test_live.py` now parses every documented invocation, **including the command lines extracted
  from the runbook itself**, which is what would have caught it: documentation and code disagreed
  and nothing compared them.
  **The loop reported the depth nowhere.** The runbook's section 4 says to read the depth the loop
  reports; there was nothing to read. The run is fullscreen so no terminal is visible, and logging
  a position at 30 fps is unreadable. `annotate_position` now writes the eye position and the
  apparent iris separation across the top left, so the frame carries both halves of the comparison.
  The phase's stated exit criterion had been unmeasurable and the plan did not notice, because
  every test asserted on frames rather than on being able to read one.
  **The prediction table was written for 63 mm only.** Depth scales linearly with the viewer's
  interpupillary distance, so at the user's 60 mm every row was 5% out. The table now gives both
  columns and says plainly that the reported depth is what to trust and the interpupillary distance
  is what to correct.
  Verified end to end afterwards on `face01.mp4` at 60 mm: head scale 0.894, which is exactly
  60/63 of the 0.939 at the default, and the frame reads `eye -0.018 -0.110 0.430 m, iris 123 px`
  against `881 * 0.060 / 123 = 0.430`. The arithmetic closes.
  The user also hit `uv run` rather than `uv run --no-sync`, which re-synced and rebuilt abyss. No
  harm this time, but it is the documented way to silently revert an editable pose-tools install,
  so the runbook now says so where the command is.
  257 tests. **The lesson is about which half gets tested**: everything the tests covered worked
  first time, and all three failures were in the seam between a document and a command line, which
  no test touched until now.
- 2026-09-01 : **the loop ran live on g7 and the tape measure check passed.** The user reports a
  reasonable calibration scale and reported distances close to the truth at 50, 70 and 100 cm, with
  the residual error attributed to positioning rather than to the model. That closes the phase's
  real exit criterion: this is the first result in the initiative that is right against the world
  rather than merely self-consistent. Exact numbers were not captured because the machine became
  too laggy to copy the terminal, which turned into its own finding below.
  **The lag was diagnosed and it was our own doing.** CPU was 93% idle, 25 GB of RAM free, the GPU
  idle at 10%, no thread hog, and the load average already falling (41 to 33 to 24). What was
  actually wrong: **swap was 100% full, 2032 MB of 2047, while 25.5 GB of RAM sat free**, with one
  VS Code process alone holding 604 MB on disk. Every interaction was faulting pages back in.
  The cause was `scripts/live.py clip`, which read the whole clip into memory: 250 frames of
  1920x1080 is 1.55 GB. That was enough to evict an idle desktop into a 2 GB swapfile, and Linux
  never pages anything back proactively, so the lag outlived the run by half an hour. The docstring
  had called slurping "acceptable because the sample clips are seconds long", which was wrong in a
  way no test could catch - the suite feeds the loop a list of small frames, so nothing measured the
  real cost.
  Fixed: `clip_frames` streams, holding one frame instead of all of them, reading the first eagerly
  because the caller needs the frame size before the loop starts. Peak resident set drops from about
  1.9 GB to **374 MB** with byte-identical behaviour - head scale 0.894, 250 frames, same log line.
  The loop already took an iterable, so this cost nothing.
  **The general lesson is about where memory gets measured.** Every performance number in this phase
  came from the benchmark, which times stages and never looks at memory, and the one allocation that
  actually hurt the user was in a manual script the benchmark does not cover. Wall clock was
  measured to three decimal places and 1.55 GB went unnoticed.
  Swap needs clearing to recover the machine (`swapoff -a && swapon -a`, safe with 25 GB free), which
  is a privileged command and was handed to the user.
- 2026-09-01 : first live run's numbers, and the swap reset fixed the machine.
  `558 frames in 67.6 s, 8.3 fps: 472 with a face, 86 held, 29 calibrating`, head scale frozen at
  1.496 from an implied interpupillary distance of 40.1 mm.
  **The 40.1 mm is correct and is not a broken measurement**, which is worth writing down because
  it reads like one. MediaPipe assumes a 63 degree vertical field of view, so 587.5 px of focal at
  720 tall, where g7 really is 945: a factor of 1.609. Its head model divided by our real focal
  gives 64.5 * 587.5 / 945 = 40.1 mm, and 60 / 40.1 = 1.496. Both reproduce to the decimal. The
  final depth is unaffected because MediaPipe's own depth cancels out, which the tape measure
  confirmed independently.
  **8.3 fps against a predicted 44 is a real problem**: 120 ms per frame with 104 ms unaccounted
  for. The suspect is the one stage the benchmark never measured, the window sink, because only
  `PngSink` exists to time offline and a window is not a file. That gap was named in the plan and
  it is exactly where the time went.
  So the loop now measures itself: `LoopStats.stage_ms` reports a median per stage and the report
  line says how much of the actual frame time the stages account for. **Capture is timed too**,
  pulled out of the iterator, because the camera read happens between stages and would otherwise
  show up as unaccounted. Offline on `face01`: capture 1.7, track 14.8, render 4.0, sink 20.2,
  measured 40.7 of 42.5 actual.
  The eye conversion is folded into `render` rather than given its own stage: the benchmark measured
  it at 0.09 ms, and a stage that cannot be the bottleneck is noise in the report.
  `frame_for` extracted while doing it, which is a better shape anyway - the loop's three states are
  one function returning the frame and which state produced it, rather than a branch threaded
  through the timing code.
  **The general lesson matches the memory one from an hour earlier**: everything measured was fine
  and the answer was in the stage nobody could measure. The plan even said the window sink was
  unmeasured and moved on.
- 2026-09-01 : **the 8.3 fps is the camera, not the loop, and the instrumentation found it in one
  run.** `capture 99.0 track 12.2 render 3.3 sink 5.4 | measured 119.9 of 119.8 actual`. A second
  run at 1280x720 output gave capture 97.4, unchanged, ruling out the render size.
  **My suspect was wrong.** I predicted the window sink, on the argument that it was the one stage
  never measured. The sink is 5.4 ms. The unmeasured stage was the right place to look and the
  wrong place to guess, which is the distinction: instrumenting cost twenty minutes and answered it,
  and no amount of reasoning from the symptom would have.
  The cause is `exposure_dynamic_framerate`, a UVC control that lets the camera drop its own rate to
  buy longer exposures in low light. It defaults to 0 and was 1. The camera advertised 30 fps in
  `--get-parm` the whole time and delivered about 10, so nothing short of timing the read pointed at
  it. Set to 0 with `v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0`, no root needed.
  99 ms is almost exactly three frame intervals at 30 fps, which is what a camera dropping to 10 fps
  looks like from the far side of a blocking read.
  The loop now warns when capture exceeds its own work, naming the control and the command. That is
  a hardware quirk encoded in library code and it earns its place: the loop is the only thing that
  can see the ratio, and a starved loop and a slow loop have opposite fixes - optimising a starved
  one changes nothing.
  With 21 ms of work the loop should reach about 47 fps once fed, which would be capped at 30 by
  the camera. Numbers to confirm on the next run.
  **Two wrong guesses in two days, both corrected by measurement in minutes**: the render fill (I
  reproduced it wrongly and blamed the benchmark) and this. The pattern in both is that the symptom
  was honest and the inference from it was not.
- 2026-09-01 : two more runs after clearing `exposure_dynamic_framerate`, and a new suspect that is
  again our own code.
  `285 frames, 14.8 fps, capture 44.0 track 13.3 render 4.0 sink 5.7`, and with an extra lamp on
  `144 frames, 14.7 fps, capture 40.3 track 16.6 render 4.5 sink 5.3`.
  So the control was worth 8.3 to 14.8 fps and capture 99 to 44 ms. **The extra light did almost
  nothing for the rate** - 44 to 40 ms - **and a great deal for tracking**: face loss went from 54 of
  285 to 1 of 144. Two separate effects that both look like "it works better with light", worth
  keeping apart: the rate is the camera's exposure policy, the tracking is the landmarker's.
  What is left is a clean factor of two. Total frame time is 67 ms against a camera interval of
  33.3, and the loop's own work is 23 to 26 ms - comfortably inside one interval, yet it lands on
  exactly two. 14.8 and 14.7 fps against a camera doing 30.
  **The suspect is `CAP_PROP_BUFFERSIZE` of 1, which is ours.** It came from the calibration
  sessions, where a reader that went idle and came back got a four frame old still. That finding is
  real and it is about idle-then-read. A loop reading continuously has the opposite problem: with a
  single buffer the driver has nowhere to put the next frame while userspace works, so a frame
  arriving during the loop's own 25 ms is dropped and the read waits for the one after it, halving
  the rate however fast the loop is. The same setting fixes one access pattern and breaks the other,
  which is why it was applied confidently and wrongly.
  Not fixed by guessing. `scripts/probe_capture_rate.py` sweeps the buffer size against a simulated
  work time and reports what arrives - camera needed, no display, no model, no face. It decides
  whether one line fixes this or whether the loop needs the capture thread the plan already names as
  the upgrade.
- 2026-09-01 : **the buffer size was it, and one is exactly half.** `probe_capture_rate.py` on g7,
  25 ms of simulated work against a 30 fps camera: 1 buffer gives a 42.9 ms read and 14.9 fps, 50%
  of the camera; 2 gives 8.0 ms and 29.7 fps, 99%; 3 and 4 are identical to 2. So two is the whole
  fix and three buys nothing, which also makes two the right choice on latency - the smallest queue
  that reaches full rate.
  The 8.0 ms read is the loop waiting out the remainder of the camera's interval after 25 ms of
  work, which is what a loop that keeps up looks like from the inside.
  `CAPTURE_BUFFERS` is now 2, with the table in its docstring.
  **The interesting part is why one was chosen.** The four-frame queue and this are the same
  setting pulling opposite ways for two different access patterns. A reader that goes idle and comes
  back wants the queue short, and that is where the finding came from - four identical calibration
  stills. A reader that works between reads wants a spare buffer for the driver to fill, or the
  frame arriving during that work is dropped. One satisfies the first and starves the second, and it
  was carried from the calibration script into the live loop without noticing the access pattern had
  inverted. **A setting justified by a measurement is only justified for the situation it was
  measured in.**
  Expect about 30 fps on the next run, capped by the camera, with capture around 8 ms.
