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
- **Config is now load-bearing.** Per-machine camera and screen numbers (g4, g7, Pixel 7 Pro) reopen
  the params layer the reboot deliberately kept minimal. Q7-Q9 decide how, and gate phase 2.

## Phases

Q1-Q6 are answered, so the sequence is settled and phase 0 is confirmed. No sub-plan files exist
yet - the sketches below are all there is until each phase is written up.

| #  | Phase                          | Plan | Status |
| -- | ------------------------------ | ---- | ------ |
| 0  | face landmarker in pose-tools  | -    | planned |
| 1  | viewer position from a clip    | -    | planned |
| 2  | camera and screen model        | -    | draft, gated on Q7-Q9 |
| 3  | off-axis projection            | -    | planned |
| 4  | render a scene through it      | -    | planned |
| 5  | close the loop, live           | -    | draft, needs g7 |

Status values: draft / planned / in progress / done / superseded / discarded.

Sketch of each, to be replaced by real sub-plans as they are picked up:

- **0 - face landmarker in pose-tools.** `landmark/face.py` over `BaseLandmarkerFrame`, plus a
  `face_landmarker` entry in `ModelManager.MODEL_FILENAMES`, released as a tag and pinned here.
  Confirmed by Q1. Whether pose runs alongside it is decided by how cleanly the two compose.
- **1 - viewer position from a clip.** Recorded video in, a per-frame 3D eye position out, smoothed
  with `SignalTracker`. Output is a plot and a CSV, both checkable headless. This is where the
  scale problem gets solved or explicitly deferred.
- **2 - camera and screen model.** Nominal per-machine numbers, not a calibration step: focal
  length or FOV, screen width and height in metres, screen origin relative to the camera, one entry
  each for g4, g7 and the Pixel 7 Pro. Q7-Q9 decide where those live and how a machine is selected.
- **3 - off-axis projection.** Eye position plus screen rectangle to a projection matrix. Pure
  maths, so it gets real unit tests: centred eye reduces to a symmetric frustum, moving the eye
  shifts the frustum the right way, corners map where they should.
- **4 - render a scene through it.** The first phase that produces something worth looking at. The
  simplest scene that shows the effect, written to files, both behind swappable interfaces.
- **5 - close the loop, live.** Camera to tracker to renderer at interactive rate, on a machine
  with a display. The only phase that cannot be finished on this box: g7 is the target.

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
