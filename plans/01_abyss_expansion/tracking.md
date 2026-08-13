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

## Phases

Draft until Q1-Q6 in `00_start.md` are answered - the answers decide whether phase 0 exists at all
and what phase 4 even means.

| #  | Phase                          | Plan | Status |
| -- | ------------------------------ | ---- | ------ |
| 0  | face landmarker in pose-tools  | -    | draft, conditional on Q1 |
| 1  | viewer position from a clip    | -    | draft |
| 2  | camera and screen model        | -    | draft, shaped by Q2 |
| 3  | off-axis projection            | -    | draft |
| 4  | render a scene through it      | -    | draft, shaped by Q3/Q4 |
| 5  | close the loop, live           | -    | draft, shaped by Q5 |

Status values: draft / planned / in progress / done / superseded / discarded.

Sketch of each, to be replaced by real sub-plans once the questions land:

- **0 - face landmarker in pose-tools.** `landmark/face.py` over `BaseLandmarkerFrame`, plus a
  `face_landmarker` entry in `ModelManager.MODEL_FILENAMES`, released as a tag and pinned here.
  Only exists if Q1 says face.
- **1 - viewer position from a clip.** Recorded video in, a per-frame 3D eye position out, smoothed
  with `SignalTracker`. Output is a plot and a CSV, both checkable headless. This is where the
  scale problem gets solved or explicitly deferred.
- **2 - camera and screen model.** Whatever Q2 picks, ending in numbers that live in
  `AbyssParams`: focal length or FOV, screen width and height in metres, screen origin relative to
  the camera.
- **3 - off-axis projection.** Eye position plus screen rectangle to a projection matrix. Pure
  maths, so it gets real unit tests: centred eye reduces to a symmetric frustum, moving the eye
  shifts the frustum the right way, corners map where they should.
- **4 - render a scene through it.** The first phase that produces something worth looking at.
  Scope depends entirely on Q4 - parallax layers and a 3D model are very different amounts of work.
- **5 - close the loop, live.** Camera to tracker to renderer at interactive rate, on a machine
  with a display. The only phase that cannot be finished on this box.

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
