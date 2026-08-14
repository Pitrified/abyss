---
status: draft
---

# Abyss expansion - own functionality and the render layer

Spun off from [`../00_template_alignment/00_feature_inventory.md`](../00_template_alignment/00_feature_inventory.md),
which covered the reboot: tooling, and deduplication against `pose-tools`. That initiative is done
and merged - the repo builds, lints, type-checks and tests, and `src/abyss/` holds only `params/`
and `metaclasses/`. Everything about what abyss *does* lives here.

## What this is

The README states the goal, and it has not changed since the first commit:

1. Compute the position of the viewer.
2. Compute the position of the screen.
3. Render the scene the viewer sees on the screen.

That is head-coupled perspective, the "fish tank VR" effect: the screen behaves like a window
rather than a picture, because the projection is rebuilt from wherever the viewer's eye actually
is. The maths at the end is an **off-axis (asymmetric) frustum** - given the eye position in the
screen's coordinate frame, and the screen rectangle's corners, the projection matrix follows
directly. That part is well-trodden and not where the risk lives.

The risk is upstream of it: getting a *metric* eye position out of a webcam, and knowing where the
screen is relative to the camera. Steps 1 and 2 are the work; step 3 is the payoff.

## What already exists

From `pose-tools` (pinned at `v0.3.0`):

| Piece | Use here |
| ----- | -------- |
| `landmark.pose.PoseLandmarkerFrame` | body landmarks, including a rough head position |
| `landmark.landmark_array.LandmarkArray` | numpy landmark container with visibility masking |
| `geometry.signal_tracker.SignalTracker` | smoothing a noisy per-frame scalar - the viewer position will jitter, and jitter in a projection matrix is visible immediately |
| `geometry.homography` | feature-matched homography between two images, plus `perspective_transform` |
| `video.load` / `video.frame` | camera and file capture |
| `landmark.model_manager.ModelManager` | resolves `.task` model files |

What is **missing**, and matters:

- **No face landmarker.** pose-tools wraps pose and hand only. MediaPipe's `FaceLandmarker`
  supports `output_facial_transformation_matrixes`, i.e. a 4x4 head pose relative to the camera -
  which is much closer to "where is the viewer's eye" than body landmarks are. If we go that way,
  `pose_tools.landmark.face` is a prerequisite, and `ModelManager.MODEL_FILENAMES` needs a
  `face_landmarker` entry. See Q1.
- **No camera model.** Nothing anywhere knows the camera's focal length or field of view, and
  without it pixel measurements cannot become distances.
- **No renderer.** Nothing draws a 3D scene. `utils.plt` and `utils.cv` display images, that is all.

`geometry.landmark_geometry` used to be a re-export shim over two functions in `utils.mediapipe`.
It was deleted in pose-tools v0.3.0, along with the template config scaffold and the `load_env()`
import side effect - see `pose-tools/scratch_space/02_cleanup/`. Coordinate conversion lives in
`utils.mediapipe`; import it from there. No shims: nothing gets routed through a namespace to
justify the namespace.

## Shape of the problem

Three coordinate frames have to be related:

```text
camera frame  --(intrinsics: FOV / focal length)-->  metric 3D
   eye position measured here

screen frame  --(rigid transform: where the screen is relative to the camera)-->  camera frame
   projection is built here
```

The camera is typically clipped to the top of the screen, so the rigid transform is *nearly*
identity plus an offset - which tempts a hardcoded guess. That guess is exactly what makes the
effect feel wrong when the viewer moves off-centre, so it deserves an explicit answer (Q2).

Scale is the subtle part. A monocular camera cannot recover absolute distance without a reference:
either a known physical dimension (interpupillary distance is the usual choice, ~63 mm mean, but it
varies per person), or a calibration step where the viewer sits at a measured distance once.
MediaPipe's world landmarks are metric-ish but centred on the subject's hips, not the camera, so
they do not hand us this for free.

## Open questions

- Q1: **Face landmarker or pose landmarker for the eye position?** Face gives a head-pose matrix and
  eye landmarks directly, at the cost of adding `face.py` to pose-tools first. Pose is already
  wrapped but its head landmarks are coarse for this purpose. A third option is both: pose to find
  the person, face for the eye.
  ANS: **Face**, added to pose-tools as `landmark/face.py` if it lands cleanly on
  `BaseLandmarkerFrame`. Run both pose and face if wiring them together is easy; if the integration
  turns out to be awkward, drop to face alone rather than forcing it. So phase 0 exists.
- Q2: **How do we learn the camera intrinsics and the screen geometry?** Options: measure by hand
  and store in `AbyssParams` (simple, per-machine, no code); a one-off checkerboard calibration
  with OpenCV (accurate, more machinery); or assume a nominal FOV and screen size and accept the
  error. This decides whether a calibration phase exists at all.
  ANS: **Deferred, but pre-wired.** Start from config with nominal defaults, structured so real
  measured values replace them with no code change. Calibration itself is not a phase now. The
  machines are known, so their entries stay separate rather than collapsing into one global set of
  numbers:

  | Machine | Role here |
  | ------- | --------- |
  | `g4` | the headless CPU-only box: no camera, no display, development and tests only |
  | `g7` | 4 GB VRAM, webcam - the first machine that can actually run the loop live |
  | Pixel 7 Pro | phone: front camera plus screen, the eventual demo target |

  Published specs for these (sensor FOV, screen size in mm) can be looked up online and used as the
  defaults; nothing here needs a physical measurement to get moving.
- Q3: **What renders, and where?** Offline frames to a file, a live OpenCV window, or a browser
  view. This box is headless and CPU-only, so a live window cannot be developed or demoed here -
  whatever we pick, the loop has to be verifiable without a display. A browser target would reopen
  the FastAPI scaffold question (#15 in the reboot inventory), currently declined.
  ANS: **Static files.** Write frames out, enough to confirm each component works. Keep the output
  sink a swappable component so a live window or a browser view can replace it later without
  touching the pipeline. FastAPI stays declined.
- Q4: **What is "the scene"?** A 3D model rendered per viewpoint (needs a real renderer - moderngl,
  pyrender, or three.js in a browser), or layered 2D parallax (much cheaper, and enough to prove the
  effect), or a reprojection of captured content. The deleted `utils/data.py` knew a
  `~/data/3d_models` folder, which hints at the first - but nothing reads it now.
  ANS: **The simplest scene that shows the effect**, behind an interface with more than one
  implementation in mind, so experiments are cheap to swap in. No commitment to a renderer yet.
- Q5: **Real-time, or offline-first?** Offline - process a recorded clip, write annotated frames -
  is far easier to test and works headless. Real-time is the actual goal and imposes a latency
  budget on CPU-only inference. The phases below assume offline first; say if that is wrong.
  ANS: **Offline**, with the same modularity requirement: the frame source is a component, so a
  live camera replaces a recorded clip without rewriting the pipeline around it.
- Q6: **One eye or two?** A single cyclopean eye is the standard simplification and is what the
  README implies. Stereo would need a display capable of it, so this is likely a no - worth
  recording as a decision rather than an omission.
  ANS: **Single eye.** No stereo. Interpupillary distance is accepted as the scale reference, with
  the value living in config like the rest of Q2 - a per-person override costs nothing later.

Second batch, raised by folding the first in:

- Q7: **How does per-machine config get selected?** Q2 wants g4 / g7 / Pixel entries kept separate,
  which needs a way to say which machine is running. Options: an env var read at startup, hostname
  detection, or an explicit argument passed by the caller with a default. This is the first thing to
  reopen the params layer that the reboot deliberately kept minimal, so it is worth deciding rather
  than drifting into.
  ANS: ...
- Q8: **Where do the device numbers live in code?** A plain dataclass per device in `abyss`, or
  pydantic models (which would bring the dependency back - it was just removed from pose-tools), or
  a data file (TOML/JSON) read at startup. Bound up with Q7.
  ANS: ...
- Q9: **Is the Pixel 7 Pro a config entry or a deployment target?** Recording clips on the phone and
  processing them on g4/g7 needs only its camera and screen numbers. Actually *running* abyss on the
  phone is a different project (Python does not deploy there without effort). Assumed to be the
  former; say if the phone is meant to run the loop itself.
  ANS: ...

## What the answers add up to

Q2-Q5 all gave the same shape of answer, so it is a single principle rather than four coincidences:
**pick the cheapest implementation now, behind a seam that lets a better one drop in later.** Four
seams follow from that, and they are what the phases have to respect:

| Seam | Cheap version now | What replaces it |
| ---- | ----------------- | ---------------- |
| frame source | recorded clip | live webcam (g7), phone camera |
| device config | published nominal specs per machine | measured or calibrated values |
| scene | the simplest thing that shows parallax | a real 3D scene, a captured one |
| output sink | frames written to disk | live window, browser, phone screen |

The risk to watch is over-abstracting: a seam is an interface with one implementation plus a second
one we can name. If the second cannot be named, it is not a seam yet.

Config is now load-bearing, which reopens something the reboot deliberately closed. abyss's params
layer is minimal on purpose ("add them when something needs them"). Per-machine camera and screen
numbers are that something - see Q7.

## Tools suggested, not evaluated

Parked here so they are not lost, neither is committed:

- **OpenGL** (via `moderngl` or `pyglet` in Python) for the render side. This is the mainstream way
  to get an off-axis frustum: the projection matrix from phase 3 is exactly what a GL pipeline
  wants, so it fits the maths cleanly. Caveat for this box: it needs a GPU context, which makes it a
  g7 target, not a g4 one - though EGL/OSMesa offscreen rendering is worth checking before ruling
  headless out.
- **NeRF** (and by extension Gaussian splatting, which has largely displaced it for real-time work)
  for the scene side: capture a real place, then render novel views of it from the viewer's eye
  position. That is a genuine fit for Q4's "reprojection of captured content", and much heavier than
  anything above - training needs the GPU, and only splatting renders fast enough to matter. A
  candidate for the scene seam much later, not a starting point.

## The boundary with pose-tools

Unchanged, and it governs where new code goes: would `climbing-wire` want it? A face landmarker
wrapper, a camera-intrinsics model, landmark smoothing - general, so upstream. A screen model, an
off-axis frustum, a scene renderer - abyss's own.

`pose-tools` must never import `abyss`.

## Phases

Sketched in [`tracking.md`](tracking.md), all `draft` until the questions above are answered.
The sequencing principle is that each phase should be verifiable headless, on a recorded clip,
before anything depends on a live camera and a screen.
