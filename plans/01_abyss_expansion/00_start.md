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
| `utils.np_signal` (`create_left_triangle_filter`, `roll_append_smooth`) | smoothing a noisy per-frame scalar - the viewer position will jitter, and jitter in a projection matrix is visible immediately. Corrected on 2026-08-14: this row named `geometry.signal_tracker.SignalTracker`, which is a gesture *classifier* built on these primitives, not a smoother - see [`01_viewer_position.md`](01_viewer_position.md) |
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
  ANS: **Wrong axis - config is not per-machine.** An env var pattern is rejected outright: the
  Pixel will record on the phone while the code processing those frames runs on a different machine
  entirely, so "which host am I" says nothing about which camera produced the pixels. Config
  attaches to **which camera captured the input** and **which display shows the output**, and is
  passed to the components at construction as two objects: an **ingestion config** and a **render
  config**. Only g7 has a real camera for now, so everything else is speculative by design - which
  is fine, that is the mode we are in. The practical consequence is that the two configs travel
  with the data, not with the process.
- Q8: **Where do the device numbers live in code?** A plain dataclass per device in `abyss`, or
  pydantic models (which would bring the dependency back - it was just removed from pose-tools), or
  a data file (TOML/JSON) read at startup. Bound up with Q7.
  ANS: **Pydantic models**, so abyss takes the dependency back even though pose-tools just dropped
  it. Division of labour: **config models define the shape, params supply the values, pydantic does
  the validation.** A malformed device entry fails at construction with a readable error rather
  than producing a silently wrong frustum ten frames later.
- Q9: **Is the Pixel 7 Pro a config entry or a deployment target?** Recording clips on the phone and
  processing them on g4/g7 needs only its camera and screen numbers. Actually *running* abyss on the
  phone is a different project (Python does not deploy there without effort). Assumed to be the
  former; say if the phone is meant to run the loop itself.
  ANS: **A config entry, on both sides**: the phone records and the phone displays, but the code
  runs elsewhere and reaches it over a webapp. Future work, not scoped now. It is the reason Q7 came
  out the way it did - capture device, compute host and display device are three separate things.

Third batch, raised by folding the second in:

- Q10: **Where do the config values physically come from?** Q8 puts the shape in pydantic models and
  the values in params, but params is a Python singleton with literals today. Either the device
  entries are literals in `abyss.params` (simplest, no file IO, but editing config means editing
  code), or params loads a TOML/JSON of device entries and validates it through the models (a real
  file, which a webapp or a phone recording could also write later).
  ANS: **Python literals in params.** Keep the known, easy path: the values are code, reached the
  way every other path in `AbyssParams` already is. A custom loader gets introduced when something
  actually needs to write config from outside the repo (the webapp, most likely), not before.
- Q11: **Does the ingestion config describe the camera, or the input?** Camera-only means
  intrinsics and lens, with the clip path or camera index passed separately as a runtime argument.
  Input means one object carrying both. The first keeps "which camera" reusable across many clips
  shot on it; the second is one thing to pass around.
  ANS: **Split.** A camera config (intrinsics, the physical device) and a stream config (where the
  frames come from) are separate models, because the same camera feeds both a recorded file and a
  live capture. g7's webcam is exactly this case: one set of intrinsics, a clip today and a live
  stream later, and nothing about the camera should change when the source does.
- Q12: **Does phase 5 become the webapp?** Q3 declined FastAPI and picked files on disk, but Q9 says
  the phone is reached over a webapp. If the phone is the real demo target then "close the loop
  live" is a browser view served to it, not an OpenCV window on g7 - which is a different phase and
  reopens the scaffold question. Or g7 with a local window stays phase 5 and the webapp is a phase 6
  after it.
  ANS: **g7 with a local window is phase 5.** The phone is a future demo target, so the webapp comes
  after and does not reshape the phases now. FastAPI stays declined for the time being.

Fourth batch, raised by folding the third in:

- Q13: **Does the output side split the same way the input side does?** Q11 separates the camera
  (the physical device) from the stream (where frames come from). The mirror image would be a
  display config (screen size and position, the geometry the frustum needs) separate from a sink
  config (write PNGs here / open a window / serve it), and only the display config affects the
  maths. Symmetry is suggestive but not an argument on its own - the input split had a concrete case
  behind it (one webcam, clip today and live later).
  ANS: **Split**, and not for symmetry: the two belong to different stages. Screen geometry is an
  **input to the rendering** - it is what the off-axis frustum is built from, consumed before a
  pixel exists. The sink is what happens to the frame **after** rendering. A single object would
  hand the renderer information about PNG paths it has no business seeing.
- Q14: **Is the webapp a later phase here, or its own initiative?** Q12 puts it after phase 5.
  Serving frames to a phone, plus the phone's own capture, is a fairly self-contained body of work
  with its own dependency (FastAPI) and its own questions. The tracked-development convention says
  work that does not belong to the current scope becomes a sibling `02_...` folder rather than an
  extra phase.
  ANS: **Its own folder**, definitely. And the same question applies to phases already sketched
  here, so the whole list was reassessed - see below.

## What the answers add up to

Q2-Q5 all gave the same shape of answer, so it is a single principle rather than four coincidences:
**pick the cheapest implementation now, behind a seam that lets a better one drop in later.** Four
seams follow from that, and they are what the phases have to respect:

| Seam | Cheap version now | What replaces it |
| ---- | ----------------- | ---------------- |
| frame source | recorded clip | live webcam (g7), phone camera |
| device config | published nominal specs per device | measured or calibrated values |
| scene | the simplest thing that shows parallax | a real 3D scene, a captured one |
| output sink | frames written to disk | live window, browser, phone screen |

The risk to watch is over-abstracting: a seam is an interface with one implementation plus a second
one we can name. If the second cannot be named, it is not a seam yet.

## Config travels with the data, not the process

Q7 corrected the axis the Q2 answer was written on. That table of machines reads naturally as "look
up the host you are on", and that is wrong: the Pixel records frames while a different machine
processes them, so the host running the code knows nothing useful about the camera that produced the
pixels. Three roles that happen to coincide on g7 and nowhere else:

| Role | Described by | g4 | g7 | Pixel 7 Pro |
| ---- | ------------ | -- | -- | ----------- |
| capture device | camera config | none | webcam | front camera, later |
| compute host | nothing - it just runs | yes | yes | no |
| display device | screen config | none | screen | phone screen, later |

Q11 then split the ingestion side in two, because a camera and a source of frames are not the same
thing: g7's webcam has one set of intrinsics whether the frames arrive from a recorded clip today or
a live capture later, and nothing about the camera should change when the source does. Q13 split the
output side too, on a different argument: the screen geometry is an *input* to the rendering, since
the frustum is built from it, while the sink only acts on a frame that already exists. Four models,
each with its own reason to change:

| Model | Holds | Changes when |
| ----- | ----- | ------------ |
| camera config | intrinsics, FOV, resolution | you point a different physical camera at the problem |
| stream config | clip path or capture index, fps | you swap recorded for live, or pick another clip |
| screen config | size in metres, origin relative to the camera | you show it on a different screen |
| sink config | write PNGs here, open a window, serve it | you change what happens to a finished frame |

All four are constructed and handed to the components, never looked up from the environment. A clip
shot on the Pixel and processed on g4 carries the Pixel's camera config, a stream config pointing at
the file, a screen config for whichever screen the result is *meant* for, and a sink config that
writes PNGs. That last pair is the point of Q13: g4 renders for a screen it does not have.

They are pydantic models (Q8): the models fix the shape, params supply the values, validation is
pydantic's job. This brings `pydantic` back as an abyss dependency, days after pose-tools dropped
it. That is not an inconsistency to fix - pose-tools has no config surface, abyss now does.

It also reopens what the reboot deliberately closed: the params layer is minimal on purpose ("add
them when something needs them"). Device config is the first real something, and Q10 keeps it as
plain Python literals there - the known path, with a loader added only when something outside the
repo needs to write config.

## Scope reassessment (Q14)

Every sketched phase was checked against the same test: does it carry its own dependencies and its
own open questions, and could someone execute it without this initiative in their head? Three came
out as separate work, and the reason differs in each case.

| Sketched as | Verdict | Why |
| ----------- | ------- | --- |
| 0 - face landmarker | upstream prerequisite | it is code in another repo, with its own tracking there; abyss's share of it is a version pin |
| 1 - viewer position | stays | the core question of this initiative |
| 2 - camera and screen model | stays | the four config models, needed by 1 and 3 |
| 3 - off-axis projection | stays | the payoff; pure maths, unit-testable |
| 4 - render a scene | splits in two | see below |
| 5 - close the loop live | stays | the exit criterion, even though it runs on g7 |
| webapp / phone | own folder | own dependency (FastAPI), own questions, no other phase waits on it |

Phase 4 is the interesting one. Q4 asked for "the simplest scene that shows the effect", and that
minimal scene has to stay here: without something drawn, phases 3 and 5 cannot be seen to work at
all. A real renderer is different work - an OpenGL context, a GPU, model loading, possibly Gaussian
splatting - with nothing in this initiative waiting on it. So the minimal scene stays as phase 4,
and the renderer becomes its own folder.

Resulting layout:

- `01_abyss_expansion` (this) - phases 1-5, with phase 4 cut down to the minimal scene.
- `02_scene_rendering` - a real renderer behind the scene seam.
- `03_phone_webapp` - serving frames to a phone, and capture from it.
- `pose-tools/scratch_space/04_face_landmarker` - the upstream prerequisite, tracked in that repo
  and planned there in five phases. Phase 1 here waits on the tag it produces.

The originating rule holds in each case: the spin-offs can be executed on their own later, and
nothing here is blocked on them. The face landmarker is the one genuine cross-repo dependency, so it
stays visible in the phases table rather than being filed away.

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
