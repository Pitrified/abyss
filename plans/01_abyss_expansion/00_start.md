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

From `pose-tools` (pinned at `v0.2.1`):

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

`geometry.landmark_geometry` is currently a re-export shim over two functions in `utils.mediapipe`;
treat it as a namespace to grow into, not as existing capability.

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
  ANS: ...
- Q2: **How do we learn the camera intrinsics and the screen geometry?** Options: measure by hand
  and store in `AbyssParams` (simple, per-machine, no code); a one-off checkerboard calibration
  with OpenCV (accurate, more machinery); or assume a nominal FOV and screen size and accept the
  error. This decides whether a calibration phase exists at all.
  ANS: ...
- Q3: **What renders, and where?** Offline frames to a file, a live OpenCV window, or a browser
  view. This box is headless and CPU-only, so a live window cannot be developed or demoed here -
  whatever we pick, the loop has to be verifiable without a display. A browser target would reopen
  the FastAPI scaffold question (#15 in the reboot inventory), currently declined.
  ANS: ...
- Q4: **What is "the scene"?** A 3D model rendered per viewpoint (needs a real renderer - moderngl,
  pyrender, or three.js in a browser), or layered 2D parallax (much cheaper, and enough to prove the
  effect), or a reprojection of captured content. The deleted `utils/data.py` knew a
  `~/data/3d_models` folder, which hints at the first - but nothing reads it now.
  ANS: ...
- Q5: **Real-time, or offline-first?** Offline - process a recorded clip, write annotated frames -
  is far easier to test and works headless. Real-time is the actual goal and imposes a latency
  budget on CPU-only inference. The phases below assume offline first; say if that is wrong.
  ANS: ...
- Q6: **One eye or two?** A single cyclopean eye is the standard simplification and is what the
  README implies. Stereo would need a display capable of it, so this is likely a no - worth
  recording as a decision rather than an omission.
  ANS: ...

## The boundary with pose-tools

Unchanged, and it governs where new code goes: would `climbing-wire` want it? A face landmarker
wrapper, a camera-intrinsics model, landmark smoothing - general, so upstream. A screen model, an
off-axis frustum, a scene renderer - abyss's own.

`pose-tools` must never import `abyss`.

## Phases

Sketched in [`tracking.md`](tracking.md), all `draft` until the questions above are answered.
The sequencing principle is that each phase should be verifiable headless, on a recorded clip,
before anything depends on a live camera and a screen.
