# The pose-tools boundary

`abyss` used to carry its own copies of the pose, video and utility code. In 2026 those copies were
deleted and `pose-tools` became the single implementation. This note records where the line sits and
why, so the duplication does not grow back.

## Why the split exists

The same MediaPipe wrangling was written three times, in `abyss`, `climbing-wire` and `holo-table`.
`pose-tools` extracts it once. Three copies of a landmark converter is three places for a MediaPipe
release to break, and only one of them gets fixed.

That is not hypothetical here. MediaPipe 1.0.0 removed `mediapipe.python.solutions.*` and
`mediapipe.framework.formats.landmark_pb2`. abyss's own `utils/mediapipe.py` and
`landmarker/drawing.py` imported both, so they were dead code by the time they were deleted.
pose-tools had already moved to `mediapipe.tasks.python.vision`, and kept working.

## Where things live

| Concern                                  | Module                                             |
| ---------------------------------------- | -------------------------------------------------- |
| Pose and hand landmarkers                | `pose_tools.landmark.pose`, `.hand`, `.base`       |
| MediaPipe `.task` model resolution       | `pose_tools.landmark.model_manager.ModelManager`   |
| Drawing landmarks onto a frame           | `pose_tools.landmark.drawing`                      |
| Landmark arrays and visibility masking   | `pose_tools.landmark.landmark_array`               |
| Landmark distances                       | `pose_tools.landmark.distance`                     |
| Frames, video iteration and loading      | `pose_tools.video.frame`, `.load`                  |
| Homography and landmark geometry         | `pose_tools.geometry.homography`, `.landmark_geometry` |
| Smoothing noisy per-frame signals        | `pose_tools.geometry.signal_tracker`               |
| OpenCV and matplotlib display helpers    | `pose_tools.utils.cv`, `.plt`                      |
| MediaPipe result conversions             | `pose_tools.utils.mediapipe`                       |

What is left in `abyss` is `params/` and `metaclasses/`, plus whatever the viewer, screen and
render work adds on top.

## Deciding where new code goes

Would `climbing-wire` want it?

- **Yes** - it is general. Add it to `pose-tools`, cut a tag there, bump the pin in
  `pyproject.toml`. Working on both at once is what `make dev-pose-tools` is for.
- **No, it is about viewers, screens or rendering a scene** - it belongs in `abyss`.

There is no third option where a general utility lives here "for now". That is how the duplication
started the first time.

## What replaced the deleted code

| Deleted from abyss                      | Use instead                                             |
| --------------------------------------- | -------------------------------------------------------- |
| `utils/data.py:get_resource("pose_fol")`| `get_abyss_paths().pose_fol`                             |
| `utils/data.py:get_resource("pose_landmarker.task")` | `ModelManager().get_model_path("pose_landmarker")` |
| `landmarker/drawing.py:draw_landmarks`  | `pose_tools.landmark.drawing.draw_pose_landmarks`        |
| `landmarker/pose.py:PoseLandmarkerFrame`| `pose_tools.landmark.pose.PoseLandmarkerFrame`           |
| `utils/mediapipe.py:list_land_to_landlist` | nothing - it fed an API MediaPipe has removed         |
| everything else in `utils/`, `video/`   | the same name under `pose_tools`                         |

The old `get_resource()` was a single function with a `Literal` of five keys and an implicit `None`
return for unknown input. Of those five, only two had a live caller; they are the two rows above.
