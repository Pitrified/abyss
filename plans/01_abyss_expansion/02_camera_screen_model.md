---
status: planned
---

# Phase 2 - camera and screen model

## Overview

The four config models the answers to Q7-Q13 called for, replacing the five-number placeholder
phase 1 left behind. Context: [`00_start.md`](00_start.md), depends on
[`01_viewer_position.md`](01_viewer_position.md), and phase 3 consumes what this produces.

Phase 1 deliberately hardcoded its assumptions in one visible module so this phase could delete it.
That module is `src/abyss/viewer/camera.py`; when this phase is done it is gone.

## What the box turned out to have

Measured while planning, and it contradicts what the repo has been claiming:

- **g4 has a webcam.** `HP HD Camera` on `uvcvideo`, USB 04ca:7063, at `/dev/video0`. It cannot be
  opened only because the user is not in the `video` group - a one-line `usermod`, not a hardware
  limit. `.github/copilot-instructions.md` says this box has no camera and that
  `scripts/camera.py` is manual-only; the first half is wrong.
- **g4 has a panel, and it reports its own size.** EDID on `card1-eDP-1` gives **309 x 173 mm at
  1920x1080**, a 13.9 inch 16:9 screen at 158 ppi. So screen geometry on a Linux machine is
  machine-readable rather than a tape-measure job, and the same read works on g7.

Both change what this phase should build. Screen size wants a reader, not a hand-typed constant, and
"this box cannot test capture" is a permissions problem rather than a fact.

## The models

Four, per the decision in `00_start.md`, each with its own reason to change:

| Model | Holds | Source of values |
| ----- | ----- | ---------------- |
| `CameraConfig` | resolution, focal length or FOV, principal point, `mirrored` | published spec, or the phase 1 fallback law |
| `StreamConfig` | clip path or device index, fps, whether it loops | per run |
| `ScreenConfig` | width and height in metres, pose relative to the camera | EDID where available, measured otherwise |
| `SinkConfig` | where finished frames go: files, window, or nothing | per run |

Pydantic models (Q8), values as plain Python literals in `params` (Q10), passed to components at
construction rather than looked up (Q7). `pydantic` returns as an abyss dependency - it was removed
from pose-tools in v0.3.0, which stays right: pose-tools has no config surface and abyss does.

### Camera intrinsics without calibration

Q2 deferred calibration, so `CameraConfig` accepts **either** a focal length in pixels **or** a
field of view in degrees, and derives one from the other:

```
f_px = (height / 2) / tan(fov_vertical / 2)
```

That is the same relation phase 1 measured on MediaPipe's own assumption, confirmed at ratio 1.021
across a 1080-tall and a 1920-tall clip. When neither is given, the config falls back to MediaPipe's
63 degree assumption, which is exactly what phase 1 does today, so behaviour is unchanged until real
numbers arrive.

The consequence phase 1 measured stays true and belongs in the docstring: because the assumed focal
follows frame height, letterboxing or padding a frame silently rescales depth.

### Screen pose relative to the camera

The piece with no upstream source. A screen is a rectangle in the camera's frame, so
`ScreenConfig` carries its size plus the offset from the camera to the screen centre, in the same
OpenCV frame phase 1 established: `+X` right, `+Y` down, `+Z` away.

For a laptop this is easier than it looks: the camera is fixed in the lid bezel, so the screen does
not move relative to the camera when the lid tilts. One offset holds for all lid angles - roughly
`(0, +h/2 + bezel, 0)`, the camera sitting centred above the panel. A separate webcam on top of an
external monitor has no such guarantee and needs its own numbers per setup.

Rotation is deliberately **not** modelled yet: for a laptop it is the identity, and phase 3 only
needs the four corners. If a phone in portrait or a tilted external monitor appears, the corners can
carry it without a rotation field.

## Plan

- `src/abyss/config/` with one module per model, plus the device registry in `params`. Registry
  entries are named by **device**, not by machine - `g4_internal`, `g7_webcam`, `pixel7pro_front` -
  because Q7 established that config travels with the data, not the process.
- `ScreenConfig.from_edid()` scanning `/sys/class/drm/*/edid`. On this box that is five nodes of
  which **four are zero bytes** - the disconnected HDMI and DisplayPort outputs - so the job is
  finding the connected panel, not reading a known path: skip empty files, check the
  `00 FF FF FF FF FF FF 00` header, then read the **detailed timing descriptor** rather than the
  centimetre-rounded basic block, which gives 309x173 mm instead of 31x17 cm. A named error when
  nothing is readable, which is every non-Linux target and the Pixel.
  The committed fixture is those 128 bytes: manufacturer `CMN`, product `0x14d7`, serial field zero.
  Checked before proposing it, since committing hardware identifiers is worth a glance - there is no
  personal data in the block.
- Delete `viewer/camera.py` and move its consumers onto `CameraConfig`. The five values map across
  directly; `ipd_m` does not - see Q15.
- Validation is what pydantic is here for: positive dimensions, FOV strictly between 0 and 180,
  resolution positive, sink path present when the sink writes files. A malformed entry fails at
  construction with a readable error rather than producing a silently wrong frustum ten frames on.
- Tests mirror `src/abyss/config/`: construction, the FOV/focal round trip at several heights, the
  MediaPipe fallback matching phase 1's numbers, EDID parsing against a committed byte fixture (the
  128 bytes from this box, which contain no personal data), and each validation error firing.

## Out of scope

- Checkerboard calibration. Q2 deferred it and nothing here reopens it. The FOV-from-spec route is
  the cheap version behind the same seam, and calibration replaces it later without touching callers.
- The rendering side of `SinkConfig`. The model exists so phase 4 has somewhere to put its choice;
  actually opening a window is phase 5.
- Fixing the `video` group so g4 can capture. That is a system change needing sudo and a box-level
  note, and it is not needed for anything here - see Q17.
- Rotation between screen and camera, as above.

## Open questions

Numbering continues from `00_start.md`.

- Q15: **Where does the viewer's interpupillary distance live?** It is a property of the *person*,
  not of any of the four devices, but phase 1 parked it on the camera placeholder because that was
  the only config that existed. Options: a fifth `ViewerConfig` model holding it alone; keep it in
  `viewer/` as a plain constant with the estimator; or leave it on `CameraConfig` and accept the
  mislabel. The estimator already derives it per session, so this is about where the *default* and
  any per-person override live.
  ANS: ...
- Q16: **Do the device entries carry measured values or published ones?** g4's screen can be read
  from EDID exactly. g4's webcam FOV is not published anywhere reliable, and the Pixel's front
  camera FOV is a published number of uncertain accuracy. Options: accept published specs and record
  their provenance per entry; or leave FOV unset so the MediaPipe fallback applies until someone
  measures. The second is honest but leaves the depth scale wrong by whatever the FOV error is.
  ANS: ...
- Q17: **Should the `video` group be fixed on g4?** One `usermod -aG video pmn` plus a re-login makes
  this box able to capture, which would let phases 2-4 be developed against a live camera instead of
  clips, and would make the `f_real` question measurable here rather than on g7. It needs sudo, so
  it needs a box-level plan note first.
  ANS: ...

## Done when

- `viewer/camera.py` is deleted and nothing imports it.
- The four models exist, validate, and are constructed from literals in `params` for at least
  `g4_internal` and one clip-based stream.
- `ScreenConfig.from_edid` returns 309x173 mm on this box, and the same numbers come back from the
  committed fixture without touching `/sys`.
- `scripts/viewer_position.py` runs unchanged in behaviour: the same three clips produce the same
  eye positions to within floating point, since the fallback reproduces phase 1's assumption.
- `make check` is green, and the suite still passes with no clip, no model, and no `/sys/class/drm`.
