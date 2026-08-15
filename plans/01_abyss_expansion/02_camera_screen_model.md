---
status: done
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

- **g4 has a webcam, and it is useless.** `HP HD Camera` on `uvcvideo`, USB 04ca:7063, at
  `/dev/video0`, blocked only because the user is not in the `video` group. So
  `.github/copilot-instructions.md` claiming this box has no camera is wrong on the hardware - but
  g4 is reached over ssh, so there is nobody in front of it and a live frame would show an empty
  room. Recorded because the docs are inaccurate, not because it unblocks anything (Q17). The
  camera that matters is g7's, which has a person sitting at it.
- **g4 has a panel, and it reports its own size.** EDID on `card1-eDP-1` gives **309 x 173 mm at
  1920x1080**, a 13.9 inch 16:9 screen at 158 ppi. So screen geometry on a Linux machine is
  machine-readable rather than a tape-measure job, and the same read works on g7.

So screen size is measured once by asking the panel rather than by holding a ruler to it, and the
live-capture work belongs on g7 for reasons of furniture rather than software.

## The models

Three devices plus the viewer, each with its own reason to change:

| Model | Holds | Source of values |
| ----- | ----- | ---------------- |
| `CameraConfig` | field of view or a measured focal length, `mirrored`, provenance | measurement, or the phase 1 fallback law |
| `StreamConfig` | clip path or device index | per run |
| `ScreenConfig` | width and height in metres, pose relative to the camera | EDID where available, measured otherwise |
| `ViewerConfig` | interpupillary distance, and later any per-person overrides | Q15: measured per person, estimated per session |

`ViewerConfig` is the extra model Q15 called for. Q15's answer said five, counting `SinkConfig`;
this phase builds four, because the sink went to phase 4 for reasons that have nothing to do with
Q15. A viewer is not a device, and the interpupillary
distance phase 1 parked on the camera placeholder was always mislabelled there. **Selecting** the
right viewer entry is explicitly deferred: there is one person today, and the session estimator
already derives their scale, so a registry keyed by person waits until a second person exists.

`SinkConfig` is **deferred to phase 4**, where its first caller appears. The four-way split stays
the architecture and the Q13 stage-boundary argument stands; what is deferred is the model, not the
decision. Nothing constructs or reads a sink here, configs are passed to components individually
rather than bundled, so adding it later widens no signature that exists. The reason to wait is that
its fields are a guess until phase 4 knows what it draws - inventing kind, output directory and
filename pattern now means params literals, validation rules and tests pinned to a shape that gets
rewritten, which is worse than absent because it looks authoritative. Same reason `StreamConfig`
loses `loops` (nothing loops) and `fps` (read from the capture today, so a config value would only
contradict it).

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

Two rules follow from that, and the plan needs both stated or the implementation gets them wrong:

- **FOV is the canonical stored value, `f_px` is derived from the actual frame height at runtime**,
  exactly as phase 1 does. A focal length in pixels is only valid at the resolution it was measured
  at: 1100 px at 1280x720 is 1650 px at 1920x1080 on the same lens. This is not hypothetical here -
  `face02_portrait.mp4` is 1080x1920 while the other two clips are 1920x1080, so no single literal
  `f_px` can serve one camera across both orientations. When a measured focal is stored, it carries
  the resolution it was measured at and is rescaled by the height ratio.
- **The FOV field is `fov_vertical_deg`, and published specs are diagonal.** Phone and webcam
  datasheets quote the diagonal figure almost without exception. Feeding one straight into
  `(H/2)/tan(fov/2)` is wrong by roughly the aspect factor, about 20% on 16:9 - an order worse than
  the per-identity scale error this phase exists to beat. A spec number is converted before entry,
  never pasted in raw, and the field name says which axis it wants.

`principal_point` stays a derived property at the frame centre rather than a config value. Nothing
has measured a real one, and Q2 deferred the calibration that would.

### Measuring a focal length

Q16 chose measurement over published specs, and the cheap version is enough. A single object of
known size at a known distance gives the focal length directly:

```
f_px = apparent_size_px * distance_m / real_size_m
```

A ruler or a sheet of paper at a metre, one frame, one measurement. It yields the one number
`CameraConfig` consumes - no checkerboard, no lens distortion model, no OpenCV calibration run.
Accuracy is limited by how well the distance is measured, which is a tape-measure problem good to a
percent or two, well inside the 16% per-identity scale error phase 1 measured and corrects (66.9 mm
against 57.7 mm of implied interpupillary distance between the two subjects; the 13% in the phase 1
plan was the pre-implementation estimate).

It is a manual step on the machine holding the camera, so it happens on g7 and the result is typed
into that device entry with a note on how it was obtained. Published specs remain acceptable as a
provisional entry, marked as provisional, so nothing blocks waiting for a measuring session.

Which means, said plainly rather than left to be discovered: **this phase ships the seam and the
procedure, not the numbers.** Nobody can execute the measurement from here - Q17 established that
g4's camera has no viewer in front of it - so every camera entry starts with its FOV unset and the
MediaPipe fallback applies everywhere, which is exactly what phase 1 does today. The first real
focal length arrives when someone sits at g7.

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
- **EDID is a one-off measurement, not a runtime dependency.** `scripts/read_edid.py` prints the
  connected panel's size; it is run once per machine and the answer is typed into that machine's
  `ScreenConfig` literal with a note on where it came from, exactly as Q10 says config values live.
  No `from_edid()` in the package, no committed byte fixture, no parser tests, and abyss never reads
  `/sys` at runtime. The script is small: EDID blocks are exposed one per **connector**, five on this
  box, so it finds the connected one, checks the `00 FF FF FF FF FF FF 00` header, and reads the
  **detailed timing descriptor** at byte 54 rather than the centimetre-rounded basic block, which is
  the difference between 309x173 mm and 31x17 cm. Two traps worth writing down in the script, since
  it will be run again on g7: every sysfs `edid` node reports `st_size` 0 including the live one, so
  filtering on file size discards the panel - read the bytes, or better read the neighbouring
  `status` file, which says `connected` for exactly one node.
- **An entry for the clips**, without which the exit criterion below cannot hold. The registry names
  devices, and none of them shot `face01` / `face02_portrait` / `face03_zoom` - their provenance is
  a README in `~/data/pose/`. So there is an `unknown_clip` entry with FOV unset, which is what puts
  those three clips on phase 1's fallback path.
- Delete `viewer/camera.py` and move its consumers onto `CameraConfig`. Four of the five values map
  across directly; `ipd_m` moves to `ViewerConfig` (Q15), which changes
  `estimate_head_scale(samples, camera)` to take the viewer as a third argument. The call sites are
  countable: `viewer/eye_position.py`, `scripts/viewer_position.py` (whose `--ipd-mm` default reads
  `DEFAULT_IPD_M`) and `tests/viewer/test_eye_position.py`. `tests/viewer/test_camera.py` is deleted
  rather than ported; its content becomes `tests/config/test_camera.py`.
- **Measure the bezel offset** while a screen is in reach, since nothing else will: the camera-to-
  screen-centre offset is the one number with no upstream source, and phase 3's frustum consumes it
  directly. One ruler reading on g4's lid gives it, and g4's *screen* is a legitimate entry even
  though its camera is not.
- Validation is what pydantic is here for: positive dimensions, FOV strictly between 0 and 180,
  positive resolution, positive interpupillary distance. A malformed entry fails at construction with
  a readable error rather than producing a silently wrong frustum ten frames on.
- Tests mirror `src/abyss/config/`: construction, the FOV/focal round trip at several heights and at
  both clip orientations, the MediaPipe fallback matching phase 1's numbers, `mirrored` flipping X
  and nothing else, and each validation error firing. No test reads `/sys`, because nothing in the
  package does.

## Out of scope

- Checkerboard calibration. Q2 deferred it and nothing here reopens it. The FOV-from-spec route is
  the cheap version behind the same seam, and calibration replaces it later without touching callers.
- `SinkConfig` entirely, model included, until phase 4 has a caller for it. The four-way split and
  the Q13 argument behind it stand; only the code waits.
- An EDID parser in the package. The read is a one-off script whose output becomes a literal.
- Unblocking the `video` group. Worth doing on **g7**, where the camera has a viewer in front of
  it, and it is a privileged system change there needing its own box-level note. On g4 it would
  change nothing useful (Q17).
- Choosing which `ViewerConfig` applies to whoever is in frame. One viewer today; deferred by Q15.
- Rotation between screen and camera, as above.

## Open questions

Numbering continues from `00_start.md`.

- Q15: **Where does the viewer's interpupillary distance live?** It is a property of the *person*,
  not of any of the four devices, but phase 1 parked it on the camera placeholder because that was
  the only config that existed. Options: a fifth `ViewerConfig` model holding it alone; keep it in
  `viewer/` as a plain constant with the estimator; or leave it on `CameraConfig` and accept the
  mislabel. The estimator already derives it per session, so this is about where the *default* and
  any per-person override live.
  ANS: **A new config**, `ViewerConfig`, so five models rather than four. Matching a config to a
  particular person is deferred: there is one viewer today, the estimator derives their scale per
  session, and picking the right entry for whoever sits down is a problem for when a second person
  exists.
- Q16: **Do the device entries carry measured values or published ones?** g4's screen can be read
  from EDID exactly. g4's webcam FOV is not published anywhere reliable, and the Pixel's front
  camera FOV is a published number of uncertain accuracy. Options: accept published specs and record
  their provenance per entry; or leave FOV unset so the MediaPipe fallback applies until someone
  measures. The second is honest but leaves the depth scale wrong by whatever the FOV error is.
  ANS: **Measure what is needed.** Not full calibration - one known object at one known distance is
  enough for a focal length, and that is all any of this consumes. Procedure in "Measuring a focal
  length" below. Published specs stay acceptable as a starting entry, marked as such.
- Q17: **Should the `video` group be fixed on g4?** One `usermod -aG video pmn` plus a re-login makes
  this box able to capture, which would let phases 2-4 be developed against a live camera instead of
  clips, and would make the `f_real` question measurable here rather than on g7. It needs sudo, so
  it needs a box-level plan note first.
  ANS: **Worth doing, but it does not unblock anything here.** g4 is reached over ssh - nobody sits
  in front of it, so its camera has no viewer to track and a live frame from it would show an empty
  room. The finding stands technically and is operationally useless. The group fix matters on **g7**,
  which has both a camera and a person, and that is where live capture gets developed.

## Done when

- `viewer/camera.py` is deleted and nothing imports it. Today `grep -rn "viewer.camera"` finds four
  files; afterwards it finds none.
- The four models exist, validate, and are constructed from literals in `params` for `g4_screen`,
  `unknown_clip`, and one clip-based stream.
- `g4_screen` carries 0.309 x 0.173 m - metres, since phase 1's unit decision applies here too and
  the millimetres stay inside the script - plus a camera offset that was measured rather than
  guessed, with its provenance in a comment.
- `scripts/viewer_position.py` reproduces phase 1 **exactly**, checked by diffing the new CSVs
  against the ones already under `cache_fol/viewer/` rather than by eye. This holds only because
  `unknown_clip` leaves FOV unset, resolution comes from the frames rather than from the config, and
  `ViewerConfig.ipd_m` defaults to the same 0.063 that `DEFAULT_IPD_M` did. Worth recording why the
  first of those matters less than it looks: with the head-scale estimator active, `head_scale` is
  proportional to the focal and `depth = depth_m * head_scale` with it, so the focal cancels out of
  `x = (u - cx) * depth / focal` exactly. A wrong focal is a pure depth error, which is both why the
  Q16 measurement is worth doing and why an unset entry reproduces phase 1 bit for bit.
- `make check` is green, and the suite still passes with no clip and no model present.
