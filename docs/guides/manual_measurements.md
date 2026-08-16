# Manual measurements

Three things that cannot be measured over ssh: a camera's focal length, whether two capture
resolutions share a field of view, and where the camera sits relative to the screen.

Each one ends the same way: paste a literal into `src/abyss/params/abyss_devices.py`, then add a
dated line to `plans/01_abyss_expansion/tracking.md`.

Why each method works, and what to do when a result looks wrong, is in
[`docs/library/camera_calibration.md`](../library/camera_calibration.md).

## What you need

- The machine holding the camera. Run everything there.
- A Kindle or a phone, for section 1.
- A ruler, for section 3.
- A tape measure, only for the 1b fallback.

Name registry entries after the **device**, not the machine.

## Before you start

Take a regression baseline on a clean tree:

```bash
uv run --no-sync python scripts/viewer_position.py
mkdir -p ~/abyss-baselines/<host>-<commit>
cp cache/viewer/*.csv ~/abyss-baselines/<host>-<commit>/
```

`cache/` is gitignored, so this is per machine. Never diff against another machine's CSVs.

## 1. Focal length

1. Make the board:

   ```bash
   uv run --no-sync python scripts/calibrate_camera.py board --device-preset kindle_pw11
   ```

   It prints the panel's implied diagonal. If that disagrees with the device's spec sheet, stop
   and fix the preset.

2. Copy the board to the device and open it full screen.

   Arrange the light so it comes from **behind the camera** and falls on the board. A window or
   lamp behind the board backlights it, the camera meters for the bright background, and the board
   goes black.

   Then set the reader's front light so the white squares look white, not glaring. Too dim and
   there is no contrast; too bright and the white squares saturate and bleed over the marker
   patterns, which is the more common failure once the room light is right. The script measures
   this and will tell you which way it went wrong.

   - **Kindle**: use `cache/calib/board_<preset>.pdf`. A Paperwhite will not open a raw PNG. Copy
     it into `documents/` over USB. If the reader offers "fit to page" or "actual size", either is
     fine, but do not let it crop margins.
   - **Phone**: use `cache/calib/board_<preset>.png`, in any viewer that does not crop.

   The PDF page is generated with exactly the panel's aspect ratio, which is the part that must be
   right. If the reader scales the page it costs nothing but the reported distances.

3. Capture, moving the board between every shot:

   ```bash
   uv run --no-sync python scripts/calibrate_camera.py capture --views 15
   ```

   Hold the board up **before** you start and keep it up for the whole run. The script says
   `HOLD THE BOARD UP NOW`, waits, then sweeps the camera's exposure to find the one that decodes
   the most markers, and refuses to continue if it still cannot see the board. So a lighting
   problem costs you one shot rather than fifteen, and you never have to guess an exposure. Pass
   `--exposure` yourself to skip the sweep.

   Then, for each view: fill a third to a half of the frame, tilt hard, 20 to 40 degrees, and vary
   roll, pitch and yaw. Move around the frame, not just the centre. Change the tilt during the
   countdown, hold still while it says `capturing`.

   Every frame is written to `cache/calib/views_<w>x<h>/`, the good ones as `view_*.png` and the
   failures as `reject_*.png`, so there is always something to look at when it goes wrong.

4. Solve:

   ```bash
   uv run --no-sync python scripts/calibrate_camera.py solve
   ```

5. Repeat steps 3 and 4 once more with fresh views. If the two focal lengths disagree by more than
   a percent, capture again with more varied tilt.

6. Record: paste the printed `CameraConfig` block into `CAMERAS`, replacing `<date>` in the
   provenance. Log the focal, the RMS, and the principal point and distortion the solve printed,
   since `CameraConfig` has nowhere to put those.

## 1b. Focal length without a second screen

Less accurate. Use only when there is nothing to show a board on.

1. Hold a sheet of A4 (297 mm tall) flat and square to the lens, near the centre of the frame.
2. Tape-measure the lens to the sheet, in metres.
3. Capture:

   ```bash
   uv run --no-sync python scripts/measure_focal.py capture --exposure 80
   ```

4. Open `cache/measure/target_<w>x<h>_grid.png` and read off the top and bottom edge rows of the
   sheet.
5. Solve, where `--pixels` is the difference between those two rows:

   ```bash
   uv run --no-sync python scripts/measure_focal.py solve \
       --pixels <bottom - top> --distance-m <measured> --size-mm 297 --at-height 720
   ```

6. Record as in step 6 above.

`focal_measured_at_height` is required: a focal length in pixels is meaningless without the
resolution it was measured at. Setting `fov_vertical_deg` instead is fine, but never from a
published spec, which quotes the diagonal.

## 2. Whether two resolutions share a field of view

**Only do this if something actually needs the second resolution.** On g7 nothing does: 1280x720
MJPG is the best mode the camera has, and capture pins it. `g7_webcam` is therefore measured at 720
and valid at 720 only, which its provenance says.

The trap to avoid is not the missing measurement, it is that `focal_px_for_height()` will rescale
to any height without knowing whether that is meaningful. Two modes of differing aspect ratio are
not necessarily the same field of view sampled at two densities. So pin the capture mode rather
than relying on the rescale, and remember a bare `cv.VideoCapture(0)` gets YUYV 640x480 silently.

If you do need it, run section 1 twice, once per resolution, and compare:

```bash
uv run --no-sync python scripts/calibrate_camera.py capture --width 1280 --height 720
uv run --no-sync python scripts/calibrate_camera.py solve   --width 1280 --height 720

uv run --no-sync python scripts/calibrate_camera.py capture --width 640 --height 480
uv run --no-sync python scripts/calibrate_camera.py solve   --width 640 --height 480
```

Scale the 720p focal by `480/720` and compare to the 480p focal. If they match, one measurement
covers both resolutions. If they do not, each resolution needs its own entry.

Record the answer either way.

## 3. Camera to screen offset

Convention: `+X` image right, `+Y` image **down**, `+Z` away from the camera, metres. A camera
above the panel puts the panel centre at positive Y.

1. Get the panel size from its own EDID:

   ```bash
   uv run --no-sync python scripts/read_edid.py
   ```

2. With a ruler, measure from the centre of the lens to the top edge of the **active** panel area,
   not the top of the case.

3. Record, with the Y offset being half the panel height plus that gap:

   ```python
   "<host>_internal": ScreenConfig(
       name="<host>_internal",
       width_m=<from EDID>,
       height_m=<from EDID>,
       camera_to_centre_m=(0.0, <height_m / 2 + gap>, 0.0),
       provenance="size from EDID; offset measured with a ruler, <date>",
   ),
   ```

Measure the gap on the machine in front of you. Do not copy another machine's offset, and drop the
word `PROVISIONAL` from the provenance only once the gap is really measured.
