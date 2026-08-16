# Manual measurements

Three measurements that need a camera, a screen and a hand. They cannot be done over ssh.
Each ends in a literal pasted into `src/abyss/params/abyss_devices.py`.

Run each experiment on the machine holding the device, and name the registry entry after the
device, not the machine.

## 0. Before any of them

Regenerate the regression baseline, on the untouched branch:

```bash
uv run --no-sync python scripts/viewer_position.py
cp cache/viewer/*.csv ~/abyss-baselines/<host>-<commit>/
```

`cache/` is gitignored, so the baseline is per machine. Do not diff against another machine's
CSVs: MediaPipe CPU inference is not guaranteed bit-identical across hardware.

Filling in a focal length does not move these numbers, because the sample clips use the separate
`unknown_clip` entry, which stays unmeasured on purpose.

## 1. Focal length

Not calibration, no checkerboard. One object of known size at a known distance:

    f_px = apparent_size_px * distance_m / real_size_m

A percent or two of tape-measure error is fine. The per-identity head scale already corrected in
phase 1 is a 16% effect.

1. Hold a sheet of A4 (297 mm tall) flat, facing the lens square on, near the **centre** of the
   frame. The lens has visible barrel distortion at the edges.
2. Tape-measure lens to sheet, in metres.
3. Capture:

   ```bash
   uv run --no-sync python scripts/measure_focal.py capture --exposure 80
   ```

   Drop `--exposure` to use auto. Auto blows out a white sheet against a lit wall, and a blown
   sheet has no edge to measure. The script warns above 15% clipping.
4. Open `cache/measure/target_<w>x<h>_grid.png` and read the top and bottom edge rows off the
   scale. Their difference is the apparent size in pixels.
5. Solve:

   ```bash
   uv run --no-sync python scripts/measure_focal.py solve \
       --pixels <bottom - top> --distance-m <measured> --size-mm 297 --at-height 720
   ```

6. Paste the printed block into `CAMERAS`, replacing the provenance placeholder with the object,
   the distance and the date.

`focal_measured_at_height` is required. A focal length in pixels is only valid at the resolution
it was measured at, and `CameraConfig.focal_px_for_height()` rescales from it.

Setting `fov_vertical_deg` instead is also valid and resolution-independent, but never from a
published spec: datasheets quote the **diagonal**, which is about 20% wrong on 16:9. Setting both
is rejected at construction.

## 2. Which resolutions share a field of view

`focal_px_for_height()` rescales by frame height alone. That is only correct if the modes share a
vertical field of view. On a camera whose modes differ in aspect ratio, that is an assumption, not
a fact.

1. Point the camera at a fixed scene with something identifiable at the top and bottom edges.
2. Capture both modes:

   ```bash
   uv run --no-sync python scripts/measure_focal.py compare-modes
   ```

3. Compare `cache/measure/modes_*.png`:

   - 4:3 only wider than 16:9, same content top to bottom: shared vertical field of view, one
     measurement covers both.
   - 4:3 also reaches higher and lower: different vertical fields of view, each mode needs its own
     measurement and `focal_px_for_height()` must not be trusted across them.

Record the answer in the log either way.

## 3. Camera to screen offset

Phase 3's frustum consumes this directly.

Frame convention, which everything downstream depends on: `+X` image right, `+Y` image **down**,
`+Z` away from the camera (OpenCV), metres. A camera above the panel puts the panel centre at
positive Y.

1. Panel size from the display's own EDID, exact, no ruler:

   ```bash
   uv run --no-sync python scripts/read_edid.py
   ```

2. Measure the bezel gap with a ruler: centre of the lens to the top edge of the **active** panel
   area, not the top of the case.
3. The Y offset is half the panel height plus that gap. Add the entry:

   ```python
   "<host>_internal": ScreenConfig(
       name="<host>_internal",
       width_m=<from EDID>,
       height_m=<from EDID>,
       camera_to_centre_m=(0.0, <height_m / 2 + gap>, 0.0),
       provenance="size from EDID; offset measured with a ruler, <date>",
   ),
   ```

Do not copy another machine's offset. Half of g7's panel height is 96.5 mm, numerically identical
to g4's entire offset, which already includes g4's bezel guess. It would look right and be wrong
by the bezel.

Drop the word `PROVISIONAL` from the provenance only once the gap is actually measured, so phase 3
cannot consume a guess believing it was measured. `test_the_screen_offset_is_flagged_as_provisional`
pins this for `g4_internal`; a new unmeasured entry is not covered unless you add it there.
