# Camera calibration

Why the measurement scripts work the way they do, and how to read a result that looks wrong.
The steps themselves are in
[`docs/guides/manual_measurements.md`](../guides/manual_measurements.md).

## Why a single photo cannot give a focal length

Apparent size obeys `size_px = f * real_size / Z`. One head-on view of a known-size target is one
equation in two unknowns, so `f` and `Z` cannot be separated. That is the whole reason
`measure_focal.py` demands a tape measure: supplying `Z` is what makes the equation solvable.

Several views of the same planar target at **different orientations** remove the need. Each view
gives a homography, and the orthonormality of the rotation columns constrains the intrinsics,
independently of where the board was. Distance stops being an input and becomes an output. This is
Zhang's method, i.e. `cv.calibrateCamera`, and it is what `calibrate_camera.py` does.

The consequence that matters in practice: **tilt is required, not tolerated**. A stack of
fronto-parallel views is the degenerate case again no matter how many you take.

## Why the board's size does not affect the focal length

Scaling the object points scales the recovered translations and leaves the intrinsics unchanged.
A wrong millimetres-per-pixel therefore gives correct intrinsics and wrong distances. Pinned by
`test_intrinsics_do_not_depend_on_the_board_size`.

Two caveats. Invariance held to 0.001 px between scale 1.0 and 2.0 but drifted 0.4% at scale 0.5,
which is numerical conditioning from very small object coordinates rather than a real effect. And
the distances are wanted anyway, to sanity-check phase 1's depth. So supply the true size; a
screen gives it exactly from the pixel pitch, with no ruler.

## Why a screen rather than paper

The physical square size follows from the panel's pixel pitch exactly. A screen is also rigid and
flat where taped paper bows.

E-ink beats a phone: matte rather than glossy, and specular glare is the main way this fails. It
also has no backlight flicker to interact with exposure.

The trap is display scaling. The board is emitted at the panel's native resolution so that a
"fit to screen" viewer is the identity transform. Per the section above this corrupts only the
distances, not the focal, but it corrupts them silently.

## Getting the board onto a Kindle

A Paperwhite will not open a raw PNG, so `board` writes a PDF alongside it. The page is sized
`pixels / ppi` inches, which makes its aspect ratio exactly the panel's: 1236x1648 at 300 ppi
gives a 296.64 x 395.52 pt page, aspect 0.750000 either way.

Aspect is the only safety-critical property, and it is worth being precise about why.

- A **uniform** scale, from letterboxing or a reader that fits the page differently, changes the
  recovered distances and leaves the focal length untouched. Harmless for section 1.
- A **stretch**, from a page whose aspect differs from the panel's, distorts the board's shape.
  That corrupts the intrinsics themselves, and nothing downstream could detect it.

PDF readers scale uniformly rather than stretch, so matching the aspect makes the second case
impossible rather than merely unlikely. `test_pdf_page_has_exactly_the_panel_aspect` pins it.

Verified round trip: rasterizing the generated PDF at 300 dpi returns 1648x1237 against an
original canvas of 1648x1236, one pixel of rasteriser rounding, and detection finds all 48 corners.

If the distances matter and the reader is suspected of scaling, measure the displayed board's
width once with a ruler and compare against `square_px * squares_x / ppi`. That closes the metric
scale without affecting anything already measured.

## What the A4 fallback gets wrong

Beyond tape-measure error, a hand-held sheet is never exactly fronto-parallel. Tilt by θ makes it
appear `cos θ` shorter, biasing the focal **low**: 1.5% at 10 degrees, with nothing in the output
to reveal it. Zhang's method measures the tilt instead of assuming it away.

## Reading a result

| Symptom | Means |
| --- | --- |
| RMS above 1 px | blur, glare, or a moving board. Recapture, do not accept it |
| `fx/fy` far from 1.0 | non-square pixels, or too few well-spread views |
| `cx,cy` far from frame centre | real, if it survives a recapture. The model assumes centre |
| distances implausible | the pixel pitch is wrong. The focal is still fine |
| focal moves between runs | not enough tilt variety, the usual cause |

Two runs on fresh views agreeing within a percent is the cheapest evidence a number is real. One
run says the solver converged, not that it is right.

Detection is expected to be partial. The 7x9 board has 48 interior corners; measured, it reads 48
of 48 on the board image and 36 of 48 after a 4x downscale plus a 12 degree rotation. Views
reporting 25 to 45 corners are healthy. Consistently under 15 means the board is too small in
frame, too far, or glared out. Tolerating partial views is the reason for ChArUco over a plain
checkerboard.

## What the model cannot yet hold

`CameraConfig` stores one focal length and assumes the principal point is the frame centre. It has
no `fx`/`fy` split and no distortion coefficients. `solve` prints all of them; `focal_px` takes
`fy`, since the model is vertical-FOV based, and the rest goes in the provenance and the log.

If the principal point lands far from centre, or distortion is large, that is a measured reason to
extend the model. g7's webcam shows visible barrel distortion, so a non-trivial `k1` is expected
there.

## OpenCV notes

`calibrateCameraCharuco` was removed in OpenCV 5. The path is `CharucoDetector.detectBoard`, then
`board.matchImagePoints`, then `cv.calibrateCamera`.

Two cv2 stubs are narrower than the runtime: `matchImagePoints` is typed for a sequence of
matrices but accepts the single `(N, 2)` array `detectBoard` returns, and `getChessboardCorners`
is typed as a sequence but returns an ndarray. Both are exercised by the tests rather than
trusted.

A source build is not needed. The `opencv-contrib-python` wheel carries aruco, FFMPEG, V4L2, Qt5
and IPP; only GStreamer and the non-free algorithms are absent and neither is wanted here. A
hand-built install in the uv venv would also be silently reverted by `uv sync`, exactly as an
editable pose-tools install is.
