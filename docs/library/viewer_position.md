# From pixels to an eye position

Steps 1 to 4 of the chain in [`geometry_overview.md`](geometry_overview.md): how a 1280x720 BGR
frame becomes a point in metres in the camera frame.
Code in `src/abyss/viewer/eye_position.py`, `src/abyss/viewer/smoothing.py` and
`src/abyss/config/camera.py`.

## The pinhole model

Everything here rests on one equation. A point $P = (X, Y, Z)$ in the camera frame, with $Z > 0$
in front of the lens, lands at the pixel

$$u = c_x + f\,\frac{X}{Z}, \qquad v = c_y + f\,\frac{Y}{Z}$$

where $f$ is the focal length **in pixels** and $(c_x, c_y)$ is the principal point, taken here as
the frame centre. `FrameGeometry` supplies both.

The inverse is what `eye_position_m` computes, and it is the entire step 3 once a depth is known:

$$X = (u - c_x)\,\frac{Z}{f}, \qquad Y = (v - c_y)\,\frac{Z}{f}$$

Two pixels and a depth give three metres. The hard part of this page is where $Z$ comes from.

### Focal length is a property of the frame, not of the camera

$f$ in pixels is the physical focal length divided by the sensor's pixel pitch, so it scales with
how finely the sensor is sampled:

$$f' = f \cdot \frac{H'}{H}$$

for a frame resampled from height $H$ to $H'$. The **field of view** is what stays fixed:

$$f = \frac{H/2}{\tan(\theta_v/2)}, \qquad \theta_v = 2\arctan\!\left(\frac{H}{2f}\right)$$

This is why `CameraConfig` stores a field of view, or a focal length **with the height it was
measured at**, and refuses a focal length without one (`MissingFocalHeightError`).
945 px on g7 at 720 high is 1417.5 px at 1080 on the same lens, and the two are the same camera.

The scaling is only meaningful within one aspect ratio, since a different aspect is a different crop
of the sensor rather than a resampling of the same one. The registry entry for `g7_webcam` records
which of the camera's eight MJPG modes are exactly 16:9 for this reason.

### MediaPipe's assumed intrinsics, and the ratio that follows

The face landmarker is given no intrinsics and assumes a vertical field of view of 63 degrees:

$$f_{\text{mp}} = \frac{H/2}{\tan(31.5^\circ)} = 1.6319 \cdot \frac{H}{2}$$

At $H = 720$ that is **587.5 px**. The measured focal on g7 is **945 px** at the same height, a
ratio of

$$\frac{f_{\text{true}}}{f_{\text{mp}}} = \frac{945}{587.5} = 1.609$$

The real camera is much longer than MediaPipe assumes: 41.7 degrees against 63.
Every metric quantity MediaPipe reports carries that factor, and the head scale calibration below
is where it is removed. It is not removed by correcting MediaPipe, which takes no intrinsics.

A related trap, measured rather than reasoned: MediaPipe's depth depends on the **frame height**,
because that is what its assumed focal length is computed from. Padding a 1920x1080 frame to
1920x1920 with identical content moved the reported depth from 0.507 m to 0.881 m.
Letterboxing a frame is not a no-op on this pipeline.

## Where the depth comes from

Two candidates, and the choice between them is the substantive design decision on this page.

### Why not apparent interpupillary distance

The obvious estimator. If the viewer's real interpupillary distance is $I$ and it subtends
$i$ pixels, then by the pinhole equation applied to a segment perpendicular to the optical axis:

$$Z = f\,\frac{I}{i}$$

One measurement, no model, no assumptions beyond the pinhole.
It fails for a specific reason: the interocular segment is only perpendicular to the axis when the
viewer faces the camera. Under a yaw of $\psi$ the segment foreshortens,

$$i(\psi) \approx i(0)\cos\psi$$

so the estimator returns $Z/\cos\psi$ and reports the viewer as **further away the more they turn
their head**. Measured over the sample clips, apparent interpupillary distance correlates $-0.76$
with absolute yaw. At 30 degrees of yaw the error is already 15%.

That is exactly the wrong failure mode: turning to look at something on the screen is normal
behaviour, and it would make the scene lurch backwards every time.

### What is used instead

MediaPipe fits a 3D morphable face model to the landmarks and returns a **facial transformation
matrix**: a rigid transform $[R \mid t]$ placing its canonical head model in the camera's frame.
Solving for the pose of a known 3D shape from its 2D projection is the perspective-n-point problem,
and its solution uses the whole set of landmarks and the model's own depth structure rather than one
frontal segment. It degrades under yaw far more gracefully because yaw is a parameter it solves for
rather than an error it absorbs.

What the matrix locates is the **model origin**, not the eye. The eye sits above and in front of it,
so the offset is applied inside the model frame and rotated with it:

$$e_{\text{cam}} = R\,e_{\text{model}} + t, \qquad e_{\text{model}} = (0,\ 2.5,\ 3.0)\ \text{cm}$$

Applying the offset *before* the rotation is the point. Adding a constant 2.7 cm to the depth
afterwards would be right only face-on and would swing by about a centimetre with yaw, which is the
same class of error the interpupillary estimator has and the reason for using the matrix at all.

$e_{\text{model}}$ was fitted rather than looked up: reprojecting the model pose onto the measured
iris pixels over `face01.mp4` gives 2.5 cm up and 3.0 cm forward with a residual of 2.5 px.

MediaPipe's model frame looks down $-Z$, so the depth is the negated Z component,
and its centimetres are converted to metres once, here.

### Yaw, for gating

The head yaw is read off the same rotation matrix:

$$\psi = \mathrm{atan2}(-R_{20},\ R_{00})$$

Exact under yaw composed with pitch, since neither leaves the first column's $X$ and $Z$ entries
alone, and approximate under roll, which is small for a seated viewer.
It is used only as a gate on the calibration below, never in the position itself.

## The head scale calibration

MediaPipe's canonical model is identity-dependent: it fits a mesh whose absolute size varies with
the subject. Measured across the two people in the sample clips, the implied interpupillary distance
was 66.9 mm for one and 57.7 mm for the other, 16% apart.
An identity-dependent model produces an identity-dependent depth, so the metric output is correct up
to one unknown scalar per person.

`estimate_head_scale` recovers that scalar. Over front-facing samples only
($|\psi| \le 10^\circ$), it computes the interpupillary distance the pipeline *implies*,

$$I_{\text{implied}} = \mathrm{median}\left( i \cdot \frac{Z_{\text{mp}}}{f_{\text{true}}} \right)$$

and compares it against the viewer's real one:

$$s = \frac{I_{\text{real}}}{I_{\text{implied}}}$$

Every depth afterwards is $s\,Z_{\text{mp}}$.

### What the scale factor actually absorbs

Two independent errors, in one constant, which is why the numbers look wrong until this is seen.

On g7 the implied interpupillary distance comes out around **40 mm**, which is not any human's.
That is not a bug. $Z_{\text{mp}}$ was computed by MediaPipe under its own assumed focal length
$f_{\text{mp}}$, while $I_{\text{implied}}$ divides by the true $f_{\text{true}}$, so the ratio
1.609 from above appears directly: $66.9 / 1.609 = 41.6$ mm.
The constant $s$ therefore absorbs **the identity error and the focal length mismatch together**,
and neither is separately recoverable from it. Do not read $s$ as a statement about the viewer's
head.

### Why the estimator is not circular

It uses apparent interpupillary distance, which the previous section rejected. The distinction is
that it uses it **once, on gated frames, to fix a constant**, rather than per frame.

Substituting the definitions, the depth on a calibration frame is

$$s\,Z_{\text{mp}} = \frac{I_{\text{real}}}{\;i\,Z_{\text{mp}}/f_{\text{true}}\;}\;Z_{\text{mp}}
= f_{\text{true}}\,\frac{I_{\text{real}}}{i}$$

which is exactly the interpupillary estimator that was rejected. $Z_{\text{mp}}$ cancels.
That identity holds **only on the frames the median was taken over**, all of which are front-facing,
where the interpupillary estimator is unbiased. Afterwards $s$ is frozen and the depth follows
$Z_{\text{mp}}(t)$, which is yaw-robust.

So the calibration is a one-time transfer: it takes absolute scale from a measurement that is
accurate only face-on, and gives it to a measurement that tracks well in every pose but has no
absolute scale of its own. Each estimator is used for the one thing it is good at.

### Freezing rather than rolling

`LiveScale` collects 30 front-facing samples, calls `estimate_head_scale` once on the buffer, and
never moves the answer again.

A rolling estimate would be more responsive and is wrong here. The scale multiplies depth, depth
sets the frustum extents, and the extents set the apparent size of everything: a scale that drifts
makes the whole scene breathe. Being consistently 2% off is invisible; being 2% off in a slowly
varying way is not. Freezing also makes the live path produce the same number as the offline path,
which is what lets a recorded clip be a regression test for a live run.

Reading the scale before it is ready raises `HeadScaleNotReadyError` rather than defaulting to 1.0,
because a default of 1.0 renders a plausible scene at the wrong depth and looks like a working loop.

The bootstrap is gated on being front-facing, not on time, so a viewer who is turned away supplies
no samples and the bootstrap simply takes longer.

## Smoothing

The eye position is filtered before anything downstream sees it, by a causal left-triangle filter
whose weights rise towards the newest sample and sum to 1 (`PositionSmoother`, five taps, one per
axis, on `pose_tools.utils.np_signal`).

The reason is a sensitivity figure. A lateral eye movement of $\delta$ moves a point at depth $D$
behind the panel by $\delta \cdot D/(e_z + D)$ across a panel of width $w$ rendered into $W$ pixels,
so the gain in pixels per metre of head movement is

$$\frac{\partial u}{\partial e_x} = \frac{D}{e_z + D}\cdot\frac{W}{w}$$

At the built scene's back wall, $D = 0.6$ m, a viewer at $e_z = 0.5$ m, $w = 0.344$ m and
$W = 1920$: about **3000 px per metre, or 3 px per millimetre**.
Landmark jitter of a millimetre is three pixels of scene movement, which is visible.

Two properties are load bearing:

- **Causal.** Nothing looks at future frames, so an offline run over a recording and a live run
  produce identical numbers. A centred filter would smooth better and make the two paths disagree.
- **Held, not interpolated, across a gap.** A frame with no face is not a measurement.
  `hold()` returns the last smoothed position unchanged rather than feeding the filter a guess or
  resetting it, because the filter's weights assume evenly spaced samples.

The history is primed with the first sample rather than with zeros, so the track does not ramp up
from the origin over the first few frames.
