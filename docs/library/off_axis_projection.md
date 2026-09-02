# From an eye position to a projection

Steps 5 and 6 of the chain in [`geometry_overview.md`](geometry_overview.md): how a point in metres
becomes a 4x4 matrix and then a pixel.
Code in `src/abyss/render/frustum.py`, `src/abyss/render/renderer.py` and
`src/abyss/render/scene.py`.

## The idea

A conventional 3D camera has a field of view and points at something.
A window does not. What the viewer sees through a real window is fixed by two things only:
where their eye is, and where the four edges of the frame are.
Reproducing that on a panel means building a projection whose image plane **is** the panel,
clipped to the panel's rectangle, seen from wherever the eye is.

The eye is generally not on the panel's axis, so the resulting frustum is **asymmetric**: its axis
does not pass through the centre of its own image. This is the off-axis or sheared frustum,
the same construction stereo rendering uses for its two eyes, and the same one CAVE displays use for
their walls.

### Off-axis, not toe-in

The common wrong implementation is to keep a symmetric frustum and rotate the camera to look at the
panel centre. It is easier, it tracks the head, and it is not the same picture.

Rotating the camera rotates the image plane away from the panel plane, so the panel's rectangle no
longer maps to the image rectangle: the scene is drawn as though the window had turned to face the
viewer, which real windows do not do. Straight lines in the room stay straight but the whole scene
shears as the head moves.

In this codebase the camera is **never rotated**. `view_projection_matrix` composes a projection
with a pure translation and no rotation at all: the view direction is always the screen frame's
$-Z$, whatever the eye does. All of the head coupling lives in the frustum's asymmetry.
That is the single sentence to keep: *head-coupled perspective shears the frustum, it does not aim
the camera*.

## The camera to screen conversion

`eye_in_screen_frame` is, as its docstring says, the one interesting function in the module.
Given the eye in the camera frame and the camera-to-panel-centre offset $c$ from `ScreenConfig`:

$$e_{\text{screen}} = M\,(e_{\text{camera}} - c), \qquad M = \operatorname{diag}(-1, -1, 1)$$

Two things about $M$.

**Why both signs flip.** $Y$ flips because the camera frame is y-down and the screen frame is y-up.
$X$ flips because the camera *faces* the viewer: a person facing you has their right hand on your
left, so the viewer's right is the image's left. Both are flips of the same kind, and the offset
subtraction happens in the camera frame, before them, which is why a camera mounted **above** a
panel is stored with a **positive** Y offset.

**Why it must be both.** $\det \operatorname{diag}(-1,-1,1) = +1$, so $M$ is a rotation: 180 degrees
about $Z$, which is the right shape for two frames that face each other.
Flipping one axis alone would give $\det = -1$, a reflection, and would silently mirror the world.
A mirrored head-coupled scene still moves with the head and still has parallax; it is inside out and
looks fine. This is why the conversion is pinned by directional tests
(`test_moving_the_eye_right_shifts_the_frustum_left`) and not only by the corner test below, which
a wrong conversion passes.

$M$ is its own inverse, so there is only one function and no direction to get backwards.

The horizontal sign additionally depends on whether the capture was mirrored, as a front-facing
phone camera is. That is applied **upstream**, in `eye_position_m`, from `CameraConfig.mirrored`.
Nothing in the render layer may flip it a second time.

## Building the frustum

The screen frame puts the panel at $z = 0$ occupying
$[-w/2, w/2] \times [-h/2, h/2]$, and the eye at $(e_x, e_y, e_z)$ with $e_z > 0$.
The near clip plane sits at distance $n$ in front of the eye.

The near plane is a clipping construct and is **not** the panel plane.
The extents on it follow from similar triangles: the panel edge at $x = w/2$ is $w/2 - e_x$ from the
eye laterally at a distance of $e_z$, and the same ray at distance $n$ is at $(w/2 - e_x)\,n/e_z$.
So

$$
r = (w/2 - e_x)\frac{n}{e_z}, \quad
l = (-w/2 - e_x)\frac{n}{e_z}, \quad
t = (h/2 - e_y)\frac{n}{e_z}, \quad
b = (-h/2 - e_y)\frac{n}{e_z}
$$

Three consequences fall straight out, and each is a test.

**The frustum's size depends only on distance.**

$$r - l = w\,\frac{n}{e_z}, \qquad t - b = h\,\frac{n}{e_z}$$

Moving sideways does not change how much is seen, only which part.
Moving closer increases $n/e_z$ and widens the view, which is what makes leaning in feel like
leaning into a window (`test_moving_closer_widens_the_view`).

**The asymmetry is exactly the lateral offset.**

$$\frac{r + l}{2} = -e_x\frac{n}{e_z}, \qquad \frac{t + b}{2} = -e_y\frac{n}{e_z}$$

A centred eye gives $r = -l$ and $t = -b$, an ordinary symmetric frustum
(`test_a_centred_eye_gives_a_symmetric_frustum`).
The shift is opposite in sign to the eye's movement, which is the sheared-frustum construction and
the source of the parallax direction below.

**The frustum always has the panel's aspect ratio.**

$$\frac{r - l}{t - b} = \frac{w}{h}$$

for every eye position. This is why `check_aspect` exists and why it is an error rather than a
correction. The projection maps the panel rectangle onto the viewport rectangle whatever the viewport
is, so rendering a 16:9 panel into a 4:3 image does not crop or letterbox: it fills perfectly and
stretches everything silently. The tolerance is 2%, because a real panel is not exactly 16:9
(g7 is 1.7824 against 1280x720's 1.7778) and an equality test would reject the actual hardware.

## The projection matrix

With the frustum as extents, the matrix is the standard `glFrustum` form, mapping eye space (viewer
at the origin, looking down $-Z$) to clip space:

$$
P = \begin{pmatrix}
\dfrac{2n}{r-l} & 0 & \dfrac{r+l}{r-l} & 0 \\[2ex]
0 & \dfrac{2n}{t-b} & \dfrac{t+b}{t-b} & 0 \\[2ex]
0 & 0 & -\dfrac{f+n}{f-n} & -\dfrac{2fn}{f-n} \\[2ex]
0 & 0 & -1 & 0
\end{pmatrix}
$$

### Where the first row comes from

A point $(x, y, z)$ with $z < 0$ projects onto the near plane at $x' = -n\,x/z$.
Mapping $x' \in [l, r]$ linearly onto $[-1, 1]$:

$$x_{\text{ndc}} = \frac{2(x' - l)}{r - l} - 1 = \frac{2x' - (r + l)}{r - l}$$

Substituting $x' = -nx/z$ and multiplying through by $-z$, which is the homogeneous $w$ the fourth
row produces:

$$x_{\text{clip}} = \frac{2n}{r-l}\,x + \frac{r+l}{r-l}\,z$$

which is the first row. The off-axis term $\frac{r+l}{r-l}$ multiplies $z$, so the shear is
proportional to depth: that term is zero for a symmetric frustum and is the only structural
difference between this and a textbook perspective matrix.

### Where the third row comes from

$z_{\text{ndc}} = (Az + B)/(-z)$, with $A$ and $B$ fixed by requiring $z = -n \mapsto -1$ and
$z = -f \mapsto +1$. Solving the pair gives the entries above.

The map is **hyperbolic in $z$, not linear**: most of the NDC depth range is spent close to the near
plane. It does not matter today, since the wireframe renderer does no depth testing at all, but it
is what makes the near plane the sensitive parameter in any depth-buffered renderer that replaces it.
$n = 0.05$ m and $f = 100$ m here, chosen to sit well inside any plausible eye distance.

### The view half

`view_projection_matrix` returns $P\,T$ where $T$ translates by $-e_{\text{screen}}$.

That is the whole view matrix: a translation, no rotation, as argued above.
Folding it in is what lets every caller work in the screen frame and never handle an eye-space
coordinate, which keeps the two frames from meeting anywhere outside this module.

## Perspective divide and viewport

`project_points` finishes the job:

$$
p_{\text{ndc}} = \frac{p_{\text{clip}}}{w}, \quad w = -z_{\text{eye}}
\qquad
u = \frac{x_{\text{ndc}} + 1}{2}\,W, \quad v = \frac{1 - y_{\text{ndc}}}{2}\,H
$$

The $1 - y$ is the last sign flip in the chain, undoing the y-up of NDC for a y-down image.

A point with $w \le 0$ is at or behind the eye and the divide is undefined, so it raises
`PointNotInFrontError` rather than producing a plausible pixel. There is no clipper: a point outside
the frustum projects to a pixel outside the image, and `cv.line` discards the rest.

That is only safe because of a constraint enforced at the other end. `Scene` refuses any geometry
with $z > 0$, so the whole scene is behind the panel while the eye is in front of it, and no point
can ever be behind the eye. Clipping is not implemented because it is not reachable, and the
constraint is checked at construction rather than assumed.
The named upgrade, needed the first time something is meant to poke out through the window, is
homogeneous-space segment clipping against the near plane.

## Two invariants

The properties that define the projection, both pinned by tests, both invisible in a live demo.

### The window invariance

**The four physical corners of the panel project to the four corners of the image, for every eye
position with $e_z > 0$, and for every choice of near and far plane.**

Proof. Take the corner $(w/2,\ h/2,\ 0)$. In eye space it is at
$x_{\text{eye}} = w/2 - e_x$ and $z_{\text{eye}} = -e_z$. Then

$$x' = -n\,\frac{x_{\text{eye}}}{z_{\text{eye}}} = (w/2 - e_x)\frac{n}{e_z} = r$$

by the construction of $r$, so $x_{\text{ndc}} = +1$ and $u = W$. The same holds for the other three
corners and for $y$. $\square$

$n$ cancels, which is the formal reason the near plane need not be the panel plane
(`test_the_near_plane_is_not_the_screen_plane`).

This is the property that makes the panel a window rather than a picture, and it is what the cyan
frame marker in the scene is instrumentation for: the marker is the panel rectangle scaled to 0.98,
so it draws as four edges with a uniform gap to the image border. A gap that stays uniform while the
head moves is this invariance, visible. A gap that opens on one side is a projection error.

Note what the corner test does **not** catch, established by mutating the code rather than assumed:
a wrong camera-to-screen conversion builds a frustum that is wrong in a matching way, and the corners
still land perfectly. The directional tests are what cover that.

### The parallax law

**A point at depth $D$ behind the panel moves across the panel with gain $D/(e_z + D)$ relative to
the eye's lateral movement, in the same direction.**

Proof. The point's image on the panel is where the segment from the eye to the point crosses
$z = 0$. With the eye at $(e_x, e_y, e_z)$ and the point at $(0, 0, -D)$, linear interpolation gives

$$x_{\text{panel}} = e_x + (0 - e_x)\frac{e_z}{e_z + D} = e_x\,\frac{D}{e_z + D} \qquad \square$$

The gain runs from 0 at the panel plane to 1 at infinity, so anything drawn *at* the window does not
move and distant things track the head almost one for one. It is the quantity the whole effect is
made of, and it is why the scene has a grid on its back wall: the grid is what the floating cube
slides against, and without something at depth there is nothing for parallax to be visible on.

For the built scene, $D = 0.6$ m with a viewer at 0.5 m gives a gain of 0.545: the back wall slides
about 55% of the head movement.
`test_a_point_behind_the_screen_stays_put_when_the_eye_moves` pins the sign, which is the half that
can be wrong.

## The renderer

`Renderer` is a protocol with one method:

```python
def render(self, view_projection: np.ndarray, width_px: int, height_px: int) -> np.ndarray: ...
```

A matrix and a size in, a BGR frame out. The seam sits here rather than at the scene deliberately:
the named second implementation is an OpenGL renderer, which produces pixels rather than geometry,
so a geometry-shaped interface would have excluded the implementation it exists for.
Candidates for that second renderer are surveyed in
[`../../plans/02_scene_rendering/01_research_renderers.md`](../../plans/02_scene_rendering/01_research_renderers.md).

`WireframeRenderer` is the current one, and it is the cheapest thing that makes the effect visible:
anti-aliased `cv.line` calls, no occlusion, no depth sort.
With wireframe a depth sort would only decide which of two crossing lines is drawn on top, which
nothing can assert and nobody notices, so depth is carried by a colour fade baked into the scene at
build time instead. The fade depends only on where a segment sits, which does not change when the
eye moves, so it costs nothing per frame.
