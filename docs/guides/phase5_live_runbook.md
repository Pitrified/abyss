# Running the loop live

Everything in phase 5 that a machine cannot do for itself, in the order to do it.

Run all of this **on g7**: it is the machine with a camera, a display and a person in front of it.
Sections 1 and 2 also run anywhere, deliberately, so a failure can be localised before the camera is
involved at all.

Each section says what to run, what a good result looks like, and what to record. Recording is not
optional: an unmeasured run does not close the phase, and "it looked right" is not a result.

Everything you record goes in one dated entry appended to
[`plans/01_abyss_expansion/tracking.md`](../../plans/01_abyss_expansion/tracking.md). A template is
at the bottom of this page.

## 0. What you need

- g7, logged in at the desk. Not over ssh: sections 3 onward need the real display.
- A tape measure or a ruler that reaches 1 m.
- A mirror or a phone camera, for section 2.
- Nobody else in frame. MediaPipe is asked for one face and will pick one of two arbitrarily.

## 1. Pre-flight, with no camera and no display

Run these first. They are quick, they need no hardware, and if any of them fails the live run will
fail in a way that is much harder to read.

```bash
make check
uv run --no-sync python scripts/live.py clip face01.mp4
```

Expected, for the second command:

- `Read 250 frames of 1920x1080 from face01.mp4`
- `Head scale frozen at 0.939` (the bootstrap, from the first 30 front-facing frames)
- `250 frames in ~8 s, ~30 fps: 250 with a face, 0 held, 29 calibrating`
- 250 PNGs in `cache/live/face01/`

Then look at two of them, because the numbers above are all true of a loop that renders nothing:

```bash
xdg-open cache/live/face01/frame_00005.png    # calibrating: grey, "Look at the camera: n/30"
xdg-open cache/live/face01/frame_00120.png    # the scene: cyan frame marker, orange cube
```

**Record**: the frame count, the achieved fps, and the frozen head scale.

The head scale is worth a second of attention. The bootstrap sees 30 front-facing frames and the
offline estimator sees all 218, and on this clip they give **0.939 against 0.941**, 0.2% apart.
That difference is the entire cost of Q23's bootstrap-and-freeze, and it is smaller than the 16%
per-identity spread it corrects. If your run shows them far apart, something is wrong with the
clip or the bootstrap, not with the theory.

## 2. Measure your interpupillary distance

**This is the last unmeasured number in the whole chain** and it maps straight into depth. The
reported depth is `focal * ipd_real / ipd_px`, so a viewer who is really 60 mm read at the 63 mm
population default comes out 5% too far away, as a constant offset at every distance.

Two ways, either is fine:

- **Mirror.** Hold a ruler against your brow, look at one eye in the mirror, and read the mark over
  the centre of each pupil. Do it three times and take the middle value.
- **Photo.** Have someone photograph you head-on holding a ruler under your eyes, then read the
  pupil centres off the photo.

Millimetres, to the nearest millimetre. Anything better is false precision; anything worse than
about 2 mm starts to matter at 3%.

**If the mirror reading is hard to call, that is fine and does not block anything.** Use your best
estimate now: section 4 measures the same number a second way, because a wrong interpupillary
distance shows up there as a constant offset at every distance, and the offset gives it back exactly
as `ipd_true = ipd_used * measured / reported`. The mirror is the starting guess; the tape measure is
the measurement.

**Record**: the value, and which method. Then either pass it as `--viewer-ipd-mm 64` for now, or add
a `VIEWERS` entry in `src/abyss/params/abyss_devices.py` once you trust it.

## 3. The live run

Sit down first. The window opens fullscreen and covers the screen.

```bash
uv run --no-sync python scripts/live.py camera --viewer-ipd-mm <yours>
```

Two invocation notes, both learned the hard way. Use `uv run --no-sync`, never a bare `uv run`: the
bare form re-syncs from `uv.lock` first and silently reverts a local editable pose-tools install.
And the shared options go **after** the subcommand, which is where argparse now accepts them.

What should happen, in order:

1. A log line `Opened 0 at 1280x720 MJPG, buffer size 1`.
2. A grey screen reading `Look at the camera: n/30`. Look straight at the camera, above the screen.
   Turned-away frames do not count towards the bootstrap, so this waits for you rather than timing
   out.
3. `Head scale frozen at ...`, and the scene appears: a wireframe room through the panel, a cyan
   frame marker just inside the border, an orange cube floating mid-depth.
4. Move your head. **The cyan marker stays welded to the edges of the screen and the room moves
   behind it.** That is the whole effect. If the marker moves with your head, the projection is
   wrong; if nothing moves, tracking is not reaching the renderer.

Keys: `q` or escape to quit, `r` to re-run the bootstrap (use it if someone else sat down, or if
you bootstrapped while turned away).

**Record**: both lines the loop prints on exit. The second one is the important one:

    250 frames in 10.6 s, 23.5 fps: 250 with a face, 0 held, 29 calibrating
    median ms per frame: capture 1.7 track 14.8 render 4.0 sink 20.2 | measured 40.7 of 42.5 actual

`capture` is waiting for the source, `track` is the landmarker, `render` is the projection and the
drawing, `sink` is the display. **If `measured` is much less than `actual`, the time is going
somewhere none of these stages covers** and that is worth reporting rather than living with.

Expect `capture` to be large in a fast loop and for that to be correct: a loop running quicker than
30 fps spends the remainder waiting for the camera, which is the pacing rather than a cost.

A slow loop honestly measured closes this phase; an unmeasured fast one does not.

If the rate is poor, the one-line experiment is to render smaller:

```bash
uv run --no-sync python scripts/live.py camera --viewer-ipd-mm <yours> --width 1280 --height 720
```

The geometry stays correct - the panel's aspect is what matters, not its pixel count - so if the
rate jumps, the cost is resolution-dependent and lives in the display path.

## 4. The tape measure check

**This is the phase's real exit criterion**, and the first check in the whole initiative that can be
wrong against the world rather than merely inconsistent with itself. Every earlier phase could only
verify internal consistency, because the sample clips have no known camera.

Measure from the **camera lens** (top bezel, above the panel centre) to the **bridge of your nose**.
Sit at each distance, hold still, and read the frame: the loop prints the eye position and the
apparent iris separation across the top left, so the number to compare is on the screen in front of
you.

    eye -0.018 -0.110 0.430 m (camera frame)   iris 123 px

**The third number is the depth, and it should equal what your tape measure says.** That is the
whole check. The chain reduces to `depth = focal * ipd / ipd_px` with g7's measured focal of 945 px
at 720 tall, so the iris column below is a cross-check on the same equation from the other side.

**The table depends on the interpupillary distance you passed in**, which is exactly why the depth
can be wrong: it scales linearly with it. Both columns are given, since 63 mm is the default and
60 mm is a plausible real value.

| you sit at | iris px at 63 mm | iris px at 60 mm |
| ---------- | ---------------- | ---------------- |
| 0.50 m | 119 | 113 |
| 0.60 m | 99 | 95 |
| 0.75 m | 79 | 76 |
| 1.00 m | 60 | 57 |

If you passed your own number and the reported depth still disagrees, the reported depth is what to
trust as the measurement and the interpupillary distance is what to correct.

Three distances is enough. What you are looking for is not perfection:

- **A constant offset at every distance** is the interpupillary distance being slightly off, and it
  is a scale factor, not a bug. Recompute: `ipd_true = ipd_used * measured / reported`.
- **An error that grows with distance** is a different problem - the focal length or the eye-offset
  model - and is worth a plan entry rather than a tweak.
- **Anything within a couple of percent** is a pass. The focal length itself is only known to 0.5%.

**Record**: each measured distance against each reported depth, and which of the two patterns above
you see. Do not tune anything away silently; if you change `ipd_m`, say what it was and why.

## 5. Known failures, and what they look like

Each of these has already happened once on this hardware. The error message is the first column.

| What you see | What it is | What to do |
| ------------ | ---------- | ---------- |
| `CaptureModeError: asked for 1280x720 and got 640x480` | The camera fell back to YUYV | Check nothing else has the camera open; `v4l2-ctl --list-formats-ext -d /dev/video0` |
| `CaptureIsBlackError` | The camera is gone, not dark | Happened once at the login screen. Interact with the desktop and re-run |
| `CaptureOpenError` | Wrong device index | `video0` is the RGB camera; `video2` is the infrared one |
| Stuck on `Look at the camera: 0/30` | Every frame reads as turned away | Face the camera squarely. Yaw over 10 degrees does not count |
| `HELD: no face` on screen | Tracking lost you | Expected when you leave frame. Persistent means lighting or distance |
| Scene renders but the marker drifts with your head | The projection is wrong | Stop and report it - this is a real defect, not a setup problem |
| The window is not fullscreen, or is letterboxed | Render size does not match the panel | Pass `--width`/`--height` to match your desktop resolution |

## 6. What to append to the log

Paste this into `plans/01_abyss_expansion/tracking.md`, filled in, dated, at the bottom:

```markdown
- YYYY-MM-DD : phase 5 run live on g7.
  Pre-flight: clip mode <N> frames at <X> fps, head scale bootstrap <A> against whole-clip <B>.
  Viewer interpupillary distance measured at <N> mm by <method>.
  Live: <N> frames in <X> s, <Y> fps, <F> with a face, <H> held, <C> calibrating.
  Tape measure: <D1> m read as <R1>, <D2> m read as <R2>, <D3> m read as <R3>.
  Pattern: <constant offset | growing error | within tolerance>, and what was done about it.
  What the effect actually looked like, including anything that felt wrong.
```

That last line matters more than it looks. The numbers say the loop ran; only a person can say
whether the screen behaved like a window.
