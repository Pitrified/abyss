# Running the phone page on g4 and the Pixel

Phase 4 of [`plans/03_phone_webapp/`](../../plans/03_phone_webapp/tracking.md), in the order to do it:
g7 serves, a person opens the page on g4 and then on the Pixel 7 Pro, and writes down what they see.

Each section says what to open, what a good result looks like, and what to record.
"It looked right" is not a result: the phase closes on numbers in the tracking log.
Everything recorded goes in one dated entry appended to
[`plans/03_phone_webapp/tracking.md`](../../plans/03_phone_webapp/tracking.md);
a template is at the bottom of this page.

## 0. What you need

- g7 serving the page over the tailnet (section 1). Nobody needs to be at g7.
- g4, with a person at it.
- The Pixel 7 Pro, on the tailnet, in Chrome.
- A tape measure that reaches 1 m, for the depth check.
- Nobody else in frame. The landmarker is asked for one face and picks one of two arbitrarily.

## 1. Serving, on g7

```bash
make serve          # 127.0.0.1:8000, loopback only
```

`tailscale serve` on g7 terminates HTTPS on the tailnet name and proxies to it, so the page is at
**`https://g7.tail5dff8b.ts.net/`** from g4 and the phone.
HTTPS matters: the camera API only exists in a secure context, which `localhost` is and a bare
tailnet IP over HTTP is not.

**Good**: `https://g7.tail5dff8b.ts.net/phone_web/?eye=0,0.1,0.6&screen=g7_internal` opens on g4 and
draws the scene: a grey box receding into the screen, a cyan frame around the edge, an orange cube
right of centre.
That page needs no camera and no face, so if it fails, the problem is serving, not tracking.

## 2. The probe, on each device

Open `https://g7.tail5dff8b.ts.net/probe/`, allow the camera, tap **fullscreen**, tap **copy**,
and paste the text into the log entry. On g4, run it in a normal window.

It reports what the device says about itself, with the camera asked for exactly as the phone page
asks for it. Read these out of it:

- **`camera.settings.width` and `height`: the stream's orientation.** On the Pixel held upright,
  good is portrait, width smaller than height, e.g. 720x1280.
  If it is landscape, the iris pixels are in a rotated frame and the focal length belongs to the
  other axis: the depth and the sideways numbers are both wrong, and nothing later in this page is
  worth recording until that is fixed. Stop and record it.
- **`screen.physical`**: 1440x3120 on the Pixel at its maximum resolution setting, or 1082x2342 at
  its default 1080p one (seen 2026-10-23). Either is the same glass: `pixel7pro` is in metres, so
  only sharpness differs. Anything else means the browser is not reporting the panel.
- **`afterTap.lock`**: `"locked"` on the Pixel. An error name means the page cannot hold the
  phone in portrait, and turning it will reflow the picture.
- **`afterTap.window.inner`** times `screen.devicePixelRatio` should be close to `screen.physical`:
  fullscreen gave the page the whole panel. If it falls well short, the bars are still taking part
  of it.
- `webgl.renderer`, `device.model`, `camera.label`: identification for the log.

The same numbers on the phone page are on `window.abyss.camera`, which on g4 is one F12 away; the
probe is for the Pixel, where there is no console without a cable.

## 3. The mirroring check, on each device

**Do this before anything else on the phone page.** It decides whether every sideways number has
the right sign, and the picture tests in `make e2e` cannot check it: they cover the maths, not the
registry's `mirrored` flag for a real stream.

Open the page with the device's own camera and screen, and no scene yet:

- g4: `https://g7.tail5dff8b.ts.net/phone_web/?camera=g4_internal`
- Pixel: `https://g7.tail5dff8b.ts.net/phone_web/?camera=pixel7pro_front`

Look at the camera until the status line stops reading `frame N: calibrating n/30` and reads
`frame N: eye x, y, z cm | f fps`. Then, keeping the device still, move your head about 10 cm to **your own
right**.

**Good**: `x` on the status line goes **negative**. The camera frame's x points to the image's right,
and in an unmirrored image of you, your right is the image's left. The scene is built on that
convention; `test_scene_moves_like_a_window` holds the renderer to it.

If `x` goes positive, the registry's `mirrored` flag is wrong for that stream. The likely case is
the Pixel: `pixel7pro_front` says `mirrored=True` on the assumption that front cameras mirror, and
Chrome usually delivers the front camera unmirrored and leaves mirroring to the preview. Flip the
flag in `src/abyss/params/abyss_devices.py`, write what you saw into its `provenance`, and redo the
check.

A second, independent look: raise your right hand. In the camera view, if it appears on the side of
the screen opposite your hand, as a person facing you would see it, the stream is unmirrored. The
page draws the stream as delivered, with no CSS mirror.

**Record**: device, the sign of `x` for a move to your right, and the flag you left the registry on.

## 4. The effect, on g4

```
https://g7.tail5dff8b.ts.net/phone_web/?camera=g4_internal&screen=g4_internal
```

Sit square to the screen at about 60 cm, look at the camera for the 30 calibrating frames, then
move.

**Good**:

- Moving sideways, the box's far wall moves the same way as your head while its rim stays put, as
  through a window; the cyan frame stays on the screen's edge.
- Moving closer, the box deepens and more of its side walls show.
- The status line reads at least 15 fps.

Expect the depth to feel slightly off: `g4_internal`'s camera is unmeasured and its offset is a
bezel guess, both marked as such in the registry. That is not a result about the page.

**Depth check.** Hold still at a distance you measure with the tape, from the screen to between
your eyes. Read `z` off the status line. Do it at about 40 cm and about 80 cm.

**Frame times.** Press F12 and run `abyss.timing` in the console after a few seconds of tracking.
Copy `median` and `fps`.

**Record**: fps, `abyss.timing.median`, the two tape-versus-`z` pairs, and one sentence on how the
effect looked.

## 5. The effect, on the Pixel

```
https://g7.tail5dff8b.ts.net/phone_web/?camera=pixel7pro_front&screen=pixel7pro
```

**Tap once** first: fullscreen and the portrait lock. Without it the browser's bars take part of the
panel and the picture is drawn at the wrong scale for the glass. Then hold the phone upright, about
35 cm away, square to your face, and look at the camera through the 30 calibrating frames.

**Good**:

- The same window behaviour as on g4, with the cyan frame on the glass's edge.
- The status line reads at least 15 fps: the floor from the renderer survey.
- The picture is sharp: it draws at the phone's pixel ratio, about 3.5 drawn pixels per CSS pixel.

The orange cube sits 8 cm right of the panel's centre and the panel is only 3.6 cm across from centre
to edge, so expect to see little or none of it.
That is the scene's layout, built for a laptop panel, not a bug in the projection.

**Delegate.** Repeat with `&delegate=GPU` added. Record fps for both; it has never been tried, so an
error on the status line is a result too.

**Stage split.** The phone has no console. The status line gives fps, which is enough for the floor.
For `abyss.timing.median`, connect the phone to g4 by USB with USB debugging on, open
`chrome://inspect` in Chrome on g4, inspect the tab, and run `abyss.timing` in its console.
Optional: record fps alone if that is not set up.

**Depth check**, as on g4, at about 30 cm and 50 cm. The Pixel's camera is unmeasured and falls back
to MediaPipe's assumed field of view, so a constant ratio between tape and `z` is expected; record
it, since it is what a focal length measurement would correct.

**Record**: fps on CPU and GPU, `abyss.timing.median` if read, the two tape-versus-`z` pairs, and one
sentence on how the effect looked held in the hand.

## Log template

```markdown
- YYYY-MM-DD : **phase 4, <device>.** Probe: stream WxH, physical WxH, lock <...>, renderer <...>.
  Mirroring: head right gave x <sign>; `mirrored` left at <...>. Effect: <one sentence>.
  fps <n> CPU, <n> GPU. Median ms: landmarker <n>, eye <n>, matrix <n>, draw <n>, interval <n>.
  Depth: tape <cm> read <cm>, tape <cm> read <cm>.
```
