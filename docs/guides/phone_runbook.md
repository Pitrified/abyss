# Running the phone page on g4 and the Pixel

Phase 4 of [`plans/03_phone_webapp/`](../../plans/03_phone_webapp/tracking.md), in the order to do it:
g7 serves, a person opens the page on a device, and writes down what they see.
The Pixel was run on 2026-10-23; g4 is not yet, and is not needed to close the phase.

Each section says what to open, what a good result looks like, and what to record.
"It looked right" is not a result: a run is recorded as numbers in the tracking log.
Everything recorded goes in one dated entry appended to
[`plans/03_phone_webapp/tracking.md`](../../plans/03_phone_webapp/tracking.md);
a template is at the bottom of this page.

## 0. What you need

- g7 serving the page over the tailnet (section 1). Nobody needs to be at g7.
- The device, on the tailnet, in Chrome: the Pixel 7 Pro, or g4 with a person at it.
- Nobody else in frame. The landmarker is asked for one face and picks one of two arbitrarily.

No tape measure: absolute depth is not checked. The Pixel's focal length comes from its published
field of view, and its likely error, a few percent, was accepted rather than measured
(2026-10-23).

## 1. Serving, on g7

```bash
make serve          # 127.0.0.1:8000, loopback only
```

`tailscale serve` on g7 terminates HTTPS on the tailnet name and proxies to it, so the page is at
**`https://g7.tail5dff8b.ts.net/`** from g4 and the phone.
HTTPS matters: the camera API only exists in a secure context, which `localhost` is and a bare
tailnet IP over HTTP is not.

**Good**: `https://g7.tail5dff8b.ts.net/phone_web/?eye=0,0.1,0.6&screen=g7_internal` opens and
draws the scene: a grey box receding into the screen, a cyan frame around the edge, an orange cube
right of centre.
That page needs no camera and no face, so if it fails, the problem is serving, not tracking.

Two things on every page, top right: the version, `ui <hash>`, which should match
`curl -s https://g7.tail5dff8b.ts.net/version.js` (an older one means a stale copy), and the
**log** button, which copies what the page saw, for pasting into a chat.
The log lives in the page: a reload empties it, so copy it before reloading.

**Loading.** While the page loads, the status line counts seconds by phase
(`loading landmarker: 7 s`), and once the landmarker is ready the log lists each large download
with its size on the wire and how long it took.
A first load fetches about 7.7 MB, gzipped where it helps; a reload revalidates and fetches almost
nothing. On the Pixel, connected directly to g7, a first load reached the landmarker in 13.7 s and
a reload in 1.65 s. When `tailscale status` on g7 shows the phone through a relay (`relay "fra"`)
rather than `direct`, expect a first load of a minute or more: the relay measured 70 to 140 kB/s.

## 2. The probe, on each device

Open `https://g7.tail5dff8b.ts.net/probe/` and allow the camera.
Tap **fullscreen**, then hold the device still for the camera sweep below; when the status reads
done, tap **copy** and paste the text into the log entry. On g4, run it in a normal window.

It reports what the device says about itself, with the camera asked for exactly as the phone page
asks for it. Read these out of it:

- **`camera.settings.width` and `height`: the stream's size and orientation.** On the Pixel held
  upright, good is 1440x1920: portrait, the sensor's full width, and the shape its focal length is
  for. Android Chrome swaps the requested width and height for an upright phone, which is why the
  page asks for 1920x1440. A landscape stream means the iris pixels are in a rotated frame and
  nothing later in this page is worth recording until that is fixed. Stop and record it.
- **`screen.physical`**: 1440x3120 on the Pixel at its maximum resolution setting, or 1082x2342 at
  its default 1080p one. Either is the same glass: `pixel7pro` is in metres, so only sharpness
  differs. Anything else means the browser is not reporting the panel.
- **`afterTap.lock`**: `"locked"` on the Pixel. An error name means the page cannot hold the
  phone in portrait, and turning it will reflow the picture.
- **`afterTap.window.inner`** times `screen.devicePixelRatio` should be close to `screen.physical`:
  fullscreen gave the page the whole panel. If it falls well short, the bars are still taking part
  of it.
- `webgl.renderer`, `device.model`, `camera.label`: identification for the log.

The same numbers on the phone page are on `window.abyss.camera`, which on g4 is one F12 away; the
probe is for the Pixel, where there is no console without a cable.

**Camera sweep.** Once the camera is open, the probe counts down three seconds, then opens the
front camera with ten different requests in turn, tracking the face in each; no tap needed. Hold
the device still, at one distance, square to your face, until the status reads done (about 15 s).
`sweep.variants` lists what each request delivered and `widthPerIpd` / `heightPerIpd`, the frame's
size in iris spacings: larger is a wider view. `sweep.widest` is the widest across. On the Pixel it
is the 1920x1440 request, delivered as 1440x1920. Two streams' `ipdPx` ratio is the ratio of their
focal lengths.

## 3. The mirroring check, on each device

**Do this before anything else on the phone page with a new camera.** It decides whether every
sideways number has the right sign, and the picture tests in `make e2e` cannot check it: they cover
the maths, not the registry's `mirrored` flag for a real stream.

Open the page with the device's own camera and screen, and no scene yet:

- g4: `https://g7.tail5dff8b.ts.net/phone_web/?camera=g4_internal`
- Pixel: `https://g7.tail5dff8b.ts.net/phone_web/?camera=pixel7pro_front`

Look at the camera until the status line stops reading `frame N: calibrating n/30` and reads
`frame N: eye x, y, z cm | f fps`. Then, keeping the device still, move your head about 10 cm to
**your own right**.

**Good**: `x` on the status line goes **negative**. The camera frame's x points to the image's right,
and in an unmirrored image of you, your right is the image's left. The scene is built on that
convention; `test_scene_moves_like_a_window` holds the renderer to it.

If `x` goes positive, the registry's `mirrored` flag is wrong for that stream: flip it in
`src/abyss/params/abyss_devices.py`, write what you saw into its `provenance`, and redo the check.
The Pixel needed exactly that: Chrome delivers its front camera unmirrored, leaving mirroring to
the preview, and `pixel7pro_front` is now `mirrored=False`, checked on the device.

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

**Frame times.** After a few seconds of tracking, tap **log**: the header's `timing` line has the
median of each stage and `fps`.

**Record**: fps, the timing medians, and one sentence on how the effect looked.

## 5. The effect, on the Pixel

```
https://g7.tail5dff8b.ts.net/phone_web/?camera=pixel7pro_front&screen=pixel7pro&delegate=GPU
```

**Tap once** first: fullscreen and the portrait lock. Without it the browser's bars take part of the
panel and the picture is drawn at the wrong scale for the glass. Then hold the phone upright, about
35 cm away, square to your face, and look at the camera through the 30 calibrating frames.

**Good**:

- The same window behaviour as on g4, with the cyan frame on the glass's edge.
- The status line reads at least 15 fps, the floor from the renderer survey. Seen: 30 fps on both
  delegates, the landmarker about 28 ms a frame.
- The picture is sharp: it draws at the phone's pixel ratio, 2.625 at the default 1080p setting.
- `scale ready` in the log near 0.74. The head scale absorbs the gap between MediaPipe's assumed
  field of view and the spec focal; seen 0.735 to 0.753.
- No `warning:` line in the log. One appears when the stream is not the 3:4 shape the focal length
  is for, and then the depth is off: record the stream size it names.

The orange cube sits 8 cm right of the panel's centre and the panel is only 3.6 cm across from centre
to edge, so expect to see little or none of it.
That is the scene's layout, built for a laptop panel, not a bug in the projection.

**Delegate.** `delegate=GPU` and the default CPU both hold 30 fps on the Pixel; drop the parameter
to compare.

**Frame times.** Tap **log** after a few seconds of tracking, as on g4. The header carries the stage
medians and fps, and the lines below a summary a second. The button does not leave fullscreen.

**Record**: fps on the delegate used, the timing medians, `scale ready`, and one sentence on how
the effect looked held in the hand.

## Log template

```markdown
- YYYY-MM-DD : **phase 4, <device>.** Probe: stream WxH, physical WxH, lock <...>, renderer <...>.
  Mirroring: head right gave x <sign>; `mirrored` left at <...>. Effect: <one sentence>.
  fps <n> on <delegate>. Median ms: landmarker <n>, eye <n>, matrix <n>, draw <n>, interval <n>.
  Scale <n>. Load: landmarker ready at <s> first load, <s> reload.
```
