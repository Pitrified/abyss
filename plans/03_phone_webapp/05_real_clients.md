---
status: planned
---

# Phase 4 - real clients: g4, then the Pixel

## Overview

The same page, served by g7 over the tailnet, on g4's browser and then on the Pixel 7 Pro. The
first phase that needs a person: someone has to sit at g4 and then hold the phone.

Context: phase 3 ([`04_frustum_and_scene.md`](04_frustum_and_scene.md)). The machine census is in
[`.github/copilot-instructions.md`](../../.github/copilot-instructions.md).

## What gets built

**Serving on the tailnet.** `getUserMedia` needs a secure context, so `localhost` stops being enough.
`tailscale serve` on g7 terminates HTTPS on its tailnet name and forwards to the local server. The
Cloudflare tunnel is not used.

**The Pixel's screen.** A `ScreenConfig` entry for the phone in `abyss_devices.py`: panel size and the
front camera's offset from the panel centre, with provenance. Browsers do not expose a panel's
physical size, so this is from the spec sheet and a ruler, not from the page. Portrait, since that is
how a phone is held.

**The Pixel's camera.** Stays on MediaPipe's assumed field of view, marked unmeasured, until the
deferred phone recording measures it (Q7 a+b). Its effect on depth is known from g7, where the
assumption was off by 38%, and the head scale calibration absorbs most of it.

**A runbook** in `docs/guides/`, in the shape of the expansion's live runbook: what to open, what a
good result looks like, what to record, for g4 and then for the phone.

## Carried in from phases 1-3

Found while building the page headless, and not checkable there. Each is small; each would make the
first run on the phone look broken for a reason that is not the maths.

- **Stage timings were not built.** Built 2026-10-23 on g7: `window.abyss.timing`, see the harness
  guide. Phase 3's plan published frame time and each stage's share on
  `window.abyss`, as the Python loop's benchmark does; the implementation did not. Phase 4's own
  measurement (frame times, the 15 fps floor) needs it, so it comes first: landmarker, eye position,
  matrix and draw, per frame, plus a rolling median, readable by the drive script.
- **Mirroring is unverified on a phone, and the registry may double-flip.** `pixel7pro_front` has
  `mirrored=True`, written on the assumption that front cameras mirror. Chrome's `getUserMedia`
  usually delivers the sensor image unmirrored and leaves mirroring to the preview's CSS. If so,
  `eyePositionM` flips x once more than it should and the scene moves the wrong way when the head
  moves sideways. Check on the device before trusting anything else: move the head to the viewer's
  right and read the sign of x on the status line (it should go negative, camera-frame left for an
  unmirrored image), then set the registry entry to what was seen, with provenance. The picture
  tests cover the frame chain, not this flag's value.
- **The stream's orientation is unverified.** The probe page (`/probe/`) reads it, with the same
  constraints as the phone page, alongside screen size, pixel ratio and safe area. The page asks for 720x1280 and reads back whatever it
  gets. A phone held upright may still deliver landscape frames, in which case the iris pixels are in
  a rotated frame and the focal length belongs to the other axis. Read `window.abyss.camera` on the
  device first.
- **The picture must fill the panel.** Built 2026-10-23: a tap enters fullscreen and tries the lock,
  reported on `window.abyss.display`, and the picture refits when the window changes. The frustum is built for the physical panel, but in a browser
  tab the address bar and navigation bar take part of it, so the page letterboxes into what is left
  and the scene is drawn at the wrong scale for the glass it sits behind. Needs the Fullscreen API
  (on a tap, since it requires a user gesture) and `screen.orientation.lock("portrait")`, which only
  works in fullscreen. Until then, expect the effect to look slightly off in depth.
- **The canvas renders at one pixel per CSS pixel.** Built 2026-10-23: `devicePixelRatio` unless
  `size=` is given. `SceneView` calls `setPixelRatio(1)` so tests
  read back a frame of the requested size. On the Pixel, with a device pixel ratio around 3.5, that
  is a blurry picture. Use `devicePixelRatio` when `size=` is not given.
- **The delegate is CPU by default.** Headless in the cloud a frame took about 55 ms on the CPU path.
  `delegate=GPU` exists and has never been tried; on the phone it is the first thing to compare.
- **A Pixel screen entry does not exist.** Added 2026-10-23 as `pixel7pro`, from specs and not a
  ruler, which would be less precise (decided by the user): 1440x3120 at 512 ppi, 71.4 x 154.8 mm,
  and the camera hole from AOSP's display cutout for the device, centred 72 px below the top edge,
  so 73.8 mm above the panel centre. Not marked provisional. The panel's curved long edges are not
  modelled. The scene's cube sits at +8 cm, beyond this panel's 3.6 cm half width, so it is barely
  in view on the phone; the scene may need laying out for a narrow panel.
- **Serving needs no code change.** `tailscale serve` on g7 terminates HTTPS on the tailnet name
  and proxies to `make serve` on `127.0.0.1:8000`, so the server keeps binding loopback. The CSP is
  same-origin, so it holds behind the proxy unchanged.
- **g4 has entries, both unmeasured.** `g4_internal` camera and screen exist; the screen offset is
  marked provisional. Fine for a first look; the numbers are not to be trusted for depth.

## Steps

1. **g4.** Open the page from g7's tailnet name. Its own webcam and screen: `g4_internal` entries
   exist, the camera unmeasured. Check the effect, record frame times.
2. **The Pixel.** Same page. Check the effect holding the phone, record frame times and the stage
   split, and whether the 15 fps floor from the renderer survey holds.

## Tests

The device numbers are unit-tested like the others. The rest is a person looking, recorded in the
tracking log with dates and numbers. Driving the cabled Pixel over CDP is deferred unless the manual
check keeps needing a second look.

## Done when

- The effect is visible on the Pixel, held in the hand, served by g7 over the tailnet.
- Frame times on g4 and on the Pixel are in the log.
- The runbook exists and was followed, not written after the fact.
