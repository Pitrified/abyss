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
