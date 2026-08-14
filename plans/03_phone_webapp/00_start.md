---
status: draft
---

# Phone webapp - capture and display over the network

Spun off from [`../01_abyss_expansion/00_start.md`](../01_abyss_expansion/00_start.md) (Q9, Q12,
Q14). The Pixel 7 Pro is a capture device and a display device, but not a compute host: Python does
not deploy there without effort, so the phone talks to a machine that runs the loop. That link is a
webapp, and it is future work.

## Why it is separate

- It reopens the FastAPI scaffold question that the reboot declined (#15 in the reboot inventory)
  and that the expansion declined again in Q3. Adding a web dependency deserves its own decision.
- The expansion's phase 5 closes the loop on g7 with a local window. Nothing there waits on this.
- The problems are different in kind: latency over a network, frame encoding, browser rendering,
  and a phone camera whose intrinsics are not measurable on this box.

## What it inherits

The config split from the expansion carries over directly, and is most of the reason it exists:

- **camera config** - the Pixel's front camera, used wherever its frames are processed
- **stream config** - frames arriving over the network rather than from a file or a local device
- **screen config** - the phone screen, which is what the frustum is built for
- **sink config** - served to a browser instead of written to disk or shown in a window

A clip recorded on the phone and processed on g4 already exercises the first two without any of this
machinery, which is the cheap way to start.

## Open questions

Numbering is local to this folder.

- Q1: **Where does the split land - frames or landmarks?** Sending camera frames to the server and
  rendered frames back is the obvious shape and the most bandwidth. Running the landmarker on the
  phone and sending only an eye position is far lighter, and MediaPipe has a web build.
  ANS: ...
- Q2: **Does the browser render, or does the server?** If the browser renders (three.js), the server
  only sends a viewer position and the phone GPU does the work. If the server renders, it sends
  video. This mostly follows from Q1.
  ANS: ...
- Q3: **Is a phone even needed for a first version?** A laptop browser on the same network exercises
  the whole path with none of the phone-specific unknowns.
  ANS: ...
