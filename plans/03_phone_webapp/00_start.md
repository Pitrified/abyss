---
status: draft
---

# Phone webapp - capture and display over the network

Spun off from [`../01_abyss_expansion/00_start.md`](../01_abyss_expansion/00_start.md) (Q9, Q12,
Q14). The Pixel 7 Pro is a capture device and a display device, but not a compute host: Python does
not deploy there without effort, so the phone talks to a machine that runs the loop. That link is a
webapp, and it is future work.

## Machines

The census is in [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md); what it
means here:

- **g7 is the server.** An unattended home server with the GPU, so it is where the loop runs and
  where an endpoint is served from.
- **g4 is the browser.** The daily driver, usually at home, so it is the quickest client to test
  against. Not a hard requirement.
- **The phone has two client shapes.** A web UI in the browser, or a custom Flutter or Unity app.
  Both have been installed on this phone before, so the choice is about fit, not about risk.
- **The network is a tailnet.** g7, g4 and the phone are on the same Tailscale network, so g7 can
  expose an endpoint to both without going through its Cloudflare tunnel. The tunnel stays an option
  for anything that has to be reachable from outside.

## Why it is separate

- It reopens the FastAPI scaffold question that the reboot declined (#15 in the reboot inventory)
  and that the expansion declined again in Q3. Adding a web dependency deserves its own decision.
- The expansion's phase 5 closes the loop on g7 with a local window. Nothing there waits on this.
- The problems are different in kind: latency over a network, frame encoding, browser rendering,
  and a phone camera whose intrinsics have not been measured.

## What it inherits

The config split from the expansion carries over directly, and is most of the reason it exists:

- **camera config** - the Pixel's front camera, used wherever its frames are processed
- **stream config** - frames arriving over the network rather than from a file or a local device
- **screen config** - the phone screen, which is what the frustum is built for
- **sink config** - served to a browser instead of written to disk or shown in a window

A clip recorded on the phone and processed on g7 already exercises the first two without any of this
machinery, which is the cheap way to start.

## Open questions

Numbering is local to this folder.

- Q1: **Where does the split land - frames or landmarks?** Sending camera frames to the server and
  rendered frames back is the obvious shape and the most bandwidth. Running the landmarker on the
  phone and sending only an eye position is far lighter, and MediaPipe has a web build.
  ANS: landmarks, and they never leave the phone. Tracking, the frustum and the drawing all run on the
  phone, so nothing per frame crosses the network and the round trip is not on the path from a head
  movement to the picture moving, which is the latency the effect is judged by. g7 serves the page and
  the assets and is otherwise out of the loop. The cost is a second implementation of the frustum
  maths, a few dozen lines, held to the Python one by a check (see
  [`01_research_dev_harness.md`](01_research_dev_harness.md)). Sending frames was rejected for that
  latency, and splitting tracking from drawing across the network keeps a hop on the path for no gain.
- Q2: **Does the browser render, or does the server?** If the browser renders (three.js), the server
  only sends a viewer position and the phone GPU does the work. If the server renders, it sends
  video. This mostly follows from Q1.
  ANS: the browser, with three.js. A native app (Flutter or Unity) was considered: Flutter is weak at
  3D and awkward for MediaPipe, and Unity is the upgrade path if the browser runs out of performance,
  not the starting point, because of its build cycle and licence. The page also runs unchanged on g4
  and in a headless Chromium, which is what makes it testable at all.
- Q3: **Is a phone even needed for a first version?** A laptop browser on the same network exercises
  the whole path with none of the phone-specific unknowns. g4 is that laptop, on the same tailnet.
  ANS: no. Order: headless Chromium with a recorded face fed in as the camera, then g4's browser,
  then the same page on the Pixel. Each step adds one unknown.

The second batch comes out of [`01_research_dev_harness.md`](01_research_dev_harness.md): how an
agent runs the page for real, with a recorded face standing in for the camera.

- Q4: **Where do the page's third-party files come from?** three.js and `@mediapipe/tasks-vision`,
  about 20 MB unpacked, most of it three 13 MB variants of the MediaPipe WASM binary. The page needs
  them served from somewhere; where they are stored and who fetches them is the question.
  a. Committed under `webapp/vendor/`. Rejected: binaries in git, and a version bump is a large diff.
  b. **A manifest in git, files fetched on demand into a gitignored cache, served by g7.** The manifest
     names each package, its exact version and its integrity hash, the same ones npm's registry
     publishes. A `make` target downloads what is missing and checks the hashes. This is
     `ModelManager.ensure_model` applied to JS, and the face model itself can come through the same
     door: g7 already holds it in `~/.mediapipe/models/`.
  c. Loaded from jsdelivr at runtime, pinned by version and by subresource integrity hash.
     Nothing stored anywhere, but every client needs the internet, and so does every test.
  Inside b, who does the fetching is the Node question: `package.json` plus `package-lock.json` and
  `npm ci` if g7 has Node, or a short Python fetcher over the registry if not.
  Recommended: b. c is fine for a throwaway prototype, and `cdn.jsdelivr.net` is allowed in the cloud
  environment since 2026-10-22.
  ANS: ...
- Q5: **Is `make e2e` part of `make check`?** The probe ran a fake-camera session in seconds, but
  MediaPipe's WASM path was around 90 ms a frame there, so a frame-exact run over a whole clip is not.
  ANS: no, `make e2e` is a separate target.
- Q6: **Where does the webapp live?** `webapp/` at the root beside `src/`, with its server and the
  golden-vector export in `src/abyss/webapp/`.
  ANS: modular. The phone webapp is the first UI, not the only one: more UIs and more renderers may
  follow, so the layout has to leave room for siblings rather than name one of them the webapp.
  Shape to settle when the first file is written, e.g. `ui/phone_web/` for the page and
  `src/abyss/ui/` for what serves it, next to the existing `src/abyss/render/` seam.
- Q7: **Where does a test clip live?** A clip recorded on the Pixel's front camera at phone distance,
  in portrait, is the fixture none of the existing clips is, and also how the Pixel's focal length gets
  measured. It is a face, so it is never on a public GitHub. Options, which combine:
  a. **Outside git on g7**, in `~/data/pose/` beside `face01.mp4`, which is the existing convention.
     Tests that need it skip where it is absent. Cloud sessions do not have it.
  b. **Numbers extracted from it, committed.** The precedent is `tests/viewer/data/face01_samples.json`:
     iris positions and head pose for 20 frames of `face01.mp4`, extracted once so the maths is
     testable with no clip and no model. Per-frame eye positions are not a face.
  c. **A synthetic clip generated at test time**: a public-domain portrait moved by ffmpeg along a
     known path, as in the probe. Runs anywhere, cloud included, and the true motion is known
     exactly, which a recorded clip never gives.
  d. **A private store** reachable from cloud sessions: a private repository, or g7 behind the
     Cloudflare tunnel with an access token. Only needed if a cloud session must run the real clip.
  Recommended: a plus b plus c. c is the fixture every run has, a is the real-world check on g7,
  b carries what a needs into places a cannot go. d only if that turns out not to be enough.
  ANS: ...
