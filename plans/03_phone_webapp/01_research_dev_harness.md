---
status: draft
---

# Dev harness research - an agent that can run the webapp

How an agent working on this repo gets to run the phone webapp for real, and how a recorded face
stands in for the person in front of the camera. Gathered before building anything. Proposals here
become numbered questions in [`00_start.md`](00_start.md) rather than being answered in this file.

Two sibling repos were read for the pattern, on 2026-10-22, at their `main`:
[`klide`](https://github.com/Pitrified/klide) and
[`fala-language-tutor`](https://github.com/Pitrified/fala-language-tutor).

## What the two repos do

Both reached the same conclusion from opposite ends: an agent that cannot run the thing ships things
that pass every check and do not work.

### klide

A Claude Code transcript rendered for an e-ink panel. The host renders, the device is a frame sink.

- **A simulator of the device**, headless, including its costs: page cache, refresh modes, sleep.
  The real device is not in hand, so the simulator is the thing every check runs against.
- **Gates with a contract** (`meta/gates.md`): one command (`scripts/check.sh`), the same command in
  CI, a failure that names the file and line, fast enough that an agent runs it, and demonstrated
  failing before it counts. Frame, session, view and walk gates compare rendered output against
  committed references, at zero tolerance.
- **`scripts/drive.py`**, a Playwright instrument that opens the viewer page in Chromium and clicks,
  taps, screenshots and reads the DOM. It folds the browser console, the viewer's log and the host's
  log into **one time-ordered stream**, which is what found a page-turn button that did nothing while
  every server-side probe said the path worked. Deliberately not a gate.
- **A cross-implementation check.** The viewer is a second implementation of the wire protocol,
  written from the spec. Duplication is accepted only because a test decodes the viewer's output
  against klide's encoder and pins its constants, so drift fails a gate.
- **A diary review** (`meta/diary/2026-09-18-...`) whose findings apply here directly:
  - *green gates over content that never exercises the feature*: text at 6.2 pt passed three phases
    because the fixture had no small text. Found by a person reading a screen.
  - *correct on this machine, wrong on the one that matters*: a design made where viewer and browser
    were the same host never saw the network cost of a frame.
  - *the weakest check that could pass*: `import tkinter` instead of `Tk()`.
- **`principles` skill**: prove it works against the real artifact, build the lever, sequence
  verifiable units, encode lessons in structure.

### fala-language-tutor

A Flutter Android app, streaming chat against OpenAI.

- **A fake behind the real interface.** `FakeInferenceEngine` for unit tests, and a separate
  `tool/mock_openai.py` that speaks the real HTTP API with **scenarios** (ok, malformed, 401, 429,
  slow) switchable at runtime, and logs every request as JSON lines so a test asserts what the app
  actually sent.
- **`scripts/e2e.sh`**: boots an Android emulator headless (`-no-window -gpu swiftshader_indirect`),
  starts the mock, runs one integration test as a single journey against the real app with a real
  clock. Minutes long, so not in `check.sh`.
- **`tool/adb_ui.sh`**: drives the app on an emulator or the real phone by reading the uiautomator
  view tree and tapping by label, not by pixel; screenshots via `adb exec-out screencap`.
- **A "Which machine" section** in the instructions: what the headless box can do, what needs the
  Pixel, and an instruction to hand those back rather than work around them.

### What carries over

| Pattern | Their version | Here |
| --- | --- | --- |
| a fake for the thing that is hard to have | device simulator; mock OpenAI with scenarios | a recorded face fed in as the camera |
| the real app, driven by the agent | Playwright on the viewer; emulator plus `adb_ui.sh` | Playwright on the webapp in Chromium |
| one stream of logs | `drive.py` | browser console, page events and server log, time-ordered |
| a second implementation pinned to the first | viewer decoded against the encoder | JS frustum and eye position checked against the Python ones |
| slow check kept out of the fast gate | `e2e.sh` outside `check.sh` | a `make e2e` beside `make check` |
| the fixture must exercise the feature | the 6.2 pt lesson | a clip where the head actually moves, and a face big enough to detect |

## The camera, which turned out not to be hard

Chromium has a fake capture device built for exactly this:

```text
--use-fake-ui-for-media-stream             # grant the permission prompt
--use-fake-device-for-media-stream         # replace the real devices
--use-file-for-fake-video-capture=<clip>   # .y4m or .mjpeg, looped
```

`getUserMedia` then returns the file as the camera, through the same code path a real webcam takes.
The page cannot tell.

### Measured in a cloud session, 2026-10-22

A throwaway probe, not committed: Playwright for Python driving the preinstalled Chromium 1194
headless, a page served from a local `http.server`, `@mediapipe/tasks-vision` 1.1.1 installed from
npm, `face_landmarker.task` float16 from Google's model bucket.

- **Fake camera works headless.** The track's label is the clip's path, the size is the clip's, and
  the pixels change frame to frame.
- **WebGL2 works headless**, through SwiftShader (`ANGLE ... SwiftShader driver`), so three.js has a
  context with no GPU and no display.
- **MediaPipe's web build runs and tracks.** A portrait slid across a grey 1280x720 frame at
  80 px/s: the iris landmark (468) moved 275 px in 3.44 s, which is 80 px/s, on every sample.
  Around 90 ms per `detectForVideo` call on the WASM CPU path in that container, which is a
  container number and not a phone or g7 number.
- **A small face is not found.** The first clip had the face about 130 px across in 1280x720 and
  MediaPipe returned nothing for all twelve samples. Scaled up to about 400 px it tracked every
  sample. A test clip has to be framed like someone holding a phone, not someone across a room.
- **jsdelivr was blocked** by that session's proxy; the npm registry and
  `storage.googleapis.com` were not. Vendoring from npm, not loading from a CDN at runtime, avoids
  the question everywhere.

### Two ways in, both needed

1. **The fake device**, above. Proves the real `getUserMedia` path, including permissions and
   constraints. Not frame-exact: the clip plays in real time and loops, so which frame a sample sees
   depends on timing.
2. **A clip source inside the app**, `?source=<clip>`: the page reads frames from a `<video>`
   element and steps through them with `requestVideoFrameCallback`, one inference per frame. Frame
   exact and repeatable, and it works in any browser, the phone's included. It skips `getUserMedia`,
   which is why it does not replace the first.

The second is what a regression test compares; the first is what proves the camera path is wired.

### Clips

The repo's existing clips (`face01.mp4` and the others in `AbyssPaths.pose_fol`) are the obvious
source, and `scripts/viewer_position.py` from phase 1 already writes the Python pipeline's
per-frame eye positions for them as `<clip>_eye.csv`. That gives
the cross-implementation check for free: run the same clip through the page with `?source=`, export
the per-frame eye positions, and compare against the Python CSV within a tolerance. The two
MediaPipe builds will not agree exactly, so the tolerance is a number to measure, not to guess.

One gap: none of those clips was recorded on the Pixel's front camera, at phone distance, in
portrait. A clip recorded on the phone is a fixture worth having, and it is also how the Pixel's
focal length gets measured.

`.y4m` is uncompressed (165 MB for five seconds of 720p in the probe). Converted with ffmpeg at test
time from the `.mp4`, into `build/` or a temp folder, never committed. `.mjpeg` is the smaller
alternative Chromium also accepts.

## The rest of the harness

- **Driving the page.** Playwright for Python, under a uv dependency group, against the Chromium the
  environment already has. Same toolchain as everything else in abyss, and the same library klide's
  `drive.py` uses.
- **Secure context.** `getUserMedia` needs HTTPS or `localhost`. Tests use `localhost`. Phone and g4
  reach g7 through `tailscale serve`, which terminates HTTPS on the tailnet name; the Cloudflare
  tunnel stays the public option.
- **The phone itself.** Chrome on Android exposes DevTools over adb
  (`adb forward tcp:9222 localabstract:chrome_devtools_remote`), and Playwright can attach over CDP,
  so the agent can load the page, read the console and screenshot on the real Pixel when it is
  cabled to g7. The camera cannot be faked there; `?source=` covers tracking, and the live camera is
  a manual check, the way phase 5 was.
- **An Android emulator** would add Android Chrome's quirks and nothing else for a web page, since
  desktop Chromium already emulates viewport, pixel ratio and touch. Its camera can be fed from a
  `v4l2loopback` device playing a clip, if that is ever needed. Deferred.
- **Node.** Unknown whether g7 has it, and klide records a box without it. A page with no bundler,
  ES modules and an import map over files vendored from npm tarballs, needs no Node at runtime or in
  tests: Playwright from Python serves and drives it, and the JS unit tests run inside the browser.
  The alternative is Vite and `@playwright/test`, which is the conventional stack and a second
  toolchain.

## Shape in abyss

A proposal. Q6 in `00_start.md` asks for room for more than one UI and more than one renderer, so the
names below are placeholders for the first of several, not the layout of the only one.

```text
ui/phone_web/                the page: index.html and ES modules
  vendor/                    gitignored; third-party files fetched by manifest (Q4)
src/abyss/ui/                what serves a UI, plus the golden-vector export
tests/ui/                    Playwright tests, run by `make e2e` and not by `make test`
scripts/drive_ui.py          the klide-style instrument: one log stream, screenshots to build/
```

Make targets: `make serve` (local) and `make e2e` (the Playwright tests, headless, fake camera and
`?source=`). `make e2e` is not part of `make check` (Q5).

What `make e2e` would prove, in order of value:

1. **Frustum agreement.** Golden vectors exported from `view_projection_matrix` (eye positions and
   screen rectangles in, 4x4 matrices out), checked by the page's own JS. Exact up to float32.
2. **Tracking agreement.** `face01.mp4` through `?source=`, per-frame eye positions against the
   Python pipeline, within a measured tolerance.
3. **The camera path.** The same clip through the fake device: a face is found, the eye position
   moves, the scene redraws. Asserted loosely, because timing is not exact.
4. **A picture.** Screenshots of the rendered scene at fixed eye positions, compared against
   references the way klide's frame gate does. Least valuable at first, since the scene is going to
   change most.
