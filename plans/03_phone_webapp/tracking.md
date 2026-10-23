# implementation tracking

Head-coupled perspective on the Pixel 7 Pro, as a web page: the phone tracks the face, builds the
frustum and draws the scene, and g7 serves the page. Decisions in [`00_start.md`](00_start.md), the
dev harness pattern and the probe that shaped it in
[`01_research_dev_harness.md`](01_research_dev_harness.md).

## Key decisions

- **All on the phone, in the browser** (Q1, Q2). Tracking, frustum and drawing run in the page, with
  `@mediapipe/tasks-vision` and three.js. Nothing per frame crosses the network. Unity is the upgrade
  path if the browser runs out of performance.
- **Headless first, then g4, then the Pixel** (Q3). Each step adds one unknown.
- **The agent runs the page for real.** Playwright drives Chromium; a clip stands in for the camera,
  through Chromium's fake capture device and through an in-app `?source=` mode that steps frame by
  frame. The harness is phase 1, before any feature, because every later phase is checked with it.
- **Python stays the reference.** The page carries a second implementation of the eye position and
  frustum maths. That duplication is accepted only because a check compares the two on the same
  inputs, the way klide's viewer is pinned to klide's encoder. Device numbers stay in
  `abyss_devices.py` and are served to the page, never retyped in JS.
- **Third-party files by manifest** (Q4 b). A manifest in git, a hash-checked gitignored cache,
  served by g7. Nothing binary is committed.
- **Synthetic clip for tests** (Q7 c). A public-domain portrait moved along a known path, generated at
  test time. A real Pixel recording, and numbers extracted from it, are deferred (Q7 a+b).
- **`make e2e` is separate from `make check`** (Q5), and **the layout leaves room for more UIs and
  renderers** (Q6).

## Phases

| #  | Phase                                   | Plan | Status  |
| -- | --------------------------------------- | ---- | ------- |
| 1  | harness and skeleton page               | [`02_harness_skeleton.md`](02_harness_skeleton.md) | done    |
| 2  | tracking in the page                    | [`03_tracking_in_page.md`](03_tracking_in_page.md) | done    |
| 3  | frustum and scene in the page           | [`04_frustum_and_scene.md`](04_frustum_and_scene.md) | done    |
| 4  | real clients: g4, then the Pixel        | [`05_real_clients.md`](05_real_clients.md) | planned |

Status values: draft / planned / in progress / done / superseded / discarded.
File numbers run one ahead of phase numbers because `01_` is the research file.

Each phase ends in a state `make e2e` can check, except phase 4, whose last step is a person holding
the phone, the way the expansion's phase 5 ended with a person at g7.

Deferred, not phased:

- A clip recorded on the Pixel's front camera, kept on g7, and numbers extracted from it committed
  (Q7 a+b). Also how the Pixel's focal length gets measured.
- Driving the page on the cabled Pixel over CDP from g7 (`adb forward ... chrome_devtools_remote`).
  Useful in phase 4 if the manual check keeps needing a second look; not built before that.
- An Android emulator, Unity, Gaussian splats.

## Next steps

Where to pick up, as of 2026-10-23. Phases 1-3 were built and verified in a cloud session; nothing
has run on g7, g4 or the phone yet. In order:

1. **g7: `make e2e` once.** `make sync`, then `uv run --no-sync playwright install chromium` (the
   cloud sessions find a preinstalled Chromium; g7 has to fetch Playwright's own), then `make check`
   and `make e2e`. Expect 29 browser tests in well under a minute. Closes phase 1's pending item. If
   anything differs from the cloud run, the drive script (`scripts/drive_ui.py`) is the first tool.
2. **Phase 4 prerequisites that need no person**, on g7: the per-stage timings phase 3 promised and
   did not build, fullscreen and the canvas pixel ratio, and a provisional Pixel screen entry. The
   list, with reasons, is "Carried in from phases 1-3" in [`05_real_clients.md`](05_real_clients.md).
3. **`tailscale serve`** on g7 in front of `make serve`, and the page loaded from g4 by a person.
4. **The Pixel**, by a person, following a runbook written before the run.

Open housekeeping, not blocking: the branch is not merged to `main`. Ruff is pinned to the same
exact version in `pyproject.toml` and `.pre-commit-config.yaml`; bump both together.

## Log

Append-only. Newest at the bottom.

- 2026-10-22 : branched `feat/phone-webapp`. Read `klide` and `fala-language-tutor` for how an agent
  runs the app it builds. Probe in a cloud session: Chromium's fake capture device plays a clip as the
  webcam, headless; WebGL2 works through SwiftShader; MediaPipe's web build tracked a portrait moving
  at 80 px/s at exactly 80 px/s, and found nothing when the face was 130 px across in 1280x720.
- 2026-10-22 : Q1-Q7 answered. Q4 b, Q5 separate target, Q6 modular, Q7 c with a+b deferred.
  Four phases planned.
- 2026-10-22 : **phase 1 done**, in a cloud session. `make e2e` green from a clean clone, 5 tests in
  about 5 s. MJPEG from OpenCV instead of `.y4m` from ffmpeg; the face model served from
  `ModelManager` rather than pinned twice; tasks-vision held at 1.0.1 and found to post usage logs
  to Google, now blocked by a CSP and checked by a test. Still to do: run `make e2e` on g7 once,
  after `uv run --no-sync playwright install chromium`.
- 2026-10-22 : **phase 2 done.** The page tracks the face and computes the eye position with a port
  of the Python modules, and agrees with Python on every frame of the clip to 0.20 px, 0.30 mm and
  0.054 deg. The JS matrix is column-major. VIDEO mode lags fast motion by up to 10.5 px in both
  builds alike. tasks-vision's usage log is now seen attempted and refused.
- 2026-10-22 : **phase 3 done.** The page builds the off-axis frustum from the smoothed eye and
  three.js draws the minimal scene through it, served as segments from `window_box` rather than
  ported. Matrices agree with Python to 1e-9 relative; the drawn frame is read back and checked at
  Python-projected points, which is the only check that catches a transposed handoff. Phases 1-3
  are done in a cloud session; `make e2e` on g7 and phase 4 (g4, then the Pixel, by hand) remain.
- 2026-10-23 : handover written for a g7 session: "Next steps" above, and the items phases 1-3 left
  for phase 4 listed in `05_real_clients.md`, including the stage timings phase 3 did not build.

- 2026-10-23 : **first run on g7.** `make check` green, 294 unit tests. `make e2e` 28 of 29: the
  no-camera test assumed that no fake device means no camera, which holds in the cloud but not on g7,
  where the Chicony is on `/dev/video0`. Headless Chromium then refuses the request with
  `NotSupportedError` (the permission prompt cannot be shown) instead of `NotFoundError`.
  `--deny-permission-prompts` changes nothing. The page reported the error correctly; the test now
  accepts that name too, and 29 of 29 pass in about 17 s. Playwright's Chromium 1243 was already
  installed. Closes phase 1's pending item.
- 2026-10-23 : ruff pinned to 0.17.2 exactly, in `pyproject.toml` and pre-commit alike, and the
  repo reformatted with it. `pre-commit run --all-files` also flags trailing blank lines in three
  plan files and `style.css`, and three vendor hashes in `ui/vendor.toml` as secrets; the hooks are
  not installed in this clone, which is how those got in. Left as found.
- 2026-10-23 : **phase 4 prerequisites built on g7**, each test seen failing first:
  `window.abyss.timing` (landmarker, eye, matrix, draw, total, interval, fps, rolling median of 60),
  fullscreen and the portrait lock on a tap, the scene refitting to the window and drawing at
  `devicePixelRatio`, a `pixel7pro` screen from the spec sheet and AOSP's cutout path, and a probe
  page at `/probe/` that reports what the device says, ready to copy. 295 unit tests, 37 browser
  tests.
- 2026-10-23 : phase 4 runbook written, `docs/guides/phone_runbook.md`: serving, the probe, the
  mirroring check (head to your right must give negative x), g4, then the Pixel, and a log template.
  On the Pixel the stage split needs `chrome://inspect` over USB; fps is on the status line.
- 2026-10-23 : pre-commit hooks installed on g7 (`make hooks`, new), and documented as a setup step.
  The trailing blank lines are fixed, and `ui/vendor.toml` is excluded from detect-secrets alongside
  `uv.lock`, for the same reason: its integrity hashes are public. `pre-commit run --all-files`
  passes.
- 2026-10-23 : **`tailscale serve` up on g7**: `https://g7.tail5dff8b.ts.net/`, tailnet only, `--bg`,
  proxying to `make serve` on `127.0.0.1:8000`. Ports 8000 and 443 checked free first; the
  Cloudflare tunnel uses 8090-8092. Serve had to be enabled for the tailnet, and the config needs
  root (run by the user; operator left unset). Headless Chromium on g7 through that URL: the page
  runs on the fake camera at 30 fps, landmarker median 25.6 ms, the rest under 0.3 ms, nothing
  off-server; the probe reports a secure context and a 720x1280 stream. Undo with
  `tailscale serve --https=443 off`, as root.
- 2026-10-23 : **phase 4, Pixel 7 Pro, probe** (runbook section 2), over the tailnet. Android 17,
  Chrome 155, Mali-G710. Stream **1280x720, landscape, with the phone upright**, 30 fps (60 max),
  `crop-and-scale`, label "camera 1, facing front": whether the content is upright-but-wide or
  rotated is not yet known. Screen 412x892 CSS at a pixel ratio of 2.625, so 1082x2342 drawn: the
  phone is on its 1080x2340 resolution setting, not the 1440x3120 maximum. The registry's sizes are
  in metres, so `pixel7pro` holds either way; only sharpness changes. Fullscreen and lock: inner
  window 411x794 before the tap, 411x891 after, `locked`, bottom safe area 24 px gone, so the page
  gets the whole panel, at an aspect within 0.05% of the panel's.
- 2026-10-23 : **phase 4, Pixel 7 Pro, mirroring** (runbook section 3). Head 10 cm to the viewer's
  right: x went 0 to +10 cm, y did not move, face upright in the view. So the landscape stream is
  upright, not rotated, and it is unmirrored: the face left the unflipped view by its left edge.
  `pixel7pro_front` was `mirrored=True` and double-flipped x, as the carried-in list feared; now
  `False`, with that provenance. 295 unit and 37 browser tests green.
- 2026-10-23 : **phase 4, Pixel 7 Pro, the effect** (runbook section 5), held in the hand, with the
  corrected flag. Head to the right now gives negative x; the face still leaves the unflipped view by
  its left edge, as it should. The effect looks natural and correct, the cyan frame stable on the
  glass's edge. CPU delegate 29.9 to 30 fps, GPU delegate works and holds 30: the camera's 30 fps is
  the ceiling either way, so the 15 fps floor holds with room. Stage split and depth not yet read.
- 2026-10-23 : **phase 4, Pixel 7 Pro, stage split**, from the log button, GPU delegate, version
  `ui d8177758` as served. Medians in ms: landmarker 28.3, eye 0.0, matrix 0.0, draw 0.1, interval
  33.3, 30.0 fps. The landmarker takes 85% of the camera's 33 ms frame, so 30 fps holds with about
  5 ms to spare and a 60 fps stream would not keep up. First GPU frame 931 ms, a one-off warm-up.
  The run was not fullscreen (no tap): picture 366x793 CSS, letterboxed. Depth reads about 14.5 cm
  held in the hand, which looks too close by about half or more: with a 1280x720 upright crop the
  frame's vertical field of view is likely far narrower than the 63 deg MediaPipe assumes, so `z`
  comes out short. To confirm with a tape.
- 2026-10-23 : asked for after the Pixel run: a version label (`/version.js`, a hash of `ui/`) to rule
  out a stale page, and a log button that copies `window.abyss.log` for pasting into a chat.
