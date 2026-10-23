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
| 1  | harness and skeleton page               | [`02_harness_skeleton.md`](02_harness_skeleton.md) | done, g7 run pending |
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

Open housekeeping, not blocking: the branch is not merged to `main`; the venv resolves ruff 0.16.2
while pre-commit pins 0.15.2 (format new files with `uvx ruff@0.15.2 format`, or settle the pin).

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

