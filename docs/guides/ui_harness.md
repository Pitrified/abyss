# The UI harness

How to serve the browser UIs, run them in Chromium with a recorded face as the camera, and watch
what they do. Everything here runs headless, so it works on g7 unattended and in a cloud session.

## Commands

```bash
make vendor   # fetch and hash-check three.js, tasks-vision and the test portrait
make clip     # write build/clips/synthetic_face.mjpeg
make serve    # serve the UIs on http://127.0.0.1:8000/
make e2e      # the browser tests; not part of `make check`
```

`make e2e` needs a Chromium. In a cloud session one is already installed under
`PLAYWRIGHT_BROWSERS_PATH` and is found automatically. On g7, once:

```bash
uv run --no-sync playwright install chromium
```

`ABYSS_CHROMIUM=/path/to/chrome` overrides both.

## What gets served

| URL | From |
| --- | --- |
| `/phone_web/` | `ui/phone_web/`, the page itself |
| `/vendor/<npm name>/<path>` | `cache/vendor/`, at the version `ui/vendor.toml` pins |
| `/models/face_landmarker.task` | the model the Python pipeline loads, through pose-tools' `ModelManager` |
| `/clips/<name>` | `build/clips/` |

`ui/vendor.toml` is the only third-party thing in git. Each npm package is pinned by version and by
the integrity hash the registry publishes; `make vendor` checks every tarball against it and records
a sha256 for every extracted file. The server checks each file against that record on every request
and answers 500 rather than serve one that changed. To bump a package, change its `version` and
`integrity` (both from `npm view <name>@<version> dist.integrity`, or from
`https://registry.npmjs.org/<name>/<version>`), then run `make vendor`.

Every response carries a Content-Security-Policy. `connect-src 'self'` is the line that matters:
`@mediapipe/tasks-vision` posts usage logs to `odml.pa.googleapis.com` every minute, and the
policy is what stops it. The page cannot load or send anything to any host but the one serving it,
and `make e2e` fails if it tries.

## The camera

Chromium's fake capture device plays a file as the webcam, through the same `getUserMedia` path a
real one takes. The tests and the drive script launch Chromium with it pointed at
`build/clips/synthetic_face.mjpeg`: a public-domain portrait moved along a known path (left-right,
up-down, closer), 720x1280 like a phone held upright, four seconds, looped. The path is in
`abyss.ui.clip.face_path`, so a test knows where the eyes are in every frame.

The clip is regenerated on every run (about a second), so it can never disagree with the code that
describes it.

## Watching a page

`scripts/drive_ui.py` serves, opens a UI with the clip as the camera, runs steps, and prints the
browser console, page errors, failed requests and the server log as one time-ordered stream:

```bash
uv run --no-sync python scripts/drive_ui.py \
    --do "until window.abyss.frames > 20" --do state --do "shot running"
```

Screenshots go to `build/ui/`. `--headed` opens a window, which needs a display; `--no-camera`
launches with no fake device; `--viewport` defaults to a Pixel 7 Pro held upright.

Use it when `make e2e` fails and the question is where between the camera and the canvas the thing
stopped. It is not a test: nothing it prints is compared against anything.

## Reading the page from outside

The page publishes its state on `window.abyss`, so a test reads what the page knows rather than
guessing from pixels. Wait on it with `abyss.ui.browser.wait_until`, not Playwright's
`wait_for_function`: the latter evaluates its predicate with `eval` inside the page, and the
page's policy correctly refuses that.
