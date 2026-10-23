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
| `/probe/` | `ui/probe/`, what a device reports about itself, as JSON to copy |
| `/vendor/<npm name>/<path>` | `cache/vendor/`, at the version `ui/vendor.toml` pins |
| `/models/face_landmarker.task` | the model the Python pipeline loads, through pose-tools' `ModelManager` |
| `/clips/<name>` | `build/clips/` |
| `/version.js` | `export const VERSION`, a hash of every file under `ui/`, computed per request |
| `/config.json?camera=&viewer=&screen=` | device configs from `abyss_devices.py` and the constants of the maths |
| `/scene.json?screen=` | the scene for that panel, as segments from `abyss.render.scene.window_box` |

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

The page ports the eye position maths to JavaScript, but none of its numbers: device configs and
constants (MediaPipe's assumed field of view, the eye offset in the face model, the bootstrap
length, the filter width) come from `/config.json`, built from the Python modules that define them.

## Page parameters

| Parameter | Effect |
| --- | --- |
| `source=/clips/<name>.mjpeg` | read the clip instead of the camera, one inference per frame, in order |
| `camera=<key>` | camera registry key; default `pixel7pro_front`. Use `unknown_clip` for the synthetic clip |
| `viewer=<key>` | viewer registry key |
| `screen=<key>` | screen registry key; draws the scene for that panel through the off-axis projection |
| `eye=<x,y,z>` | no tracking: draw one frame for this camera-frame eye, in metres; needs `screen=` |
| `size=<WxH>` | output size, one drawn pixel per CSS pixel, for tests that read frames back; by default the picture is the largest with the panel's aspect that fits the window, refits when the window changes, and draws at `devicePixelRatio` |
| `scale=<float>` | freeze the head scale instead of bootstrapping it |
| `delegate=GPU` | MediaPipe delegate; CPU by default |
| `fps=<float>` | the clip's frame rate, for VIDEO-mode timestamps; default 30, which `make clip` writes |

With `source=`, the page publishes one entry per frame on `window.abyss.track` and sets
`window.abyss.status` to `done` at the end. With the camera it keeps `window.abyss.latest` current.
With `screen=`, each entry also carries the view projection `matrix` it drew with, row-major.

Every frame also updates `window.abyss.timing`: `last` and a rolling `median` over 60 frames of
`landmarker`, `eye`, `matrix` and `draw` in milliseconds, their `total`, the frame `interval`, and
`fps`. A stage a frame skipped is absent, not zero. `draw` is the WebGL calls only; a slow GPU shows
in `interval`. With the camera, the status line shows the fps. A tap anywhere enters fullscreen and
tries the portrait lock; `window.abyss.display` says whether it is fullscreen and what the lock did
(`"locked"` or the error name; desktop Chromium refuses it).

The top-right corner shows the served UI's version, `ui <hash>`, from `/version.js`. Every response
is `no-store`, so it should always match `curl -s <server>/version.js`; an older hash on a device
means it is running a stale copy. Next to it, the **log** button copies a header (version, URL,
user agent, camera, output, display, median timings) and `window.abyss.log` to the clipboard: the
newest 500 lines of events (config, camera, output size, scale ready, fullscreen, lock, errors) and
one summary line a second with the eye, fps and stage medians. If the clipboard is refused it shows
the text selected instead. The button does not count as the fullscreen tap.

While the page loads, the status line counts seconds and names the phase (`modules`, `config`,
`scene`, `landmarker`, `camera`); `js/boot.js` runs it, a classic script loaded before the modules,
since nothing in `app.js` runs until three.js and tasks-vision have arrived. Log times are seconds
since navigation, so the first line's stamp is how long the modules took. After `landmarker ready`
the log lists every download over 50 kB with its size, start and duration: the time between the
`.wasm` finishing and `landmarker ready` is compiling and setup, not the network.

The scene is drawn by three.js from the served segments. three.js's own cameras cannot express an
off-axis frustum, so a bare `THREE.Camera` at the origin carries abyss's whole view projection as
its projection matrix. `Matrix4.set` takes row-major arguments; passing the matrix to `fromArray`
instead transposes it, and only the picture tests in `tests/e2e/test_page_scene.py` notice.

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
