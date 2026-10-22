---
status: planned
---

# Phase 1 - harness and skeleton page

## Overview

Build the lever before the feature. At the end of this phase there is a page that shows the camera
and nothing else, and an agent can serve it, open it in Chromium with a clip as the camera, read
what it reports, and screenshot it, with one command. Every later phase is checked through this.

Context: [`00_start.md`](00_start.md) Q4-Q7, and the probe in
[`01_research_dev_harness.md`](01_research_dev_harness.md), which this phase turns from a throwaway
script into the repo's own tooling.

## What gets built

**The page**, `ui/phone_web/` (placeholder name, Q6): `index.html` and one ES module. Requests the
front camera, shows it full screen, and publishes its state on `window.abyss` (camera label, frame
size, frames seen, last error) so a test reads state rather than pixels. No framework, no bundler.

**Third-party files** (Q4 b):

- A manifest in git listing each package, exact version, and the integrity hash npm's registry
  publishes for its tarball, plus the files the page needs out of it. First entries:
  `@mediapipe/tasks-vision` and `three`, even though phase 1 imports neither, so the fetch path is
  proven before anything depends on it. The face model goes through the same manifest, by URL and
  sha256.
- `make vendor` downloads what is missing into a gitignored cache, checks every hash, and refuses
  to serve a file that does not match.
- Who fetches is decided here, by checking g7: `npm ci` over a `package-lock.json` if g7 has Node,
  otherwise a short Python fetcher in `src/abyss/ui/`. Recommended: the Python fetcher regardless, so
  `make vendor` behaves the same on every machine and in cloud sessions.
- The manifest cannot be `.json` without a `.gitignore` exception, since `*.json` is ignored. TOML
  avoids that and reads with the standard library.

**The server**, `src/abyss/ui/serve.py`: static files plus the vendor cache, on `localhost` by
default. `make serve`. Binding the tailnet address is phase 4.

**The synthetic clip** (Q7 c): a generator that takes a public-domain portrait and moves it along a
known path with ffmpeg, writing `.y4m` (and `.mp4` for `?source=`, phase 2) into a gitignored
`build/` folder. The portrait is fetched by URL and hash, like a vendored file. The path is a
parameter; the default covers left-right, up-down, and a scale change for approach and retreat. The
probe's lesson is built in: the face is at least 400 px across at 1280x720, and a test asserts the
generator never produces less.

**Playwright**, for Python, in a new `e2e` dependency group. Launches the Chromium the environment
has (`/opt/pw-browsers` in cloud sessions, the system one or `playwright install chromium` on g7),
with the three fake-media flags.

**`make e2e`** (Q5), separate from `make check`. Generates the clip if missing, runs `make vendor`,
starts the server, runs `tests/ui/` headless. `pytest` skips `tests/ui/` unless it is asked for, so
`make test` is unchanged.

**`scripts/drive_ui.py`**, the instrument from klide's `drive.py`: serves the page, opens it, folds
the browser console, page errors and the server log into one time-ordered stream, takes screenshots
into `build/ui/` on request. Not a gate.

## Tests

- **Fake camera reaches the page.** The page reports the clip's frame size and a frame count that
  increases.
- **Vendored files are served and match.** Every manifest entry is fetched through the server and
  its hash checked; a corrupted cache file is refused.
- **The clip is what it claims.** Frame size, frame count, and the minimum face size.
- **The failure path.** No camera available: the page reports the error on `window.abyss` instead
  of hanging.

Each one shown failing once before it counts, by breaking what it guards.

## Out of scope

Tracking, the scene, HTTPS, any device but headless Chromium.

## Done when

- `make e2e` is green in a cloud session and on g7, headless, from a clean clone after `make sync`.
- `make check` is unchanged and green.
- `scripts/drive_ui.py` produces a screenshot of the page showing the synthetic clip.
- `docs/guides/` has a page for `make vendor`, `make serve` and `make e2e`, and
  `.github/copilot-instructions.md` mentions the harness, so the next session finds it.
