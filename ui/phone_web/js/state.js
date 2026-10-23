// The page's state, published on `window.abyss` so a test or the drive script
// reads what the page knows rather than guessing from pixels.
//
// status:  "starting" | "running" | "done" | "error"
// source:  "camera" or the clip URL
// camera:  { label, width, height } once frames are known
// frames:  frames processed
// error:   { name, message } when status is "error"
// latest:  { sample, position, smoothed } for the newest frame with a face
// scale:   { ready, collected, needed, value }
// track:   clip mode only: one { idx, msec, sample, position, smoothed } per frame
// timing:  per-stage frame times in ms, last and rolling median; see timing.js
// output:  { width, height, pixelRatio } of the scene, with a screen
// display: { fullscreen, lock }; see display.js
// version: the served UI's hash, from /version.js, also shown in the corner
// log:     the newest LOG_LINES lines of what the page did; see log()

export const state = {
  status: "starting",
  source: null,
  camera: null,
  frames: 0,
  error: null,
  latest: null,
  scale: null,
  track: [],
  log: [],
};

const LOG_LINES = 500;

window.abyss = state;

export function setStatus(text) {
  const line = document.getElementById("status");
  if (line) line.textContent = text;
}

// One line of the log a person copies off the device: events as they happen,
// and a summary a second from the camera loop. Seconds since navigation, so
// the time the modules took to arrive shows as the first line's stamp.
export function log(text) {
  const seconds = (performance.now() / 1000).toFixed(2);
  state.log.push(`${seconds} ${text}`);
  if (state.log.length > LOG_LINES) state.log.shift();
}

// Name the load phase for boot.js's counter and log it; null stops the counter.
export function loading(phase) {
  window.abyssLoading = phase;
  if (phase) log(`loading ${phase}`);
}

// Every file over 50 kB, with its size, what crossed the network (less when
// gzipped, headers only when the browser's copy was still current), and how
// long it took: on a slow link, which file the wait is in.
export function logDownloads() {
  const kb = (bytes) => (bytes / 1000).toFixed(0);
  for (const entry of performance.getEntriesByType("resource")) {
    if (entry.decodedBodySize < 50_000) continue;
    const start = (entry.startTime / 1000).toFixed(2);
    const took = (entry.duration / 1000).toFixed(2);
    const size = `${kb(entry.decodedBodySize)} kB, ${kb(entry.transferSize)} kB on the wire`;
    log(`fetched ${new URL(entry.name).pathname} ${size}, started ${start} s, took ${took} s`);
  }
}

export function fail(err) {
  loading(null);
  state.status = "error";
  state.error = { name: err?.name ?? "Error", message: err?.message ?? String(err) };
  setStatus(`${state.error.name}: ${state.error.message}`);
  log(`error ${state.error.name}: ${state.error.message}`);
  console.error(`abyss: ${state.error.name}: ${state.error.message}`);
}
