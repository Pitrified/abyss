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

export const state = {
  status: "starting",
  source: null,
  camera: null,
  frames: 0,
  error: null,
  latest: null,
  scale: null,
  track: [],
};

window.abyss = state;

export function setStatus(text) {
  const line = document.getElementById("status");
  if (line) line.textContent = text;
}

export function fail(err) {
  state.status = "error";
  state.error = { name: err?.name ?? "Error", message: err?.message ?? String(err) };
  setStatus(`${state.error.name}: ${state.error.message}`);
  console.error(`abyss: ${state.error.name}: ${state.error.message}`);
}
