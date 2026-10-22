// The page's state, published on `window.abyss` so a test or the drive script
// reads what the page knows rather than guessing from pixels.
//
// status: "starting" | "running" | "error"
// camera: { label, width, height } once a stream is open
// frames: video frames presented since the stream opened
// error:  { name, message } when status is "error"

export const state = {
  status: "starting",
  camera: null,
  frames: 0,
  error: null,
};

window.abyss = state;

const statusLine = () => document.getElementById("status");

export function setStatus(text) {
  const line = statusLine();
  if (line) line.textContent = text;
}

export function fail(err) {
  state.status = "error";
  state.error = { name: err?.name ?? "Error", message: err?.message ?? String(err) };
  setStatus(`${state.error.name}: ${state.error.message}`);
  console.error(`abyss: ${state.error.name}: ${state.error.message}`);
}
