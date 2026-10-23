// The log button: copies a header and window.abyss.log to the clipboard, so a
// person on the phone can paste what the page saw into a chat. If the
// clipboard is refused, the text is shown selected for a manual copy.
//
// Publishes state.copied: the number of lines copied, or the error name.

import { state } from "./state.js";

function report() {
  const header = [
    `abyss ${state.version}`,
    `url ${location.href}`,
    `agent ${navigator.userAgent}`,
    `camera ${JSON.stringify(state.camera)}`,
    `output ${JSON.stringify(state.output ?? null)}`,
    `display ${JSON.stringify(state.display ?? null)}`,
    `timing ${JSON.stringify(state.timing?.median ?? null)} fps ${state.timing?.fps ?? null}`,
    `status ${state.status} ${JSON.stringify(state.error)}`,
  ];
  return [...header, "", ...state.log].join("\n");
}

export function logButton() {
  const button = document.getElementById("copy-log");
  const fallback = document.getElementById("log-out");
  button.addEventListener("click", async (event) => {
    // Not a tap for fullscreen.
    event.stopPropagation();
    const text = report();
    try {
      await navigator.clipboard.writeText(text);
      state.copied = state.log.length;
      button.textContent = `copied ${state.copied}`;
    } catch (err) {
      state.copied = err.name;
      fallback.value = text;
      fallback.hidden = false;
      fallback.select();
    }
  });
  fallback.addEventListener("click", (event) => event.stopPropagation());
}
