// Fullscreen and the portrait lock, on a tap. Both need a user gesture, and
// the lock only works in fullscreen. Without them the browser's bars take
// part of the panel and the picture is drawn at the wrong scale for the glass.
//
// Publishes state.display: { fullscreen, lock }, where `lock` is null until
// tried, then "locked" or the name of the error that refused it.

import { state } from "./state.js";

export async function enterFullscreen(target = document.documentElement) {
  if (document.fullscreenElement) return { fullscreen: true, lock: "already fullscreen" };
  try {
    await target.requestFullscreen({ navigationUI: "hide" });
  } catch (err) {
    return { fullscreen: false, lock: `not tried: ${err.name}` };
  }
  try {
    await screen.orientation.lock("portrait");
    return { fullscreen: true, lock: "locked" };
  } catch (err) {
    return { fullscreen: true, lock: err.name };
  }
}

export function fullscreenOnTap() {
  state.display = { fullscreen: false, lock: null };
  document.addEventListener("fullscreenchange", () => {
    state.display.fullscreen = Boolean(document.fullscreenElement);
  });
  document.addEventListener("click", async () => {
    if (document.fullscreenElement) return;
    const { lock } = await enterFullscreen();
    state.display.lock = lock;
  });
}
