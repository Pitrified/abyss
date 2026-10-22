import { eachFrame, openCamera } from "./camera.js";
import { fail, setStatus, state } from "./state.js";

async function main() {
  const video = document.getElementById("camera");
  state.camera = await openCamera(video);
  state.status = "running";
  const { width, height } = state.camera;
  eachFrame(video, () => {
    state.frames += 1;
    setStatus(`${width}x${height}, frame ${state.frames}`);
  });
}

main().catch(fail);
