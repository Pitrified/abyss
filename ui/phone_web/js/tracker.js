// The face landmarker, from the vendored tasks-vision and the model the
// Python pipeline uses. Same settings as abyss.ui.reference: VIDEO mode, one
// face, transformation matrices on.

import { FaceLandmarker, FilesetResolver } from "/vendor/@mediapipe/tasks-vision/vision_bundle.mjs";

export async function createLandmarker({ delegate = "CPU" } = {}) {
  const fileset = await FilesetResolver.forVisionTasks("/vendor/@mediapipe/tasks-vision/wasm");
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });
}

// VIDEO mode needs strictly increasing timestamps; clamp to at least 1 ms
// apart, as abyss.loop does.
export function monotonicClock() {
  let last = -1;
  return (msec) => {
    last = Math.max(msec, last + 1);
    return last;
  };
}
