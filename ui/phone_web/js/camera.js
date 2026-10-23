// The front camera, opened into a <video> element.

// Android Chrome reads width and height in the sensor's landscape terms and
// swaps them for a phone held upright. Asked for 720x1280, the Pixel 7 Pro
// cropped a landscape 1280x720 out of the middle of the picture; asked for
// 1920x1440, it delivers 1440x1920 uncropped, the sensor's full width (probe
// sweep, 2026-10-23). A laptop webcam or a fake device answers with the
// nearest it has, and the page reads the size it actually got. The probe
// page asks with the same constraints, to report what this gets.
export const CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: "user",
    width: { ideal: 1920 },
    height: { ideal: 1440 },
  },
};

export async function openCamera(video) {
  if (!navigator.mediaDevices?.getUserMedia) {
    const err = new Error("no camera API: the page needs https or localhost");
    err.name = "InsecureContextError";
    throw err;
  }
  const stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
  video.srcObject = stream;
  await video.play();
  const track = stream.getVideoTracks()[0];
  return { label: track.label, width: video.videoWidth, height: video.videoHeight };
}

// Call `onFrame` once per presented video frame.
export function eachFrame(video, onFrame) {
  const tick = (now, meta) => {
    onFrame(now, meta);
    video.requestVideoFrameCallback(tick);
  };
  video.requestVideoFrameCallback(tick);
}
