// The front camera, opened into a <video> element.

// Ask for portrait at phone resolution. A fake device or a laptop webcam
// answers with whatever it has, and the page reads the size it actually got.
const CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: "user",
    width: { ideal: 720 },
    height: { ideal: 1280 },
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
