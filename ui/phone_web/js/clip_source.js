// A clip as the source instead of the camera: `?source=/clips/<name>.mjpeg`.
//
// Fetches the MJPEG written by abyss.ui.clip and splits it into its JPEG
// frames, so each frame is seen exactly once and in order, and the bytes are
// the ones the Python reference decodes. Port of abyss.ui.clip.split_mjpeg.

const SOI = [0xff, 0xd8, 0xff];

export function splitMjpeg(bytes) {
  const starts = [];
  for (let i = 0; i + 2 < bytes.length; i++) {
    if (bytes[i] === SOI[0] && bytes[i + 1] === SOI[1] && bytes[i + 2] === SOI[2]) starts.push(i);
  }
  return starts.map((start, k) => bytes.subarray(start, starts[k + 1] ?? bytes.length));
}

export async function loadClip(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const err = new Error(`clip ${url}: ${response.status}`);
    err.name = "ClipError";
    throw err;
  }
  return splitMjpeg(new Uint8Array(await response.arrayBuffer()));
}

export function decodeFrame(jpeg) {
  return createImageBitmap(new Blob([jpeg], { type: "image/jpeg" }));
}
