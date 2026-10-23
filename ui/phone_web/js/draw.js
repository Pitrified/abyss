// What a person sees: the frame, and a dot on the eye midpoint.

export function drawFrame(canvas, bitmap) {
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
}

// In camera mode the canvas only carries the overlay, sized to the video.
export function drawEye(canvas, sample, size) {
  const g = canvas.getContext("2d");
  if (size && (canvas.width !== size.width || canvas.height !== size.height)) {
    canvas.width = size.width;
    canvas.height = size.height;
  }
  if (size) g.clearRect(0, 0, canvas.width, canvas.height);
  if (!sample) return;
  g.fillStyle = "#0f0";
  g.beginPath();
  g.arc(sample.u_px, sample.v_px, 6, 0, 2 * Math.PI);
  g.fill();
}
