// A camera bound to the frames it produced. Port of
// abyss.config.camera.CameraConfig.focal_px_for_height, .focal_fits_frame and
// FrameGeometry.
//
// `camera` is a CameraConfig as served at /config.json, `constants` the
// constants block from the same document.

export function focalPxForHeight(camera, height, constants) {
  if (!(height > 0)) throw new RangeError(`frame height must be positive, got ${height}`);
  if (camera.focal_px != null && camera.focal_measured_at_height != null) {
    // Focal length in pixels scales with the sampling of the sensor.
    return (camera.focal_px * height) / camera.focal_measured_at_height;
  }
  const fov = camera.fov_vertical_deg ?? constants.mediapipe_vertical_fov_deg;
  return height / 2 / Math.tan(((fov / 2) * Math.PI) / 180);
}

// False only when the measured shape is known and the frame's differs: a
// different crop, which the rescaled focal does not describe.
export function focalFitsFrame(camera, width, height) {
  const { focal_px: focal, focal_measured_at_width: w, focal_measured_at_height: h } = camera;
  if (focal == null || w == null || h == null) return true;
  return width * h === height * w;
}

export function frameGeometry(camera, width, height, constants) {
  if (!(width > 0) || !(height > 0)) {
    throw new RangeError(`frame size must be positive, got ${width}x${height}`);
  }
  return {
    width,
    height,
    focal: focalPxForHeight(camera, height, constants),
    principalPoint: [width / 2, height / 2],
    mirrored: camera.mirrored,
  };
}
