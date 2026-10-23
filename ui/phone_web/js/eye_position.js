// Eye position in the camera frame, from a face landmarker result. Port of
// abyss.viewer.eye_position; read that module for the reasoning.
//
// Frame convention, as in Python: origin at the optical centre, +X right of
// the image, +Y down the image, +Z away from the camera, metres.

// MediaPipe's flat matrix data is column-major, the default layout of its
// MatrixData proto: element (row, col) is data[col * rows + row].
export function matrixAt(matrix, row, col) {
  return matrix.data[col * matrix.rows + row];
}

// Returns null when the frame has no face or no matrix: no guesses.
export function extractEyeSample(result, geometry, idx, msec, constants) {
  const landmarks = result.faceLandmarks?.[0];
  const matrix = result.facialTransformationMatrixes?.[0];
  if (!landmarks || !matrix) return null;

  const right = landmarks[constants.right_iris_center];
  const left = landmarks[constants.left_iris_center];
  const rightPx = [right.x * geometry.width, right.y * geometry.height];
  const leftPx = [left.x * geometry.width, left.y * geometry.height];

  const r = (row, col) => matrixAt(matrix, row, col);
  const eye = constants.eye_in_model_cm;
  const eyeCm = [0, 1, 2].map(
    (row) => r(row, 0) * eye[0] + r(row, 1) * eye[1] + r(row, 2) * eye[2] + r(row, 3),
  );

  return {
    idx,
    msec,
    u_px: (rightPx[0] + leftPx[0]) / 2,
    v_px: (rightPx[1] + leftPx[1]) / 2,
    ipd_px: Math.hypot(leftPx[0] - rightPx[0], leftPx[1] - rightPx[1]),
    // MediaPipe looks down -Z, so the eye's distance is the negated Z.
    depth_m: -eyeCm[2] * constants.cm_to_m,
    yaw_deg: (Math.atan2(-r(2, 0), r(0, 0)) * 180) / Math.PI,
  };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Factor correcting MediaPipe's metric scale for this viewer, from
// near-front-facing frames only. 1.0 when there are none.
export function estimateHeadScale(samples, geometry, viewer, constants) {
  const usable = samples.filter((s) => Math.abs(s.yaw_deg) <= constants.front_facing_yaw_deg);
  if (usable.length === 0) return 1.0;
  const implied = usable.map((s) => (s.ipd_px * s.depth_m) / geometry.focal);
  return viewer.ipd_m / median(implied);
}

export class HeadScaleNotReadyError extends Error {
  constructor(have, need) {
    super(`Head scale is not ready: ${have} of ${need} front-facing samples`);
    this.name = "HeadScaleNotReadyError";
  }
}

// Collects front-facing samples until it has enough, then freezes. Pass
// `scale` to start frozen, as Python's LiveScale does for a replay.
export class LiveScale {
  constructor(geometry, viewer, constants, { needed, scale } = {}) {
    this.geometry = geometry;
    this.viewer = viewer;
    this.constants = constants;
    this.needed = needed ?? constants.bootstrap_samples;
    this.collected = [];
    this.frozen = scale ?? null;
  }

  get isReady() {
    return this.frozen !== null;
  }

  get progress() {
    return [this.collected.length, this.needed];
  }

  get scale() {
    if (this.frozen === null) throw new HeadScaleNotReadyError(this.collected.length, this.needed);
    return this.frozen;
  }

  update(sample) {
    if (this.frozen !== null) return;
    if (Math.abs(sample.yaw_deg) > this.constants.front_facing_yaw_deg) return;
    this.collected.push(sample);
    if (this.collected.length >= this.needed) {
      this.frozen = estimateHeadScale(this.collected, this.geometry, this.viewer, this.constants);
    }
  }

  reset() {
    this.collected = [];
    this.frozen = null;
  }
}

export function eyePositionM(sample, geometry, headScale = 1.0) {
  const [cx, cy] = geometry.principalPoint;
  const depth = sample.depth_m * headScale;
  let x = ((sample.u_px - cx) * depth) / geometry.focal;
  const y = ((sample.v_px - cy) * depth) / geometry.focal;
  if (geometry.mirrored) x = -x;
  return [x, y, depth];
}
