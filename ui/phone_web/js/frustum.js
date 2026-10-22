// Off-axis projection. Port of abyss.render.frustum; read that module for
// the three frames and why X and Y flip between camera and screen.
//
// Every function takes the eye in the CAMERA frame, metres, as Python's do.
// Matrices are arrays of four rows, row-major, as numpy prints them.

export class EyeBehindScreenError extends Error {
  constructor(eyeZ) {
    super(`eye is not in front of the panel: screen-frame z=${eyeZ.toFixed(4)} m`);
    this.name = "EyeBehindScreenError";
  }
}

export class InvalidClipPlanesError extends Error {
  constructor(near, far) {
    super(`clip planes must satisfy 0 < near < far, got near=${near} far=${far}`);
    this.name = "InvalidClipPlanesError";
  }
}

export class PointNotInFrontError extends Error {
  constructor(count) {
    super(`${count} point(s) at or behind the eye`);
    this.name = "PointNotInFrontError";
  }
}

export function eyeInScreenFrame(eyeCameraM, screen, constants) {
  const flip = constants.camera_to_screen;
  return [0, 1, 2].map((i) => (eyeCameraM[i] - screen.camera_to_centre_m[i]) * flip[i]);
}

export function frustumForEye(screen, eyeCameraM, constants, near = constants.near_m, far = constants.far_m) {
  if (near <= 0 || far <= near) throw new InvalidClipPlanesError(near, far);
  const [ex, ey, ez] = eyeInScreenFrame(eyeCameraM, screen, constants);
  if (ez <= 0) throw new EyeBehindScreenError(ez);
  const scale = near / ez;
  const halfW = screen.width_m / 2;
  const halfH = screen.height_m / 2;
  return {
    left: (-halfW - ex) * scale,
    right: (halfW - ex) * scale,
    bottom: (-halfH - ey) * scale,
    top: (halfH - ey) * scale,
    near,
    far,
  };
}

export function projectionMatrix(f) {
  const { left, right, bottom, top, near, far } = f;
  return [
    [(2 * near) / (right - left), 0, (right + left) / (right - left), 0],
    [0, (2 * near) / (top - bottom), (top + bottom) / (top - bottom), 0],
    [0, 0, -(far + near) / (far - near), (-2 * far * near) / (far - near)],
    [0, 0, -1, 0],
  ];
}

function multiply(a, b) {
  return a.map((row) => [0, 1, 2, 3].map((j) => row.reduce((sum, v, k) => sum + v * b[k][j], 0)));
}

// Screen-frame points to clip space: the projection with the eye translated
// to the origin folded in.
export function viewProjectionMatrix(screen, eyeCameraM, constants, near, far) {
  const frustum = frustumForEye(screen, eyeCameraM, constants, near, far);
  const eye = eyeInScreenFrame(eyeCameraM, screen, constants);
  const translation = [
    [1, 0, 0, -eye[0]],
    [0, 1, 0, -eye[1]],
    [0, 0, 1, -eye[2]],
    [0, 0, 0, 1],
  ];
  return multiply(projectionMatrix(frustum), translation);
}

// Screen-frame points to pixels, y down. No clipping, as in Python.
export function projectPoints(matrix, points, widthPx, heightPx) {
  const clip = points.map(([x, y, z]) => matrix.map((r) => r[0] * x + r[1] * y + r[2] * z + r[3]));
  const behind = clip.filter((c) => c[3] <= 0).length;
  if (behind) throw new PointNotInFrontError(behind);
  return clip.map(([x, y, , w]) => [((x / w + 1) / 2) * widthPx, ((1 - y / w) / 2) * heightPx]);
}

// Refuse an output that would stretch the scene: port of check_aspect.
export function checkAspect(screen, widthPx, heightPx, constants) {
  const screenAspect = screen.width_m / screen.height_m;
  const outputAspect = widthPx / heightPx;
  if (Math.abs(outputAspect - screenAspect) / screenAspect > constants.aspect_tolerance) {
    const err = new Error(
      `panel aspect ${screenAspect.toFixed(4)} does not match output aspect ${outputAspect.toFixed(4)}`,
    );
    err.name = "AspectMismatchError";
    throw err;
  }
}
