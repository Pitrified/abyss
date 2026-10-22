// Causal smoothing for a 3D position track. Port of
// abyss.viewer.smoothing.PositionSmoother over pose_tools' left-triangle
// filter: weights 1..n normalised to sum to 1, newest sample heaviest.

export function leftTriangleFilter(size) {
  const total = (size * (size + 1)) / 2;
  return Array.from({ length: size }, (_, i) => (i + 1) / total);
}

export class PositionSmoother {
  constructor(filterSize) {
    this.filter = leftTriangleFilter(filterSize);
    this.history = null; // per axis, oldest first
    this.last = null;
  }

  update(position) {
    if (this.history === null) {
      // Prime with the first sample so the filter starts settled.
      this.history = position.map((v) => this.filter.map(() => v));
    }
    this.last = this.history.map((axis, i) => {
      axis.shift();
      axis.push(position[i]);
      return axis.reduce((sum, v, k) => sum + v * this.filter[k], 0);
    });
    return [...this.last];
  }

  // A frame with no face: keep the last estimate rather than invent one.
  hold() {
    return this.last === null ? null : [...this.last];
  }
}
