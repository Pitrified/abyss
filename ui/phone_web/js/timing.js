// Per-stage frame times, as the Python loop's benchmark reports them: the
// newest frame's, and a rolling median over the last `size` frames.
//
// Stages are whatever the caller records; a stage a frame skipped (no screen,
// so no matrix or draw) is absent from that frame rather than zero. `draw`
// times the WebGL calls, not the GPU finishing the frame: WebGL is
// asynchronous, so a slow GPU shows up in `interval` rather than in `draw`.
//
// summary(): { last: { <stage>: ms, total }, median: { <stage>: ms, total,
//   interval }, fps }, where `interval` is the time between frames.

export const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

export class StageTimer {
  constructor(size = 60) {
    this.size = size;
    this.history = {};
    this.previous = null;
    this.current = null;
  }

  begin(now) {
    if (this.previous !== null) this.push("interval", now - this.previous);
    this.previous = now;
    this.current = {};
  }

  // Run `fn` as one stage and return what it returns.
  time(stage, fn) {
    const start = performance.now();
    const result = fn();
    this.record(stage, performance.now() - start);
    return result;
  }

  record(stage, ms) {
    this.current[stage] = ms;
  }

  end() {
    const total = Object.values(this.current).reduce((a, b) => a + b, 0);
    this.current.total = total;
    for (const [stage, ms] of Object.entries(this.current)) this.push(stage, ms);
    this.last = this.current;
  }

  push(stage, ms) {
    const values = (this.history[stage] ??= []);
    values.push(ms);
    if (values.length > this.size) values.shift();
  }

  summary() {
    const medians = Object.fromEntries(
      Object.entries(this.history).map(([stage, values]) => [stage, median(values)]),
    );
    const fps = medians.interval ? 1000 / medians.interval : null;
    return { last: this.last, median: medians, fps };
  }
}
