// Wiring: a source of frames, the landmarker, the eye position port.
//
// Query parameters:
//   source=<url>   a clip instead of the camera, stepped frame by frame
//   camera=<key>   camera registry key (default: the server's, the Pixel)
//   viewer=<key>   viewer registry key
//   scale=<float>  freeze the head scale instead of bootstrapping it
//   delegate=GPU   MediaPipe delegate, CPU by default
//   fps=<float>    the clip's frame rate, for timestamps; 30 by default

import { eachFrame, openCamera } from "./camera.js";
import { decodeFrame, loadClip } from "./clip_source.js";
import { loadConfig } from "./config.js";
import { drawEye, drawFrame } from "./draw.js";
import { extractEyeSample, eyePositionM, LiveScale } from "./eye_position.js";
import { frameGeometry } from "./geometry.js";
import { PositionSmoother } from "./smoothing.js";
import { fail, setStatus, state } from "./state.js";
import { createLandmarker, monotonicClock } from "./tracker.js";

const params = new URLSearchParams(location.search);

// One frame through the pipeline. Returns what was learnt, for the track.
function process(ctx, result, idx, msec) {
  const sample = extractEyeSample(result, ctx.geometry, idx, msec, ctx.config.constants);
  let position = null;
  let smoothed = null;
  if (sample) {
    ctx.scale.update(sample);
    if (ctx.scale.isReady) {
      position = eyePositionM(sample, ctx.geometry, ctx.scale.scale);
      smoothed = ctx.smoother.update(position);
    }
    state.latest = { sample, position, smoothed };
  } else {
    smoothed = ctx.smoother.hold();
  }
  const [collected, needed] = ctx.scale.progress;
  state.scale = {
    ready: ctx.scale.isReady,
    collected,
    needed,
    value: ctx.scale.isReady ? ctx.scale.scale : null,
  };
  state.frames += 1;
  return { idx, msec, sample, position, smoothed };
}

function context(config, width, height) {
  const geometry = frameGeometry(config.camera, width, height, config.constants);
  const fixed = params.has("scale") ? Number(params.get("scale")) : undefined;
  return {
    config,
    geometry,
    scale: new LiveScale(geometry, config.viewer, config.constants, { scale: fixed }),
    smoother: new PositionSmoother(config.constants.filter_size),
  };
}

function describe(entry) {
  if (!entry.sample) return `frame ${entry.idx}: no face`;
  if (!entry.position) return `frame ${entry.idx}: calibrating ${state.scale.collected}/${state.scale.needed}`;
  const cm = entry.smoothed.map((v) => (v * 100).toFixed(1)).join(", ");
  return `frame ${entry.idx}: eye ${cm} cm`;
}

async function runClip(url, config, landmarker) {
  state.source = url;
  const frames = await loadClip(url);
  const canvas = document.getElementById("view");
  const fps = Number(params.get("fps") ?? 30);
  let ctx = null;
  state.status = "running";
  for (let idx = 0; idx < frames.length; idx++) {
    const bitmap = await decodeFrame(frames[idx]);
    if (ctx === null) {
      ctx = context(config, bitmap.width, bitmap.height);
      state.camera = { label: url, width: bitmap.width, height: bitmap.height };
    }
    const msec = (idx * 1000) / fps;
    const entry = process(ctx, landmarker.detectForVideo(bitmap, msec), idx, msec);
    state.track.push(entry);
    drawFrame(canvas, bitmap);
    drawEye(canvas, entry.sample);
    bitmap.close();
    setStatus(describe(entry));
  }
  // Release the WASM graph. This also flushes tasks-vision's usage log, which
  // the server's CSP then refuses to let leave; make e2e checks both halves.
  landmarker.close();
  state.status = "done";
}

async function runCamera(config, landmarker) {
  state.source = "camera";
  const video = document.getElementById("camera");
  state.camera = await openCamera(video);
  const ctx = context(config, state.camera.width, state.camera.height);
  const canvas = document.getElementById("view");
  const clock = monotonicClock();
  state.status = "running";
  eachFrame(video, (now) => {
    const msec = clock(now);
    const entry = process(ctx, landmarker.detectForVideo(video, msec), state.frames, msec);
    drawEye(canvas, entry.sample, state.camera);
    setStatus(describe(entry));
  });
}

async function main() {
  setStatus("loading");
  const config = await loadConfig({ camera: params.get("camera"), viewer: params.get("viewer") });
  state.config = config;
  const landmarker = await createLandmarker({ delegate: params.get("delegate") ?? "CPU" });
  const source = params.get("source");
  if (source) await runClip(source, config, landmarker);
  else await runCamera(config, landmarker);
}

main().catch(fail);
