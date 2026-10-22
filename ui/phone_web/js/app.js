// Wiring: a source of frames, the landmarker, the eye position port, and,
// when a screen is named, the off-axis projection and the scene.
//
// Query parameters:
//   source=<url>     a clip instead of the camera, stepped frame by frame
//   camera=<key>     camera registry key (default: the server's, the Pixel)
//   viewer=<key>     viewer registry key
//   screen=<key>     screen registry key; draws the scene for that panel
//   eye=<x,y,z>      no tracking: draw one frame for this camera-frame eye
//   size=<WxH>       output size in pixels, for eye=; default fits the window
//   scale=<float>    freeze the head scale instead of bootstrapping it
//   delegate=GPU     MediaPipe delegate, CPU by default
//   fps=<float>      the clip's frame rate, for timestamps; 30 by default

import { eachFrame, openCamera } from "./camera.js";
import { decodeFrame, loadClip } from "./clip_source.js";
import { loadConfig, loadScene } from "./config.js";
import { drawEye, drawFrame } from "./draw.js";
import { extractEyeSample, eyePositionM, LiveScale } from "./eye_position.js";
import { checkAspect, viewProjectionMatrix } from "./frustum.js";
import { frameGeometry } from "./geometry.js";
import { SceneView } from "./scene.js";
import { PositionSmoother } from "./smoothing.js";
import { fail, setStatus, state } from "./state.js";
import { createLandmarker, monotonicClock } from "./tracker.js";

const params = new URLSearchParams(location.search);

// The largest size with the panel's aspect that fits the window. The frustum
// is built for the physical panel, so the picture must keep its shape: a
// stretched picture is wrong everywhere, a letterboxed one only at the edges.
function fitToWindow(screen) {
  const aspect = screen.width_m / screen.height_m;
  const width = Math.min(window.innerWidth, Math.round(window.innerHeight * aspect));
  return [width, Math.round(width / aspect)];
}

function outputSize(screen) {
  const size = params.get("size");
  return size ? size.split("x").map(Number) : fitToWindow(screen);
}

// Draws the scene for an eye, and publishes the matrix it used.
function makeDrawer(config, sceneDoc) {
  if (!config.screen) return null;
  const canvas = document.getElementById("scene");
  canvas.hidden = false;
  const view = new SceneView(canvas, sceneDoc);
  const [width, height] = outputSize(config.screen);
  checkAspect(config.screen, width, height, config.constants);
  view.setSize(width, height);
  state.output = { width, height };
  return (eye) => {
    const matrix = viewProjectionMatrix(config.screen, eye, config.constants);
    view.render(matrix);
    return matrix;
  };
}

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
  const matrix = smoothed && ctx.draw ? ctx.draw(smoothed) : null;
  const [collected, needed] = ctx.scale.progress;
  state.scale = {
    ready: ctx.scale.isReady,
    collected,
    needed,
    value: ctx.scale.isReady ? ctx.scale.scale : null,
  };
  state.frames += 1;
  return { idx, msec, sample, position, smoothed, matrix };
}

function context(config, draw, width, height) {
  const geometry = frameGeometry(config.camera, width, height, config.constants);
  const fixed = params.has("scale") ? Number(params.get("scale")) : undefined;
  return {
    config,
    draw,
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

async function runClip(url, config, draw, landmarker) {
  state.source = url;
  const frames = await loadClip(url);
  const canvas = document.getElementById("view");
  const fps = Number(params.get("fps") ?? 30);
  let ctx = null;
  state.status = "running";
  for (let idx = 0; idx < frames.length; idx++) {
    const bitmap = await decodeFrame(frames[idx]);
    if (ctx === null) {
      ctx = context(config, draw, bitmap.width, bitmap.height);
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

async function runCamera(config, draw, landmarker) {
  state.source = "camera";
  const video = document.getElementById("camera");
  state.camera = await openCamera(video);
  const ctx = context(config, draw, state.camera.width, state.camera.height);
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

// No tracking at all: one frame for a given eye, for tests and for looking.
function runFixedEye(config, draw) {
  if (!draw) {
    const err = new Error("eye= needs screen= to draw anything");
    err.name = "ParameterError";
    throw err;
  }
  const eye = params.get("eye").split(",").map(Number);
  state.source = "fixed";
  state.latest = { sample: null, position: eye, smoothed: eye };
  state.matrix = draw(eye);
  state.frames = 1;
  setStatus(`fixed eye ${eye.map((v) => (v * 100).toFixed(1)).join(", ")} cm`);
  state.status = "done";
}

async function main() {
  setStatus("loading");
  const screen = params.get("screen");
  const config = await loadConfig({ camera: params.get("camera"), viewer: params.get("viewer"), screen });
  state.config = config;
  const draw = makeDrawer(config, screen ? await loadScene(screen) : null);
  if (draw) document.body.classList.add("with-scene");
  if (params.has("eye")) return runFixedEye(config, draw);
  const landmarker = await createLandmarker({ delegate: params.get("delegate") ?? "CPU" });
  const source = params.get("source");
  if (source) await runClip(source, config, draw, landmarker);
  else await runCamera(config, draw, landmarker);
}

main().catch(fail);
