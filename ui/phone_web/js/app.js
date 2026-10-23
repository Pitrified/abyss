// Wiring: a source of frames, the landmarker, the eye position port, and,
// when a screen is named, the off-axis projection and the scene.
//
// Query parameters:
//   source=<url>     a clip instead of the camera, stepped frame by frame
//   camera=<key>     camera registry key (default: the server's, the Pixel)
//   viewer=<key>     viewer registry key
//   screen=<key>     screen registry key; draws the scene for that panel
//   eye=<x,y,z>      no tracking: draw one frame for this camera-frame eye
//   size=<WxH>       output size in pixels, one per CSS pixel, for tests;
//                    by default the picture fits the window, refits when it
//                    changes, and draws at the device pixel ratio
//   scale=<float>    freeze the head scale instead of bootstrapping it
//   delegate=GPU     MediaPipe delegate, CPU by default
//   fps=<float>      the clip's frame rate, for timestamps; 30 by default

import { eachFrame, openCamera } from "./camera.js";
import { decodeFrame, loadClip } from "./clip_source.js";
import { loadConfig, loadScene } from "./config.js";
import { fullscreenOnTap } from "./display.js";
import { drawEye, drawFrame } from "./draw.js";
import { extractEyeSample, eyePositionM, LiveScale } from "./eye_position.js";
import { checkAspect, viewProjectionMatrix } from "./frustum.js";
import { frameGeometry } from "./geometry.js";
import { SceneView } from "./scene.js";
import { PositionSmoother } from "./smoothing.js";
import { fail, setStatus, state } from "./state.js";
import { StageTimer } from "./timing.js";
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

// The matrix for an eye, and drawing the scene through one. The picture
// refits to the window unless size= pins it, and redraws the last matrix,
// since resizing a canvas clears it.
function makeDrawer(config, sceneDoc) {
  if (!config.screen) return null;
  const canvas = document.getElementById("scene");
  canvas.hidden = false;
  const view = new SceneView(canvas, sceneDoc);
  const size = params.get("size");
  let last = null;
  const resize = () => {
    const [width, height] = size ? size.split("x").map(Number) : fitToWindow(config.screen);
    const pixelRatio = size ? 1 : window.devicePixelRatio;
    checkAspect(config.screen, width, height, config.constants);
    view.setSize(width, height, pixelRatio);
    state.output = { width, height, pixelRatio };
    if (last) view.render(last);
  };
  resize();
  if (!size) window.addEventListener("resize", resize);
  return {
    matrix: (eye) => viewProjectionMatrix(config.screen, eye, config.constants),
    render: (matrix) => {
      view.render(matrix);
      last = matrix;
    },
  };
}

// One frame through the pipeline, each stage timed. Returns what was learnt,
// for the track.
function process(ctx, detect, idx, msec, now) {
  const timer = ctx.timer;
  timer.begin(now);
  const result = timer.time("landmarker", detect);
  const { sample, position, smoothed } = timer.time("eye", () => locateEye(ctx, result, idx, msec));
  let matrix = null;
  if (smoothed && ctx.draw) {
    matrix = timer.time("matrix", () => ctx.draw.matrix(smoothed));
    timer.time("draw", () => ctx.draw.render(matrix));
  }
  timer.end();
  state.timing = timer.summary();
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

function locateEye(ctx, result, idx, msec) {
  const sample = extractEyeSample(result, ctx.geometry, idx, msec, ctx.config.constants);
  if (!sample) return { sample, position: null, smoothed: ctx.smoother.hold() };
  let position = null;
  let smoothed = null;
  ctx.scale.update(sample);
  if (ctx.scale.isReady) {
    position = eyePositionM(sample, ctx.geometry, ctx.scale.scale);
    smoothed = ctx.smoother.update(position);
  }
  state.latest = { sample, position, smoothed };
  return { sample, position, smoothed };
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
    timer: new StageTimer(),
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
    const detect = () => landmarker.detectForVideo(bitmap, msec);
    const entry = process(ctx, detect, idx, msec, performance.now());
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
    const entry = process(ctx, () => landmarker.detectForVideo(video, msec), state.frames, msec, now);
    drawEye(canvas, entry.sample, state.camera);
    const fps = state.timing.fps ? ` | ${state.timing.fps.toFixed(1)} fps` : "";
    setStatus(describe(entry) + fps);
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
  state.matrix = draw.matrix(eye);
  draw.render(state.matrix);
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
  fullscreenOnTap();
  if (params.has("eye")) return runFixedEye(config, draw);
  const landmarker = await createLandmarker({ delegate: params.get("delegate") ?? "CPU" });
  const source = params.get("source");
  if (source) await runClip(source, config, draw, landmarker);
  else await runCamera(config, draw, landmarker);
}

main().catch(fail);
