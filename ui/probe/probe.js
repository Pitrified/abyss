// What a device reports about itself: screen, viewport, safe area, GPU, and
// the camera stream the phone page would get. Shown as JSON ready to copy,
// logged to the console, and published on `window.probe` for the drive script.
//
// The fullscreen button is the gesture fullscreen needs; after it, what
// fullscreen changes (window, orientation, safe area) is read again into
// `afterTap`, with the portrait lock's outcome.
//
// Then, with no tap, it sweeps the camera: after a countdown to hold still, it
// opens the front camera with each request in VARIANTS, tracks the face in a
// few frames of each, and records the pixels between the irises. Held still,
// a wider field of view puts fewer pixels between the eyes, so
// width / spacing and height / spacing compare how much of the scene each
// stream covers, and the ratio of spacings between two streams is the ratio
// of their focal lengths. `sweep.widest` is the most coverage across.
//
// status: "starting" | "sweeping" | "done" | "error"; a part that fails
// carries { error } instead of failing the probe. ?delay=<s> sets the
// countdown, 3 s by default.

import { VERSION } from "/version.js";
import { CONSTRAINTS } from "/phone_web/js/camera.js";
import { loadConfig } from "/phone_web/js/config.js";
import { enterFullscreen } from "/phone_web/js/display.js";
import { median } from "/phone_web/js/timing.js";
import { createLandmarker, monotonicClock } from "/phone_web/js/tracker.js";

// Camera requests to compare, all for the front camera and all `ideal`, so a
// device answers each with the nearest it has rather than refusing.
const VARIANTS = [
  ["page", CONSTRAINTS.video],
  ["landscape 1280x720", { width: { ideal: 1280 }, height: { ideal: 720 } }],
  ["portrait 720x960", { width: { ideal: 720 }, height: { ideal: 960 } }],
  ["landscape 960x720", { width: { ideal: 960 }, height: { ideal: 720 } }],
  ["aspect 3:4", { aspectRatio: { ideal: 3 / 4 } }],
  ["aspect 4:3", { aspectRatio: { ideal: 4 / 3 } }],
  ["no size", {}],
  ["no resize", { resizeMode: "none" }],
  ["max 3440x2448", { width: { ideal: 3440 }, height: { ideal: 2448 } }],
  ["1920x1440", { width: { ideal: 1920 }, height: { ideal: 1440 } }],
].map(([name, video]) => [name, { audio: false, video: { facingMode: "user", ...video } }]);

const SWEEP_FRAMES = 8;
const SETTLE_MS = 400;

const probe = { status: "starting", version: VERSION, when: new Date().toISOString() };
window.probe = probe;

const out = document.getElementById("out");
const status = document.getElementById("status");

function show() {
  out.textContent = JSON.stringify(probe, null, 2);
}

async function part(name, read) {
  try {
    probe[name] = await read();
  } catch (err) {
    probe[name] = { error: `${err.name}: ${err.message}` };
  }
}

async function device() {
  const ua = navigator.userAgentData;
  const high = ua ? await ua.getHighEntropyValues(["model", "platformVersion"]) : null;
  return {
    userAgent: navigator.userAgent,
    model: high?.model ?? null,
    platform: high ? `${high.platform} ${high.platformVersion}` : null,
    mobile: ua?.mobile ?? null,
    cores: navigator.hardwareConcurrency ?? null,
    memoryGb: navigator.deviceMemory ?? null,
    secureContext: window.isSecureContext,
  };
}

function screenInfo() {
  const ratio = window.devicePixelRatio;
  return {
    width: screen.width,
    height: screen.height,
    availWidth: screen.availWidth,
    availHeight: screen.availHeight,
    devicePixelRatio: ratio,
    physical: [Math.round(screen.width * ratio), Math.round(screen.height * ratio)],
  };
}

function windowInfo() {
  const viewport = window.visualViewport;
  return {
    inner: [window.innerWidth, window.innerHeight],
    outer: [window.outerWidth, window.outerHeight],
    visualViewport: viewport ? [viewport.width, viewport.height, viewport.scale] : null,
    orientation: screen.orientation ? [screen.orientation.type, screen.orientation.angle] : null,
    safeArea: safeArea(),
  };
}

function safeArea() {
  const style = getComputedStyle(document.getElementById("safe-area"));
  const px = (side) => parseFloat(style[`padding${side}`]);
  return { top: px("Top"), right: px("Right"), bottom: px("Bottom"), left: px("Left") };
}

function webgl() {
  const gl = document.createElement("canvas").getContext("webgl2");
  if (!gl) return { error: "no WebGL2" };
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  return {
    version: gl.getParameter(gl.VERSION),
    renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    vendor: debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
  };
}

// Device and group ids are per-origin hashes: noise in a pasted report.
function withoutIds({ deviceId, groupId, ...rest }) {
  return rest;
}

// Open the camera as the phone page does, read the stream, and keep the
// preview running so a person can see which way it is turned.
async function camera() {
  const stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
  const video = document.getElementById("camera");
  video.srcObject = stream;
  await video.play();
  const track = stream.getVideoTracks()[0];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return {
    label: track.label,
    settings: withoutIds(track.getSettings()),
    capabilities: track.getCapabilities ? withoutIds(track.getCapabilities()) : null,
    video: { width: video.videoWidth, height: video.videoHeight },
    cameras: devices.filter((d) => d.kind === "videoinput").map((d) => d.label),
  };
}

const nextFrame = (video) => new Promise((resolve) => video.requestVideoFrameCallback(resolve));
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function stopCamera(video) {
  video.srcObject?.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
}

// Pixels between the iris centres, median over a few frames with a face.
async function irisSpacing(video, landmarker, clock, constants) {
  const spacings = [];
  for (let tries = 0; tries < SWEEP_FRAMES * 3 && spacings.length < SWEEP_FRAMES; tries++) {
    const now = await nextFrame(video);
    const face = landmarker.detectForVideo(video, clock(now)).faceLandmarks?.[0];
    if (!face) continue;
    const right = face[constants.right_iris_center];
    const left = face[constants.left_iris_center];
    spacings.push(Math.hypot((right.x - left.x) * video.videoWidth, (right.y - left.y) * video.videoHeight));
  }
  return { ipdPx: spacings.length ? median(spacings) : null, faces: spacings.length };
}

function thumbnail(video, name) {
  const canvas = document.createElement("canvas");
  canvas.width = 120;
  canvas.height = Math.round((120 * video.videoHeight) / video.videoWidth);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  const figure = document.createElement("figure");
  const caption = document.createElement("figcaption");
  caption.textContent = `${name}: ${video.videoWidth}x${video.videoHeight}`;
  figure.append(canvas, caption);
  document.getElementById("thumbs").append(figure);
}

async function measureVariant(video, landmarker, clock, constants, [name, constraints]) {
  stopCamera(video);
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
  await wait(SETTLE_MS);
  const settings = stream.getVideoTracks()[0].getSettings();
  const { ipdPx, faces } = await irisSpacing(video, landmarker, clock, constants);
  thumbnail(video, name);
  const [width, height] = [video.videoWidth, video.videoHeight];
  return {
    name,
    request: constraints.video,
    settings: {
      width: settings.width,
      height: settings.height,
      aspectRatio: settings.aspectRatio,
      frameRate: settings.frameRate,
      resizeMode: settings.resizeMode,
    },
    video: [width, height],
    faces,
    ipdPx,
    widthPerIpd: ipdPx ? width / ipdPx : null,
    heightPerIpd: ipdPx ? height / ipdPx : null,
  };
}

async function sweep() {
  const delay = Number(new URLSearchParams(location.search).get("delay") ?? 3);
  for (let left = delay; left > 0; left--) {
    status.textContent = `hold still: camera sweep in ${left}`;
    await wait(1000);
  }
  const { constants } = await loadConfig();
  const landmarker = await createLandmarker();
  const clock = monotonicClock();
  const video = document.getElementById("camera");
  const variants = [];
  for (const variant of VARIANTS) {
    status.textContent = `sweeping ${variants.length + 1}/${VARIANTS.length}: ${variant[0]}, hold still`;
    try {
      variants.push(await measureVariant(video, landmarker, clock, constants, variant));
    } catch (err) {
      variants.push({ name: variant[0], request: variant[1].video, error: `${err.name}: ${err.message}` });
    }
  }
  stopCamera(video);
  landmarker.close();
  const measured = variants.filter((v) => v.widthPerIpd);
  const widest = measured.reduce((best, v) => (!best || v.widthPerIpd > best.widthPerIpd ? v : best), null);
  return { status: "done", widest: widest?.name ?? null, variants };
}

async function main() {
  await part("device", device);
  await part("screen", screenInfo);
  await part("window", windowInfo);
  await part("webgl", webgl);
  await part("camera", camera);
  probe.fullscreenEnabled = document.fullscreenEnabled;
  probe.status = "sweeping";
  show();
  await part("sweep", sweep);
  probe.status = "done";
  status.textContent = "done";
  show();
  console.log(`probe: ${JSON.stringify(probe)}`);
}

document.getElementById("fullscreen").addEventListener("click", async () => {
  const { fullscreen, lock } = await enterFullscreen();
  // Let the browser settle the new viewport before reading it.
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  probe.afterTap = { fullscreen, lock, window: windowInfo() };
  show();
  console.log(`probe afterTap: ${JSON.stringify(probe.afterTap)}`);
});

document.getElementById("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(out.textContent);
    status.textContent = "copied";
  } catch (err) {
    // No clipboard permission: select the text so a long press copies it.
    getSelection().selectAllChildren(out);
    status.textContent = `select and copy by hand (${err.name})`;
  }
});

main().catch((err) => {
  probe.status = "error";
  probe.error = `${err.name}: ${err.message}`;
  status.textContent = probe.error;
  show();
});
