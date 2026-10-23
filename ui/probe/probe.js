// What a device reports about itself: screen, viewport, safe area, GPU, and
// the camera stream the phone page would get. Shown as JSON ready to copy,
// logged to the console, and published on `window.probe` for the drive script.
//
// The fullscreen button is the gesture fullscreen needs; after it, what
// fullscreen changes (window, orientation, safe area) is read again into
// `afterTap`, with the portrait lock's outcome.
//
// status: "starting" | "done" | "error"; a part that fails carries { error }
// instead of failing the probe.

import { CONSTRAINTS } from "/phone_web/js/camera.js";
import { enterFullscreen } from "/phone_web/js/display.js";

const probe = { status: "starting", when: new Date().toISOString() };
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

async function main() {
  await part("device", device);
  await part("screen", screenInfo);
  await part("window", windowInfo);
  await part("webgl", webgl);
  await part("camera", camera);
  probe.fullscreenEnabled = document.fullscreenEnabled;
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
