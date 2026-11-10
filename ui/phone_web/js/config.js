// Device configs and constants from the server: no number is retyped here.

export async function loadConfig({ camera, viewer, screen } = {}) {
  const query = new URLSearchParams();
  if (camera) query.set("camera", camera);
  if (viewer) query.set("viewer", viewer);
  if (screen) query.set("screen", screen);
  const response = await fetch(`/config.json?${query}`);
  if (!response.ok) {
    const err = new Error(`config: ${response.status} ${await response.text()}`);
    err.name = "ConfigError";
    throw err;
  }
  return response.json();
}

// Where a splat file is and how it sits in the screen frame: { file,
// position, quaternion as x y z w, scale }, from /splats/<name>.json.
export async function loadSplatPlacement(name) {
  const response = await fetch(`/splats/${encodeURIComponent(name)}.json`);
  if (!response.ok) {
    const err = new Error(`splat ${name}: ${response.status} ${await response.text()}`);
    err.name = "ConfigError";
    throw err;
  }
  return response.json();
}

export async function loadScene(screen) {
  const response = await fetch(`/scene.json?screen=${encodeURIComponent(screen)}`);
  if (!response.ok) {
    const err = new Error(`scene: ${response.status} ${await response.text()}`);
    err.name = "ConfigError";
    throw err;
  }
  return response.json();
}
