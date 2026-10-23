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

export async function loadScene(screen) {
  const response = await fetch(`/scene.json?screen=${encodeURIComponent(screen)}`);
  if (!response.ok) {
    const err = new Error(`scene: ${response.status} ${await response.text()}`);
    err.name = "ConfigError";
    throw err;
  }
  return response.json();
}
