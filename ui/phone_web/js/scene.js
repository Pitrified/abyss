// Draw the served scene through a view projection matrix, with three.js.
//
// The scene arrives as segments in the screen frame (/scene.json). three.js's
// own cameras cannot express an off-axis frustum, so a bare Camera sits at the
// eye, unrotated, and carries the frustum as its projection matrix: three then
// multiplies projection x translation to the eye x identity model, which is
// exactly the matrix abyss computed.

import * as THREE from "three";

const rgb = ([b, g, r]) => new THREE.Color(r / 255, g / 255, b / 255);

export class SceneView {
  constructor(canvas, sceneDoc) {
    // preserveDrawingBuffer so a test can read the frame back after drawing.
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.setClearColor(rgb(sceneDoc.background));
    this.scene = new THREE.Scene();
    const positions = sceneDoc.segments.flat(2);
    const colours = sceneDoc.colours.flatMap((c) => {
      const col = rgb(c);
      return [col.r, col.g, col.b, col.r, col.g, col.b];
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colours, 3));
    const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true }));
    lines.frustumCulled = false;
    this.scene.add(lines);
    this.camera = new THREE.Camera();
  }

  // A Gaussian splat, placed in the screen frame by its placement file
  // (/splats/<name>.json). Spark is imported only here, so a page without a
  // splat never downloads it. The file is loaded once and kept, so the rate
  // sweep can redraw it at several counts. Resolves to the number drawn.
  async addSplat(placement, keep = 1) {
    this.spark = await import("@sparkjsdev/spark");
    this.scene.add(new this.spark.SparkRenderer({ renderer: this.renderer }));
    this.splatSource = new this.spark.PackedSplats({ url: `/splats/${placement.file}` });
    await this.splatSource.initialized;
    this.placement = placement;
    return this.useKeep(keep);
  }

  // Draw the loaded splat thinned evenly to `keep` of its splats; thinned
  // splats keep their base colour only. Resolves to the number drawn.
  async useKeep(keep) {
    const { SplatMesh } = this.spark;
    const source = this.splatSource;
    const mesh =
      keep < 1
        ? new SplatMesh({
            constructSplats: (out) =>
              source.forEachSplat((i, center, scales, quaternion, opacity, color) => {
                if (Math.floor((i + 1) * keep) > Math.floor(i * keep)) {
                  out.pushSplat(center, scales, quaternion, opacity, color);
                }
              }),
          })
        : new SplatMesh({ splats: source });
    await mesh.initialized;
    mesh.position.set(...this.placement.position);
    mesh.quaternion.set(...this.placement.quaternion);
    mesh.scale.setScalar(this.placement.scale);
    // A mesh disposes the splats it draws, so the one drawing the loaded
    // file is only removed: the next share is thinned from that file.
    if (this.splatMesh) {
      this.scene.remove(this.splatMesh);
      if (this.splatMesh.splats !== source) this.splatMesh.dispose();
    }
    this.scene.add(mesh);
    this.splatMesh = mesh;
    return mesh.numSplats;
  }

  // Width and height in CSS pixels; the canvas holds `pixelRatio` drawn pixels
  // per CSS pixel, so a test passes 1 to read back a frame of the size it asked.
  setSize(widthPx, heightPx, pixelRatio = 1) {
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.setSize(widthPx, heightPx);
  }

  // `view` is from eyeView: the eye in the screen frame, and the projection as
  // four rows, row-major; Matrix4.set takes row-major arguments.
  render(view) {
    this.camera.position.set(...view.eye);
    this.camera.updateMatrixWorld();
    this.camera.projectionMatrix.set(...view.projection.flat());
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    this.renderer.render(this.scene, this.camera);
  }
}
