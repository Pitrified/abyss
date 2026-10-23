// Draw the served scene through a view projection matrix, with three.js.
//
// The scene arrives as segments in the screen frame (/scene.json). three.js's
// own cameras cannot express an off-axis frustum, so a bare Camera carries the
// whole view projection as its projection matrix and stays at the origin:
// three then multiplies projection x identity view x identity model, which is
// exactly the matrix abyss computed.

import * as THREE from "/vendor/three/build/three.module.js";

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
    this.camera.matrixAutoUpdate = false;
  }

  // Width and height in CSS pixels; the canvas holds `pixelRatio` drawn pixels
  // per CSS pixel, so a test passes 1 to read back a frame of the size it asked.
  setSize(widthPx, heightPx, pixelRatio = 1) {
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.setSize(widthPx, heightPx);
  }

  // `matrix` is four rows, row-major; Matrix4.set takes row-major arguments.
  render(matrix) {
    this.camera.projectionMatrix.set(...matrix.flat());
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    this.renderer.render(this.scene, this.camera);
  }
}
