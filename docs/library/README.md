# docs/library

How the code works and why it is shaped that way. Narrative notes rather than API reference:
what a docstring cannot hold because it spans several modules, or because it is the reasoning behind
a decision rather than the decision itself.

The procedures these explain are in [`../guides/`](../guides/README.md).

## The geometry

The core of the project, in reading order. Start at the overview, which indexes the other two.

| Page | What it covers |
| ---- | -------------- |
| [`geometry_overview.md`](geometry_overview.md) | the whole chain from camera frame to pixel, the three coordinate frames, and the four calibrations |
| [`viewer_position.md`](viewer_position.md) | pixels to a metric eye position: the pinhole model, MediaPipe's assumed intrinsics, and the head scale calibration |
| [`off_axis_projection.md`](off_axis_projection.md) | eye position to pixels: the sheared frustum, the `glFrustum` matrix, and the two invariants that define the effect |

Mathematics is written as LaTeX in markdown, which GitHub and the VS Code preview render.
Figures are missing and deferred to [`../../plans/04_docs_site/00_start.md`](../../plans/04_docs_site/00_start.md).

## Everything else

| Page | What it covers |
| ---- | -------------- |
| [`camera_calibration.md`](camera_calibration.md) | why the calibration methods work, and how to read a measurement that came back wrong |
| [`pose_tools_boundary.md`](pose_tools_boundary.md) | which code belongs upstream in `pose-tools` and which belongs here |
| [`params.md`](params.md) | the params and paths layer, and what was deliberately left out of it |

## Related, outside this folder

The renderer that will replace the wireframe is not chosen yet. The candidates, and the constraints
they are judged against, are in
[`../../plans/02_scene_rendering/01_research_renderers.md`](../../plans/02_scene_rendering/01_research_renderers.md).
