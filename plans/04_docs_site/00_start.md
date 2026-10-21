---
status: draft
---

# Docs site - somewhere the mathematics can have figures

Spun off while writing [`../../docs/library/geometry_overview.md`](../../docs/library/geometry_overview.md)
and the two pages under it. The geometry is written up in full and the maths is readable, so this is
not a blocker on anything. It exists because one specific thing could not be written and was left
out rather than faked.

## What is missing

**Figures.** The three pages describe six diagrams in prose that a picture would carry better:

1. The camera frame and the screen frame facing each other, with the 180 degree rotation between them.
2. The similar-triangles construction of the frustum extents on the near plane.
3. An off-axis frustum beside a toed-in one, showing why the panel corners stop being the image
   corners.
4. The parallax gain: one point at depth, two eye positions, the two rays crossing the panel.
5. The full chain as a diagram, which the table in the overview stands in for.
6. The pinhole equation, which is one triangle and needs no words at all.

Prose is a poor substitute for every one of these, and geometry diagrams are the case markdown has
no answer for. Mermaid draws flowcharts, not 3D constructions.

## What is not missing

The **mathematics itself is fine in markdown**. LaTeX in `$...$` and `$$...$$` renders on GitHub and
in the VS Code preview, which is where these files are read today. That was checked before opening
this, and it is why the deferral is figures only rather than the whole technical write-up.
A plain text editor shows the LaTeX source, which is legible for expressions this short.

## What this would be

Not decided. The obvious shape is a static site over the existing `docs/` tree, which the repo
already writes for: "there is no mkdocs site yet, so write files that a site could later consume
unchanged" (`.github/copilot-instructions.md`). That constraint has been held, so nothing in `docs/`
should need rewriting for this.

The pieces that need a decision are the generator, the maths extension, and above all **how figures
are authored**, which is the actual problem: a diagram that is a checked-in PNG rots the moment the
geometry changes, and one that is generated from source needs the source to be something a person
will maintain.

## Open questions

Numbering is local to this folder.

- Q1: **Static site generator.** mkdocs-material is the fleet-conventional choice and reads the
  existing markdown tree unchanged. Sphinx would also give API docs from the Google-style docstrings,
  which are already written throughout and currently go unread.
  a. mkdocs-material, docs only.
  b. Sphinx with napoleon and autodoc, docs plus generated API reference.
  c. Neither yet; the figures are the point and they can be committed as images without a site.
  Recommended: c first, then a. The figures are the deliverable and a site is the hosting question,
  which is a separate and smaller problem.
  NEW_ANS:
- Q2: **How are figures authored?** They are 3D geometric constructions, so they need a source
  format that survives a change to the geometry.
  a. TikZ, compiled to SVG. Best output, a LaTeX toolchain to install and maintain.
  b. A Python script per figure using matplotlib, committed beside the image it produces.
     Consistent with a repo that already computes all of this, and testable.
  c. Drawn by hand in Inkscape and committed as SVG.
  Recommended: b, because the frustum in the figure would then be the same `frustum_for_eye` the
  code uses, and a figure that disagrees with the code becomes a test failure rather than a
  drawing to notice.
  NEW_ANS:
- Q3: **Does the maths need a renderer at all**, or is GitHub plus the VS Code preview enough?
  Adding KaTeX matters only if the docs are ever read somewhere else.
  NEW_ANS:
- Q4: **Where do generated figures live**, given that `docs/` is currently plain text and images
  are binary. They are small, but the repo has kept binaries out so far.
  NEW_ANS:
