---
status: in progress
---

# Phase 3 - a Pixel capture, placed against the panel

## Overview

The first scene of our own: captured with the Pixel, posed with COLMAP and trained on g7 (phase 2's
pipeline), then placed against the panel so the depth gives the "wow". Drafted only; detailed once
phases 1 and 2 report.
Context: [`00_start.md`](00_start.md) Q10, depends on [`01_pixel_splat.md`](01_pixel_splat.md)
and [`02_g7_training.md`](02_g7_training.md).

## Goals

1. A capture of a chosen place, trained to a splat on g7.
2. A hand-set transform that places it behind and in front of the panel (Q10).
3. The user sees the effect on the Pixel.

## Plan

- **Capture (given).** A metro carriage, looking down the length of the train: an 11 s portrait
  video on the Pixel (1080x1920, 30 fps, 335 frames). The phone moves left and right, then turns
  in place, looking up, down, left and right. Kept in `~/data/splat_train/metro/`.
- **Frames.** Every second frame at half size, 540x960: 168 images.
- **Poses.** COLMAP on the CPU with the sequential matcher, since the frames are a video: each
  frame against its neighbours instead of all pairs. Checked before training; a capture that does
  not pose is cheaper to redo than to train.
- **Train, convert, place.** `scripts/train_splat.py` at its defaults, proposed first with its
  duration. The capture is not a ring around a subject, so the placement camera is chosen by hand.
- **Budget.** Phase 1 and 2: about 500k splats for about 15 fps, 1.5M for 10 fps; `keep=` thins.

## Capture guide

A draft, from what 3DGS needs and what the panel shows; the open choices are `00_start.md` Q14 to
Q18. It moves to `docs/guides/` once a capture made by it has worked.

**What the splat can show.** Only what some frame saw, and sharp only near the viewpoints the
frames were taken from. Depth comes from the camera *moving*; turning in place gives COLMAP
nothing to triangulate. The window is looked through from a patch about 30 cm wide and 20 cm tall
in front of it, so that patch, with a margin, is what the capture has to cover.

**Pick the scene.**

- Depth at several distances: something within a metre, something mid-way, something far.
  A corridor, a shelf in front of a room, a plant in front of a window.
- Still: no one walking through, nothing swinging. People who stay still are fine.
- Texture: surfaces with detail pose well; blank walls, glass, mirrors and shiny floors do not.
- Even, steady light: no flicker, no sun going in and out.

**Set the camera.**

- 1x lens only; do not zoom, or the Pixel switches lenses mid-video.
- Lock exposure and focus: long-press on a mid-distance object until it says locked.
- Stabilisation off if the camera app allows it; it warps frames by a varying amount.
- 4K at 30 fps in good light; 1080p at 60 fps in dim light, for a shorter shutter.
- Portrait, as the phone is held to view it.

**Move.**

- Stand where the viewer's eye will be. Hold the phone at eye height, facing the scene.
- Keep it facing the same way throughout; only slide it. Turn it at most a little, toward the middle.
- Slide left and right about 40 cm, slowly, about 10 cm a second: a full sweep takes 4 s.
- Repeat at three or four heights, about 10 cm apart, from chin to forehead.
- End with one slow move forward 20 cm and back.
- 30 to 60 s in all, one continuous take, starting and ending in the middle.
- Elbows in, phone in both hands, steps small; a smooth slow move beats a fast one with blur.

**Check before leaving.** Scrub the video: frames sharp when paused, brightness steady, nobody
crossing. If in doubt, take two.

## Pitfalls

- Little parallax: the sideways sway is short and the turn in place gives none. COLMAP may pose
  only part of the video, or fail.
- Repeating poles and seats can mismatch; the shiny floor and the lights change with the view.
- The Pixel's video stabilisation warps frames slightly, which COLMAP's single camera model
  does not describe.

## Done when

- The captured scene runs on the Pixel and the user judges the depth.
