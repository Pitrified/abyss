# docs/guides

How to do something. Tooling, commands, and procedures meant to be followed step by step while
sitting at a machine.

Why any of it works is in [`../library/`](../library/README.md), and the two are deliberately
separate: a procedure that stops to explain itself is harder to follow, and an explanation
interrupted by shell commands is harder to read.

| Page | What it covers |
| ---- | -------------- |
| [`makefile.md`](makefile.md) | the task runner, every target, and why to use it over a bare `uv run` |
| [`uv.md`](uv.md) | dependency groups and the uv commands this repo uses |
| [`pre_commit.md`](pre_commit.md) | maintaining the hook config and bumping its versions |
| [`manual_measurements.md`](manual_measurements.md) | the three measurements that need a person: focal length, matched fields of view, camera-to-screen offset |
| [`phase5_live_runbook.md`](phase5_live_runbook.md) | running the head-coupled loop live on g7, with the known failures and what to record |
| [`phone_runbook.md`](phone_runbook.md) | phase 4 of the phone webapp: the page on g4 and the Pixel, served by g7, with the mirroring check and what to record |
| [`ui_harness.md`](ui_harness.md) | serving the browser UIs and running them headless in Chromium with a recorded face as the camera |

## Notes

`manual_measurements.md` and `phase5_live_runbook.md` both need physical access to the machine
holding the camera, which on this fleet means g7. g7 is now an unattended home server, so either one
means going and sitting at it. Everything else runs anywhere.

`phase5_live_runbook.md` was written for a phase that is now complete
([`plans/01_abyss_expansion/`](../../plans/01_abyss_expansion/tracking.md)). It is kept as a guide
rather than folded into the plan record because the procedure is still the way to run the loop.
