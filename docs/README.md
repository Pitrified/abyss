# docs

Plain markdown, read in the repo. There is no site yet, so pages are written to be read as files and
to be consumable unchanged if one is ever built ([`plans/04_docs_site/`](../plans/04_docs_site/00_start.md)).

Two folders, split by what a page is for:

| Folder | What is in it |
| ------ | ------------- |
| [`guides/`](guides/README.md) | how to do something: tooling, commands, procedures to follow |
| [`library/`](library/README.md) | how something works and why: the code and the reasoning behind it |

The rule of thumb is the tense. A guide tells you what to type next.
A library note explains a decision that was already taken.

Documentation that is neither belongs elsewhere: decisions in flight live in
[`plans/`](../plans/README.md), and anything that only makes sense next to the code lives in a
docstring.

## Where to start

**Understanding what abyss does.**
[`library/geometry_overview.md`](library/geometry_overview.md), which is the index for the
mathematics and links onward to the two detailed pages.

**Setting the project up.** The root [`README.md`](../README.md), then
[`guides/makefile.md`](guides/makefile.md) for what the commands are.

**Running the live loop.** [`guides/phase5_live_runbook.md`](guides/phase5_live_runbook.md).

**Calibrating a new machine.** [`guides/manual_measurements.md`](guides/manual_measurements.md)
for the steps, [`library/camera_calibration.md`](library/camera_calibration.md) for why they work
and how to read a result that looks wrong.

**Adding code.** [`library/pose_tools_boundary.md`](library/pose_tools_boundary.md) first, since
the most common mistake is writing something here that belongs upstream.
