@.github/copilot-instructions.md

## Claude Code

This repo's canonical instructions live in `.github/copilot-instructions.md`
(imported above) so Copilot and Claude share one source of truth.

`abyss` is the application half of the pose split: it owns viewers, screens, and
rendering. The general pose, video and geometry utilities live in the sibling
`pose-tools` repo, which `abyss` depends on by git tag. The dependency is
strictly one-way: `pose-tools` must never import `abyss`.

Planning lives in `plans/`, one folder per initiative. Read
`plans/00_template_alignment/tracking.md` first for where the reboot stands.
