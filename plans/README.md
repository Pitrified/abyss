# plans

Planning notes for abyss, one folder per initiative (`00_template_alignment` is the first).

Multi-phase initiatives follow the tracked-development layout: `00_start.md`-style bootstrap file
(analysis, decisions, numbered `Qn` open questions with `ANS:` placeholders), `tracking.md` as the
index to read first (phases table + append-only log), and one `NN_phase_name.md` sub-plan per phase.
Work that surfaces mid-effort but is out of scope becomes a **sibling folder**, not an extra phase.

## Plans are ADRs - for closed folders

- The **current** folder (highest number, the feature being worked on) is freely editable:
  refine text in place, fix decisions, restructure. No amendment trail inside it - that is noise.
- What we do record inside the current folder: **rejected alternatives and why**.
  That inline note is what stops a later "we might do it the other way" from re-hitting the same problem.
- When a future feature revisits a decided area, the old folder **freezes** (becomes the ADR)
  and the new `NN` folder starts with an "old state as is" assessment, then the new design,
  rationale, and migration. Past folders are never edited retroactively.
- Superseded or discarded plans keep their files, with the reason noted at the top of the body.

## Frontmatter

The bootstrap file and every phase sub-plan carry the tracked-development style frontmatter.
`tracking.md` does not - it is the index, and the phases table is where status lives:

```yaml
---
status: draft   # draft | planned | in progress | done | superseded | discarded
---
```

`draft` = brainstorm material, not yet the plan of record; `done` = decision taken or analysis closed;
`in progress` = currently being implemented. A sub-plan's frontmatter and its row in `tracking.md`
must agree with each other and with reality. The bootstrap file is `done` once its open questions
are answered, and goes back to `draft` if a new batch is opened.
