# plans

Planning notes for abyss, one folder per initiative (`00_template_alignment` is the first).

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

Every markdown file in a plan folder carries the tracked-development style frontmatter:

```yaml
---
status: draft   # draft | planned | in progress | done | superseded | discarded
---
```

`draft` = brainstorm material, not yet the plan of record; `done` = decision taken or analysis closed;
`in progress` = currently being implemented. The frontmatter and reality must agree.
