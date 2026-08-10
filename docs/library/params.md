# Params and paths

`abyss` uses the fleet's params pattern, cut down to what it actually needs.

## Shape

```text
src/abyss/
  params/
    abyss_params.py   AbyssParams (singleton) + get_abyss_params() / get_abyss_paths()
    abyss_paths.py    AbyssPaths
  metaclasses/
    singleton.py      Singleton metaclass
```

Reach the paths through the accessor, never by constructing the class:

```python
from abyss.params.abyss_params import get_abyss_paths

video_fp = get_abyss_paths().pose_fol / "yoga01.mp4"
```

`AbyssParams` uses the `Singleton` metaclass, so it loads once per process. Tests that need a fresh
instance clear `Singleton._instances`.

## Paths

| Attribute   | Points at                        | Note                                    |
| ----------- | -------------------------------- | ---------------------------------------- |
| `src_fol`   | the installed `abyss` package    | derived from `abyss.__file__`             |
| `root_fol`  | the repository root              | `src_fol.parents[1]`                      |
| `cache_fol` | `<root>/cache`                   | gitignored                                |
| `data_fol`  | `<root>/data`                    | gitignored                                |
| `pose_fol`  | `~/data/pose`                    | sample videos, outside the repo           |

MediaPipe model files are **not** here: pose-tools' `ModelManager` resolves them under
`~/.mediapipe/models/`.

## What was deliberately left out

The template's params layer also carries env stage/location dispatch (`EnvType`,
`EnvStageType`, `EnvLocationType`), a `config/` package of Pydantic models, `BaseModelKwargs`, and
`load_env()` reading `~/cred/<project>/.env`. None of it is here, because:

- abyss runs in one place. `LOCAL` vs `RENDER` is a hosting distinction for web projects.
- abyss has no secrets, so there is nothing for `load_env()` to load.
- nothing yet has typed settings to model.

This is a deliberate application of "build only what has a caller", not an oversight. Add any of
them the moment something needs it - the sibling repos are the reference implementation, and
`pose-tools` carries the full version of every piece.

The same rule governs `AbyssPaths`: a new entry goes in when code reads it, not in anticipation.
