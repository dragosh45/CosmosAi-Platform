# Agent Guide

`docs/` is not part of this repository. When it is present (cloned separately and
ignored by Git), read `docs/AGENTS.md` first and follow it. Never copy files or text
from `docs/` into tracked files of this repository.

Without `docs/`, work from `README.md`, `deploy/README.md` and the code.

## Basic Rules

- Keep implemented behaviour, measured results and limitations accurate in `README.md`.
- Never commit credentials, `.env` files, full datasets, caches or arbitrary run
  directories. Only the small model artifacts allowlisted in `.gitignore` belong in Git.
- Do not modify or commit the unrelated `assignment-property-dashboard/` folder.
- Do not deploy to Cloud Run.

## Common Checks

```bash
PYTHONPATH=. .venv/bin/pytest -q
git diff --check
```
