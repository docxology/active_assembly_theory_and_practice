# `tests/` — Agent Guide

## Overview

Real-computation test suite covering the `src/` producer modules, the
`assembly_packages` adapters, and the Julia/RxInfer bridge. No mock frameworks;
network and Julia coverage is opt-in so the default suite stays hermetic.

## Running

```bash
# from the project root
uv run pytest tests/ -q
uv run pytest tests/test_assembly_pipeline.py tests/test_optional_adapters.py -q
```

## Layout

- `conftest.py` — shared fixtures; headless matplotlib.
- `test_*.py` — one file per producer module or pipeline stage; names follow
  the module under test. The [`../src/assembly_packages/`](../src/assembly_packages/AGENTS.md)
  adapters are covered by `test_assembly_pipeline.py` and
  `test_optional_adapters.py`.
- `fixtures/` — committed test inputs.

## Rules

- Tests exercise the real import path (`src/...`), never a project-root
  `sys.path` shortcut.
- Generated artifacts are written under `tmp_path` or `output/`, never read
  back from a prior run.
- Julia/RxInfer tests require the native package in [`../julia/`](../julia/AGENTS.md);
  they are skipped when Julia is absent.
