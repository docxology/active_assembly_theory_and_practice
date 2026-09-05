# `scripts/` — Agent Guide

## Overview

Thin orchestrators. Every command here delegates to `src/` modules; no
scientific logic lives in `scripts/`. Run everything from the project root with
`uv run`.

## File Organization

| Script | Purpose |
| --- | --- |
| `00_preflight.py` | Manuscript preflight via the template `infrastructure` package (exits `2` without a template checkout) |
| `run_scientific_program.py` | Canonical scientific-program CLI (subcommands incl. `acquire-external`) |
| `run_assembly_inference.py` | Decoding, typed observations, and Julia/RxInfer inference entry point |
| `run_formal_verification.py` | Builds `formal/` and checks the correspondence fixture |
| `run_replication.py` | Same-host replay of the label-blind public bundle |
| `run_rxinfer_backend.py` | Direct Julia/RxInfer backend invocation |
| `run_julia_tests.py` | Runs the native Julia package test suite |
| `run_full_verification.sh` | End-to-end local verification sequence |
| `validate_analysis_outputs.py` | Output artifact validation |
| `validate_docx_render.py` | Post-render DOCX checks |
| `validate_pdf_accessibility.py` | Post-render PDF tag-tree / accessibility audit |
| `z_generate_manuscript_variables.py` | `{{TOKEN}}` hydration for `manuscript/` |

## Conventions

- Scripts import from `src/` only; they may not implement numerics.
- Network-reaching commands must pass an explicit `--allow-network` flag; see
  [`../configs/AGENTS.md`](../configs/AGENTS.md).
- Generated artifacts land under `output/` and are disposable.
