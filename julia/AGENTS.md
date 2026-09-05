# `julia/` — Agent Guide

## Overview

Native Julia side of the project: the `ActiveAssemblyRxInfer` package that runs
Bethe Free Energy inference over the typed categorical observations produced by
the Python decoders. It is called from `src/rxinfer_bridge.py`, which pins the
expected Julia and RxInfer versions.

## Layout

- `ActiveAssemblyRxInfer/` — the package: `Project.toml`, `Manifest.toml`,
  `src/`, `test/`, and `scripts/`.
- `ActiveAssemblyRxInfer/Project.toml` — package dependencies (RxInfer pinned).
- `ActiveAssemblyRxInfer/test/` — the package test suite, run by
  `../scripts/run_julia_tests.py`.

## Rules

- Version expectations live in `src/rxinfer_bridge.py`
  (`EXPECTED_JULIA_VERSION`, `EXPECTED_RXINFER_VERSION`); keep
  `Project.toml` compat entries in step.
- The Julia side is an inference backend, not a data source: it consumes the
  JSON trace schema produced by `src/` and returns free-energy values. It does
  not read `data/` or `figures/` directly.
