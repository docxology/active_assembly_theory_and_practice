# `src/assembly_packages/` — Agent Guide

## Overview

Adapter layer between real assembly-analysis packages and the project stable
JSON trace schema. Two adapters are required dependencies (Capstone, Unicorn);
everything else is probed for availability and recorded as provenance, never
imported eagerly.

## Contents

| File | Purpose |
| --- | --- |
| `__init__.py` | Re-exports `BasicBlock`, `CfgEdge`, `PackageProvenance`, `TraceBundle`, `build_static_cfg`, `build_trace_bundle`, `trace_bundle_to_json` |
| `schema.py` | Trace-bundle dataclasses, `TRACE_SCHEMA_VERSION` (`"1.4.0"`), static CFG construction, JSON serialization |
| `capstone_adapter.py` | `capstone_provenance()` and `disassemble_with_capstone()` over `../disassembly.py` |
| `unicorn_adapter.py` | `unicorn_provenance()` and `emulate_with_unicorn()` over `../emulation.py` |
| `optional_adapters.py` | `probe_optional_packages()` — availability probes with no eager import |

## Rules

- Changing the bundle shape means bumping `TRACE_SCHEMA_VERSION` and updating
  the consumers and the validation oracle together.
- Adapters attach `PackageProvenance` rather than returning bare Python
  objects; provenance travels with every trace.
- Optional packages are recorded as provenance, never hard requirements.

## See also

- [`README.md`](README.md) — quick reference and quick start
- [`../AGENTS.md`](../AGENTS.md) — `src/` layer contract
