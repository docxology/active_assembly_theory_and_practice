# src/assembly_packages

The adapter layer between real assembly-analysis packages and the project's
stable JSON trace schema. Two adapters are required dependencies (Capstone,
Unicorn); everything else is probed for availability and recorded as
provenance, never imported eagerly.

## Contents

| File | Lines | Purpose |
| --- | --- | --- |
| `__init__.py` | 21 | Re-exports `BasicBlock`, `CfgEdge`, `PackageProvenance`, `TraceBundle`, `build_static_cfg`, `build_trace_bundle`, `trace_bundle_to_json`. |
| `schema.py` | 307 | The trace bundle dataclasses, `TRACE_SCHEMA_VERSION`, static CFG construction, and JSON serialization. |
| `capstone_adapter.py` | 36 | `capstone_provenance()` and `disassemble_with_capstone()` over [`../disassembly.py`](../disassembly.py). |
| `unicorn_adapter.py` | 34 | `unicorn_provenance()` and `emulate_with_unicorn()` over [`../emulation.py`](../emulation.py). |
| `optional_adapters.py` | 106 | `probe_optional_packages()` — 14 availability probes with no eager import. |

## Quick Start

```python
from src.assembly_packages import build_trace_bundle, trace_bundle_to_json
```

```bash
# from the project root
uv run pytest tests/test_assembly_pipeline.py tests/test_optional_adapters.py -q
```

## Key Points

- `TRACE_SCHEMA_VERSION` is `"1.4.0"`. Changing the bundle shape means bumping
  it and updating the consumers and the validation oracle together.
- The two required adapters are `capstone_static_disassembly` and
  `unicorn_bounded_emulation`. Both attach `PackageProvenance` rather than
  returning bare Python objects.
- Optional probes report `available: false` when a package or command is
  missing. That is a recorded fact, not a failure and not a backlog item.
- `build_static_cfg` emits `external_0x...` edges for direct branch targets
  outside the decoded byte span instead of fabricating blocks.

## More Information

See [AGENTS.md](AGENTS.md) for the adapter contract and
[`../../docs/assembly_packages.md`](../../docs/assembly_packages.md) for the
full trace-schema and provenance specification.
