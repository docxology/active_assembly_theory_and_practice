# configs — Agent Guide

## Overview

`configs/` holds the four pinned YAML contracts that bound the project's
external-source, labelling, and replay-VM surfaces. They exist so that the
network, license, compilation, and virtualization boundaries are inspectable in
version control rather than implied by code. Producers under [`../src`](../src)
read these files and fail closed when a required field, digest, or source is
missing.

## File Organization

```
configs/
├── AGENTS.md                        this file
├── README.md                        quick reference
├── external_label_contracts.yaml    generated-then-committed contract rows (3545 lines)
├── external_label_design.yaml       how contract rows are constructed (313 lines)
├── external_sources.yaml            pinned acquisition manifest (170 lines)
└── replay_vm.yaml                   pinned Lima guest template (48 lines)
```

## `external_sources.yaml`

Header fields: `schema_version: "2.0.0"`, `manifest_id`,
`network_policy: "explicit_acquisition_only"`, `static_only: true`.

Each `sources` row records `source_id`, `project_id`, `revision_label`,
`repository_url`, `revision`, `license`, `license_file`,
`license_file_sha256`, `source_archive_sha256`, and `source_family`.

The manifest is an acquisition *specification*, not an execution list. It is
consumed by [`../src/external_sources.py`](../src/external_sources.py). The only
command that may reach the network is:

```bash
# from the project root
uv run python scripts/run_scientific_program.py acquire-external \
  --manifest configs/external_sources.yaml \
  --allow-network
```

## `external_label_design.yaml`

Declares the row-construction policy: `minimum_labelable_examples`,
`equivalent_variants_per_source`, `negative_controls_per_source`,
`near_miss_variants_per_source`, and a `source_designs` list naming
`source_id`, `source_path`, and `function_id`.

Equivalent rows append comments after the source (preserving preprocessing
tokens); near-miss rows apply explicit source-text edits with stated
preconditions; negative controls are byte-identical pairs retained as
non-separating controls. The producer refuses a missing or multiply matched
edit, a missing source, a failed static compilation, or a failed Capstone
decode.

## `external_label_contracts.yaml`

Written by `src.external_labeling.generate_label_contract` (recorded in its own
`generated_by` field) and committed so the contract is auditable without
re-running acquisition. Header carries `schema_version: 2.0.0`,
`design_id`, `design_schema_version`, `status`, `minimum_labelable_examples`,
`design_sha256`, and `manifest_sha256`. Each `contracts` row carries
`example_id`, `project_id`, `revision_label`, `source_id`, `source_family`,
`source_path`, `function_id`, `role`, `transformation_family`, and
`preconditions`.

Regenerate rather than hand-edit:

```bash
# from the project root
uv run python scripts/run_scientific_program.py label-external
```

## `replay_vm.yaml`

Pinned Lima template for the same-host, separately controlled replay guest:
`minimumLimaVersion: "2.2.0"`, `vmType: "qemu"`, `arch: "aarch64"`,
`os: "Linux"`, a pinned Ubuntu 24.04 ARM64 cloud image with its SHA-256
`digest`, and fixed CPU/memory/disk sizing. Consumed by
[`../src/replay_vm.py`](../src/replay_vm.py), which refuses any VM root outside
`/Volumes/blue`.

```bash
# from the project root
limactl validate configs/replay_vm.yaml
uv run python scripts/run_scientific_program.py vm-probe
```

## Best Practices

- Do change a pin deliberately and regenerate the dependent receipt; a digest
  mismatch is designed to fail the gate rather than warn.
- Do keep `external_label_contracts.yaml` machine-generated. Hand edits break
  its `design_sha256` / `manifest_sha256` binding.
- Do not treat `external_sources.yaml` as permission to execute an acquired
  artifact. The pipeline is static-only: compile and decode, never run.

## See Also

- [`../docs/replay_vm.md`](../docs/replay_vm.md) — pinned VM boundary
- [`../docs/replay_operator_runbook.md`](../docs/replay_operator_runbook.md) — operator handoff
- [`../src/AGENTS.md`](../src/AGENTS.md) — modules that consume these files
- [`../AGENTS.md`](../AGENTS.md) — project boundaries
