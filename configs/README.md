# configs

Checked-in, pinned configuration inputs for the external-source, labelling, and
replay-VM surfaces. Nothing here is generated at run time; every file is a
source-bound contract that a producer reads and refuses to run without.

## Contents

| File | Lines | Purpose |
| --- | --- | --- |
| `external_sources.yaml` | 170 | Pinned acquisition manifest: repository URL, revision, license digest, and archive SHA-256 per source. |
| `external_label_design.yaml` | 313 | Declares how label contracts are constructed from pinned source files (variant counts, controls, edit rules). |
| `external_label_contracts.yaml` | 3545 | Generated-then-committed contract rows produced by `src.external_labeling.generate_label_contract`. |
| `replay_vm.yaml` | 48 | Pinned Lima template for the blue-backed replay guest (image URL, digest, CPU/memory/disk). |

## Quick Start

```bash
# from the project root
uv run python scripts/run_scientific_program.py prepare-external
uv run python scripts/run_scientific_program.py validate-external
limactl validate configs/replay_vm.yaml
```

## Key Points

- `external_sources.yaml` is `static_only: true` and
  `network_policy: explicit_acquisition_only`. Acquisition is the single
  network-enabled command and always requires `--allow-network`.
- `external_label_contracts.yaml` carries `design_sha256` and `manifest_sha256`
  so a contract cannot drift from the design or manifest it was built from.
- `replay_vm.yaml` state lives under `/Volumes/blue`; see
  [`../docs/replay_vm.md`](../docs/replay_vm.md).

## More Information

See [AGENTS.md](AGENTS.md) for the field-level contract and failure modes.
