# Blue-backed replay VM

The project includes a reproducible, same-host VM boundary for the frozen
label-blind replay interface. It is intended for a separately controlled
operator run on the owner workstation when a second physical machine or CI
worker is not yet available. It is evidence of replay under a distinct guest
control boundary; it is not independent reimplementation, independent
physical-machine replication, or a new scientific sample.

## Pinned environment

The checked-in [`configs/replay_vm.yaml`](../configs/replay_vm.yaml) pins:

- Lima `2.2.0`;
- Lima `qemu` with the pinned ARM64 `qemu-system-aarch64` binary;
- native Apple-Silicon execution of the ARM64 guest (QEMU is selected because
  the blue volume is Journaled HFS+ over USB and native `vz` disk finalization
  did not complete reliably there);
- Ubuntu 24.04 ARM64 server cloud image from the 2026-07-05 release;
- the image SHA-256 digest;
- 4 virtual CPUs, 8 GiB memory, and a 64 GiB guest disk;
- no host checkout mounts, proxy propagation, additional networks, or
  containerd services.

All Lima state, image cache, source-transfer archives, logs, and receipts are
stored below `/Volumes/blue/aapta-replay-vm`.
The command refuses to use a path outside `/Volumes/blue` or the volume root.
The volume should have enough free space for the guest disk and image cache;
the manifest records the observed capacity before provisioning.

Provisioning is the only VM operation that may use the network. It requires an
explicit `--allow-network` flag and retrieves only the pinned guest image,
the pinned Linux ARM64 `uv` release, and packages required to build the
locked Python environment. The source archive is created from a clean Git
`HEAD`; the public bundle is a deterministic archive containing exactly the
four files under `output/replication/public/`. The host checkout is never
mounted into the guest.

## Reproducible procedure

Install and verify the pinned host tool first:

```text
brew install lima
limactl --version                 # must report 2.2.0
limactl validate configs/replay_vm.yaml
```

Generate the public bundle and verify that the operator surface does not need
private labels:

```text
uv run python scripts/run_scientific_program.py generate-public
uv run python scripts/run_scientific_program.py verify-public
```

The VM transfer requires a clean committed checkout because its source archive
is bound to `git rev-parse HEAD`. After the source-bound changes have been
committed, create and provision the guest:

```text
uv run python scripts/run_scientific_program.py vm-probe
uv run python scripts/run_scientific_program.py vm-create --allow-network
uv run python scripts/run_scientific_program.py vm-provision --allow-network
```

Run the replay without a provisioning network permission:

```text
uv run python scripts/run_scientific_program.py vm-replay \
  --model calibrated_interpretable \
  --seed 0
```

The replay command enters a fresh guest network namespace and requires that
the namespace contain only `lo` and no routes. It runs `verify-public`,
`replay --offline`, the selected deterministic label-blind model, and the
public-only separation check. It writes the namespace proof, pending receipt,
predictions, stdout, and stderr to the blue-backed receipt directory.

Attest the controlled boundary and perform protected scoring only after the
host validates the hashes:

```text
uv run python scripts/run_scientific_program.py vm-attest-replay
uv run python scripts/run_scientific_program.py promote-evidence
```

`vm-attest-replay` records the explicit control scope
`same_host_separately_controlled_guest`, verifies that private gold labels and
source-bound label contracts were not transferred, and invokes the existing
protected scorer. The resulting `output/data/independent_replay_receipt.json`
may report `independent_replay_observed` for this separately controlled VM
run. The manuscript must still describe it as same-host controlled replay,
not as an independent machine, independent reimplementation, or independent
external validation.

Stop the VM without deleting its disk or receipts when it is not in use:

```text
uv run python scripts/run_scientific_program.py vm-stop
uv run python scripts/run_scientific_program.py vm-status
```

There is intentionally no `vm-delete` command in the scientific interface.
Removing a guest disk is a separate destructive lifecycle decision and must be
handled outside the receipt-producing workflow after the receipts have been
archived.

## Safety and evidence boundary

The guest receives a clean source archive and public input files only. It does
not receive `output/replication/private/gold_labels.json`, the owner
replication manifest, external label contracts, credentials, or host mounts.
Arbitrary external binaries are never executed. The guest only runs the
repository's deterministic public replay code. Provisioning package commands
are logged and hashed, but their current Ubuntu package mirror is not treated
as part of the offline replay evidence.

The VM receipt distinguishes:

- `implemented_local`: the pinned configuration and command surface exist;
- `validated`: provisioning or replay completed under the declared boundary;
- `independent_replay_observed`: a protected scorer accepted a receipt with
  the same-host controlled-run attestation;
- `failed` or `blocked_external`: the boundary or a required hash/check failed.

A same-host VM does not satisfy the stronger requirement for a separately
operated physical machine or independently controlled CI service. Such a run
can be ingested through the ordinary external receipt interface, but its
`control_scope` and limitation must remain visible in the promotion ledger.
