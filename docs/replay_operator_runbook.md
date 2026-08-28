# Separately controlled replay runbook

This runbook is the handoff for the frozen public replay interface. It is
written for an operator who is not the producer of the repository's gold-label
receipt and who can run the public bundle on a separate machine or separately
controlled CI worker. A local rerun by the repository owner is a rehearsal,
not independent evidence.

## What the operator receives

The operator receives the repository at a pinned revision plus the contents of
`output/replication/public/`:

- `public_manifest.json` — label-blind sequences and public features;
- `protocol_manifest.json` — protocol identifier and digest;
- `toolchain_receipt.json` — producer-side runtime receipt;
- `REPLAY_INSTRUCTIONS.md` — the short offline command guide.

The operator must not receive `output/replication/private/gold_labels.json`, any
source-family or split manifest, the source-bound label contract, or a file
whose ordering or filename encodes the protected target. The public bundle is
already digest-ordered to prevent source-generation order from becoming a
label channel.

## Operator procedure

Run from the pinned repository checkout with network access disabled at the
machine or job boundary:

```text
uv run python scripts/run_scientific_program.py verify-public
uv run python scripts/run_scientific_program.py replay --offline
uv run python scripts/run_scientific_program.py run-model --model calibrated_interpretable
uv run python scripts/run_scientific_program.py verify-separation
uv run python scripts/run_scientific_program.py write-replay-receipt \
  --predictions output/replication/calibrated_interpretable_predictions.json \
  --output external-replay-receipt.json \
  --operator-id <operator-id> \
  --machine-id <machine-or-ci-id> \
  --bundle-sha256 current \
  --model calibrated_interpretable \
  --seed 0 \
  --started-at <run-start-iso8601> \
  --completed-at <run-completion-iso8601> \
  --command replay --offline
```

The `--command` argument is the final argument group and preserves
`--offline` inside the recorded command. The operator should retain the exact
stdout, stderr, Python/package versions, operating-system identifier, and
network-disabled control evidence alongside the JSON receipt. Timestamps must
be timezone-qualified ISO-8601 values measured around the replay command; if
the two timestamp options are omitted, the producer records the local
rehearsal's receipt-generation times. The prediction file must contain exactly
one `record_id` and one finite probability per public row, in public order,
with no gold-label fields.

The operator then transfers only these two files back to the protected scoring
boundary:

1. `external-replay-receipt.json`;
2. the prediction file named by that receipt.

The protected owner-side command is:

```text
uv run python scripts/run_scientific_program.py ingest-replay \
  --receipt <external-replay-receipt.json> \
  --predictions <predictions.json>
```

Ingestion rejects a protocol or public-bundle hash mismatch, missing or extra
identifiers, forbidden public fields, probability drift, and a receipt that
does not explicitly attest a separately controlled run. Gold labels are read
only inside the protected scorer. A successful ingestion produces
`output/data/independent_replay_receipt.json` with endpoint status
`independent_replay_observed`.

## Evidence boundary

The repository can generate a valid pending receipt and exercise the entire
public-only interface, but it cannot honestly create the external attestation
itself. The following are distinct outcomes:

- `implemented_local`: the code path and receipt schema run on the producer's
  machine;
- `submitted_pending_verification`: a pending receipt binds a public bundle and
  prediction file but has no protected external attestation;
- `independent_replay_observed`: a separate operator or separately controlled
  machine supplied the receipt, hashes match, and protected scoring succeeded.

Independent replay is evidence that the frozen interface was replayed. It is
not an independent reimplementation, a new scientific sample, or evidence of
application effectiveness. If the run fails, preserve the receipt and
diagnostics as a first-class failure; do not rerun until a favorable result is
obtained or alter the protocol to clear the gate.
