# Claim Register Contract

`src/claim_register.py` is the machine-readable claim boundary for the
manuscript. It generates:

- `output/data/claim_register.json`
- `output/data/claim_register.md`
- `output/figures/claim_evidence_map.png`

Each claim row includes:

- `claim_id`
- `claim_type`
- `statement`
- `support_keys`
- `artifact_paths`
- `validation_checks`
- `boundary`
- `status`

The validation oracle fails if a claim cites a source key outside the scholarship
ledger, points at a missing artifact, names a validation check that does not
exist, is missing from the generated register, or is not marked `validated`.
The artifact-reproducibility claim also requires the generated manifest and
manifest figure to exist before claim closure can pass.

This claim register is deliberately stricter than the prose. It prevents a
future manuscript edit from adding a confident claim that is not tied to a
source, generated artifact, and executable validation check.
