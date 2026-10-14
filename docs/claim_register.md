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

`validated` means that a row clears its bounded artifact/check contract. It
never promotes an external or independent milestone by itself. The generated
scientific-program and replay receipts own those status fields.

Run `uv run --locked python scripts/check_claim_map.py` after generating the
register. It requires identifier-set equality with [the authored claim map](claim_map.md)
and rejects duplicated IDs. [The docs gate](README.md) checks authored structure
and citations; [analysis validation](analysis_validation.md) checks executable
artifact closure. These checks require human review of changed claim meaning:
they do not parse arbitrary prose to prove that every assertion is supported.
