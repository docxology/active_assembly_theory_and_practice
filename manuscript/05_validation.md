# Validation and Scholarship Boundary {#sec:validation}

The project writes a source ledger with `{{SCHOLARSHIP_SOURCE_COUNT}}` entries
to `{{SCHOLARSHIP_SOURCE_LEDGER}}`. Each source is assigned a project role and
a claim boundary, so software-package documentation, peer-reviewed active
inference theory, and optional binary-analysis adapters do not collapse into a
single undifferentiated support claim.

The project also writes a claim register with `{{CLAIM_COUNT}}` bounded claims
to `{{CLAIM_REGISTER}}`; `{{VALIDATED_CLAIM_COUNT}}` are linked to sources,
generated artifacts, validation checks, and explicit boundary text.
The generated artifact surface is separately recorded in
`{{REPRODUCIBILITY_MANIFEST}}`, which tracks
`{{REPRODUCIBILITY_ARTIFACT_COUNT}}` analysis artifacts by file size and
SHA-256 digest.

![Scholarship source roles used to support theory, software, documentation, and binary-analysis context.]({{SCHOLARSHIP_FIGURE}}){#fig:scholarship-source-roles}

![Scholarship evidence tiers partitioned by implemented, comparison, roadmap, and context stages.]({{SCHOLARSHIP_MATRIX_FIGURE}}){#fig:scholarship-evidence-matrix}

![Claim-register evidence coverage across source keys, artifacts, and validation checks.]({{CLAIM_FIGURE}}){#fig:claim-evidence-map}

![Reproducibility manifest coverage across generated artifact classes.]({{REPRODUCIBILITY_FIGURE}}){#fig:artifact-hash-manifest}

The analysis validator reports `{{VALIDATION_PASS_COUNT}}` passing checks out
of `{{VALIDATION_TOTAL_COUNT}}`, with `{{VALIDATION_FAIL_COUNT}}` failures and
`all_passed = {{VALIDATION_ALL_PASSED}}`. The report is written to
`{{VALIDATION_REPORT}}`. It checks trace schema fields, sample hashes, required
package provenance, trace observation semantics, dynamic-register trace shape,
RxInfer free-energy finiteness, RxInfer companion artifact consistency,
data-hash consistency, reproducibility-manifest hashes and byte sizes, figure
nonblankness, source-ledger bibliography coverage, manuscript citation usage,
citation closure, source-shape quality, source-stage and evidence-tier
partitioning, claim-register links, RedTeam remediation status, and cautious
claim-boundary language.

![Validation summary for trace, RxInfer, figure, scholarship, and manuscript checks.]({{VALIDATION_FIGURE}}){#fig:validation-summary}

This validation layer is intentionally adversarial. It is designed to fail if a
future edit reports RxInfer free energy without finite JSONL values, adds a
source-ledger entry without a bibliography record, drops a required adapter
provenance row, changes observation labels without updating category IDs, drops
RxInfer posterior/configuration artifacts, changes a manifest-tracked artifact
after hash recording, adds an unlinked claim, or renders empty figures while
the basic pipeline still exits successfully.
