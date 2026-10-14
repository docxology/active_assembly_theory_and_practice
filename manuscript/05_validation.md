# Validation and Scholarship Boundary {#sec:validation}

The project writes a source ledger with `{{SCHOLARSHIP_SOURCE_COUNT}}` entries
to `{{SCHOLARSHIP_SOURCE_LEDGER}}`. Each source is assigned a project role and
a claim boundary, so software-package documentation, peer-reviewed active
inference theory, and optional binary-analysis adapters do not collapse into a
single undifferentiated support claim.
The role partition is shown in [@fig:scholarship-source-roles].

The project also writes a claim register with `{{CLAIM_COUNT}}` bounded claims
to `{{CLAIM_REGISTER}}`; `{{VALIDATED_CLAIM_COUNT}}` are linked to sources,
generated artifacts, validation checks, and explicit boundary text. The linked
evidence matrix and claim-to-evidence map are shown in
[@fig:scholarship-evidence-matrix] and [@fig:claim-evidence-map].
The generated artifact surface is separately recorded in
`{{REPRODUCIBILITY_MANIFEST}}`, which tracks
`{{REPRODUCIBILITY_ARTIFACT_COUNT}}` analysis artifacts by file size and
SHA-256 digest; the manifest is visualized in [@fig:artifact-hash-manifest].

The sibling docxology/template renderer is also configured to emit a standard
combined DOCX at `{{DOCX_RENDER_ARTIFACT}}`. The current source-bound interface
reports `{{DOCX_RENDER_STATUS}}`, with `{{DOCX_RENDER_FIGURE_COUNT}}` embedded
figures and `{{DOCX_RENDER_CAPTION_COUNT}}` caption paragraphs. The post-render
receipt at `{{DOCX_RENDER_RECEIPT}}` checks OOXML ZIP integrity, required
document parts, figure/caption parity, unresolved tokens, the hydrated
manuscript hash, and the substantive figure-registry hash. The two
validator-produced summary figures are excluded from that digest because
their own checks are self-referential. This is a format and freshness contract;
it does not establish PDF/UA tagging, reading order, or scholarly validity,
which remain separate gates.

The generated cover illustration is source-controlled at
{{COVER_IMAGE_ARTIFACT}} and is included through the template's config-driven
paper-cover contract. The same validation surface checks the bounded symbolic
semantics report, stepwise transfer-function/Unicorn refinement, direct-target CFG partition, held-out fitting membership, and
cover-image presence/nonblankness; these checks are engineering and provenance
contracts, not evidence of scientific generality.

![{{FIGURE_SCHOLARSHIP_SOURCE_ROLES_CAPTION}}]({{FIGURE_SCHOLARSHIP_SOURCE_ROLES_PATH}}){#fig:scholarship-source-roles width=92%}

![{{FIGURE_SCHOLARSHIP_EVIDENCE_MATRIX_CAPTION}}]({{FIGURE_SCHOLARSHIP_EVIDENCE_MATRIX_PATH}}){#fig:scholarship-evidence-matrix width=92%}

![{{FIGURE_CLAIM_EVIDENCE_MAP_CAPTION}}]({{FIGURE_CLAIM_EVIDENCE_MAP_PATH}}){#fig:claim-evidence-map width=95%}

![{{FIGURE_ARTIFACT_HASH_MANIFEST_CAPTION}}]({{FIGURE_ARTIFACT_HASH_MANIFEST_PATH}}){#fig:artifact-hash-manifest width=92%}

The shared artifact boundary is separately audited rather than inferred from
the reproducibility manifest. The current receipt layer is schema
`{{ARTIFACT_RECEIPT_SCHEMA_VERSION}}`, with
`{{ARTIFACT_RECEIPT_REQUIRED_FIELD_COUNT}}` required common fields across
`{{ARTIFACT_RECEIPT_COUNT}}` project-owned JSON artifacts and
`{{ARTIFACT_RECEIPT_ERROR_COUNT}}` envelope errors. The audit command is
`{{ARTIFACT_RECEIPT_AUDIT_COMMAND}}`. Domain schema versions remain intact;
the receipt layer adds protocol, producer, input, toolchain, limitation, and
status lineage, visualized in [@fig:artifact-receipt-closure]. This design
follows provenance and research-object packaging
practice [@w3cprov2013;@rocrate2023;@sandve2013repro], while preserving the
important distinction between local reproducibility and independent replication
[@acm2020artifact].

![{{FIGURE_ARTIFACT_RECEIPT_CLOSURE_CAPTION}}]({{FIGURE_ARTIFACT_RECEIPT_CLOSURE_PATH}}){#fig:artifact-receipt-closure width=100%}

The analysis validator reports `{{VALIDATION_PASS_COUNT}}` passing checks out
of `{{VALIDATION_TOTAL_COUNT}}`, with `{{VALIDATION_FAIL_COUNT}}` failures and
`all_passed = {{VALIDATION_ALL_PASSED}}`. The report is written to
`{{VALIDATION_REPORT}}` and summarized in [@fig:validation-summary]. It checks
trace schema fields, sample hashes, required
package provenance, trace observation semantics, dynamic-register trace shape,
RxInfer free-energy finiteness, RxInfer companion artifact consistency,
categorical-companion model identity, memory-operand to dynamic-event joins,
data-hash consistency, reproducibility-manifest hashes and byte sizes, figure
nonblankness, source-ledger bibliography coverage, manuscript citation usage,
citation closure, source-shape quality, source-stage and evidence-tier
partitioning, claim-register links, fixed-parameter generative adequacy, and
cautious claim-boundary language. The newer check families added with the
landed tranches are contract-plus-freshness pairs for the compiler matrix
(`compiler_transfer_contract`/`compiler_transfer_freshness`), the quotient
classifier and its invariance protocol
(`quotient_classifier_contract`/`quotient_classifier_freshness` and
`quotient_invariance_contract`/`quotient_invariance_freshness`, which together
pin the development-only hyperparameter selection, deterministic-start ledger,
family-shuffle control, bounded metadata probe, near-duplicate declarations,
and cross-encoding consistency), and the external preparation audit
(`external_preparation_audit_contract`/`external_preparation_audit_freshness`).

![{{FIGURE_VALIDATION_SUMMARY_CAPTION}}]({{FIGURE_VALIDATION_SUMMARY_PATH}}){#fig:validation-summary width=96%}

This validation layer is intentionally adversarial. It is designed to fail if a
future edit reports RxInfer free energy without finite JSONL values, adds a
source-ledger entry without a bibliography record, drops a required adapter
provenance row, changes observation labels without updating category IDs, drops
RxInfer posterior/configuration artifacts, changes a manifest-tracked artifact
after hash recording, bypasses the common artifact envelope, adds an unlinked claim, or renders empty figures while
the basic pipeline still exits successfully.

## Open-roadmap receipt gates {#sec:open-roadmap-validation}

The extended gates are source-bound rather than green-by-construction. The
protocol hash is recomputed before validating the quotient, correspondence,
compiler, sensor, replication, and theorem-phase receipts. Each producer is
then rerun and compared with the disk artifact, with the timing-only part of
the sensor receipt excluded from the deterministic freshness comparison because
wall-clock calibration is expected to vary. Contract checks separately require
label blindness, three-way family splits, nonempty metrics, explicit
unavailable states, static-only compiler analysis, exact public/gold separation,
and a proved-or-falsified theorem boundary.

The new source-bound artifact family is:

| Surface | Contract that is actually checked | Promotion boundary |
| --- | --- | --- |
| quotient classifier | protocol hash, label-blind features, disjoint training/development/held-out families, finite metrics and clustered intervals; development-only hyperparameter selection, five-start ledger, family-shuffle and bounded metadata-probe controls, declared near-duplicate siblings, and cross-encoding consistency | bounded endpoint only; semantic discovery is not established and the retrospective correction withholds superiority claims |
| formal correspondence | real Capstone decode, bounded Unicorn trace, Python reference step alignment, modeled flags, and Lean prerequisite | declared transparent subset only; no complete x86-64 theorem |
| compiler transfer | exact local target command, source/object/disassembly hashes, Capstone decode, no object execution, explicit unavailable cells | capability receipt only; no compiler-generalization or application claim |
| sensing costs | deterministic vectors, machine-specific calibration, exact finite dynamic-program objective and gap | finite benchmark certificate only; no universal optimality law |
| external static corpus | `{{EXTERNAL_STATIC_ROW_COUNT}}` pinned source rows, archive/license digests, static compilation, disassembly hashes, Capstone coverage, and explicit failed rows | static coverage and lineage only; no application effectiveness |
| source-bound label contract | `{{EXTERNAL_LABEL_ROW_COUNT}}` accepted rows across `{{EXTERNAL_LABEL_PROJECT_COUNT}}` projects and two revisions per project: `{{EXTERNAL_LABEL_EQUIVALENT_COUNT}}` lexical-equivalence witnesses, `{{EXTERNAL_LABEL_NEAR_MISS_COUNT}}` contract-directed near misses, and `{{EXTERNAL_LABEL_NEGATIVE_COUNT}}` identity negative controls; base/pair source, object, and disassembly hashes; project-held-out test split | source-contract and static provenance evidence only; lexical comments are not universal equivalence proofs, near misses are not executed behavioral labels, and no application claim follows |
| replication bundle | public/gold field separation, public protocol/toolchain/instruction hashes, identifier/hash equality, no-network replay metadata | frozen interface and local replay contract; no independent implementation claim |
| replay ingestion | protected scoring boundary, explicit separate-control attestation, mismatch retention, and source-bound receipt hashes | `{{INDEPENDENT_REPLAY_STATUS}}` in `{{INDEPENDENT_REPLAY_CONTROL_SCOPE}}`; observed controlled replay does not imply independent physical-machine replication or independent reimplementation |
| theorem phase | checked Lean theorem list plus generated bounded counterexample | fail-closed; unresolved claims remain unresolved |
| external preparation audit | eight preparation steps (manifest parse, archive digest verify, archive-safety rehearsal, license check, static compile, build reproducibility, label contract, exclusion accounting) each validated without execution or network use | rehearsal-only preparation path; not an efficacy claim; promotion requires MAJ-01 independent replay of the acquisition/build recipe |

The registry and figure validator apply the same source-bound rule to the
visualization suite. A caption is not accepted merely because it exists: its
input artifacts, claim IDs, uncertainty method, sample hierarchy, units, and
alt text are generated from the figure registry. Missing, unseen, unsupported,
and unavailable cells must be described rather than drawn as zero. The
compiler, cost, correspondence, quotient, and replication panels are therefore
part of the analysis validation surface, not decorative post-processing.
