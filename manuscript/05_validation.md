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

![Counts of sources in the curated scholarship ledger by source type. The ledger separates peer-reviewed theory and binary-analysis context from software papers, package documentation, and roadmap references; counts describe support roles rather than evidence strength by themselves.]({{SCHOLARSHIP_FIGURE}}){#fig:scholarship-source-roles width=92%}

![Scholarship-source counts crossed by project stage and evidence tier. Implemented and comparison rows contextualize executed paths, while roadmap and context rows remain explicitly outside the currently executed evidence-producing surface.]({{SCHOLARSHIP_MATRIX_FIGURE}}){#fig:scholarship-evidence-matrix width=92%}

![Claim-register coverage map. For each bounded claim, bars count linked source keys, generated artifacts, and validation checks; these counts audit traceability and do not turn a linked claim into independent scientific replication.]({{CLAIM_FIGURE}}){#fig:claim-evidence-map width=95%}

![Reproducibility-manifest coverage by artifact class. Bar height is total tracked file size in KiB and annotations give file counts; each file is independently recorded with SHA-256 and byte size.]({{REPRODUCIBILITY_FIGURE}}){#fig:artifact-hash-manifest width=92%}

The analysis validator reports `{{VALIDATION_PASS_COUNT}}` passing checks out
of `{{VALIDATION_TOTAL_COUNT}}`, with `{{VALIDATION_FAIL_COUNT}}` failures and
`all_passed = {{VALIDATION_ALL_PASSED}}`. The report is written to
`{{VALIDATION_REPORT}}`. It checks trace schema fields, sample hashes, required
package provenance, trace observation semantics, dynamic-register trace shape,
RxInfer free-energy finiteness, RxInfer companion artifact consistency,
categorical-companion model identity, memory-operand to dynamic-event joins,
data-hash consistency, reproducibility-manifest hashes and byte sizes, figure
nonblankness, source-ledger bibliography coverage, manuscript citation usage,
citation closure, source-shape quality, source-stage and evidence-tier
partitioning, claim-register links, fixed-parameter generative adequacy, and
cautious claim-boundary language.

![Generated validation checks for trace structure, assembly-listing lineage, inference artifacts, figures, scholarship, manuscript hydration, provenance, and reproducibility. A pass means the named project oracle evaluated true; it is not a substitute for scientific peer review.]({{VALIDATION_FIGURE}}){#fig:validation-summary width=96%}

This validation layer is intentionally adversarial. It is designed to fail if a
future edit reports RxInfer free energy without finite JSONL values, adds a
source-ledger entry without a bibliography record, drops a required adapter
provenance row, changes observation labels without updating category IDs, drops
RxInfer posterior/configuration artifacts, changes a manifest-tracked artifact
after hash recording, adds an unlinked claim, or renders empty figures while
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
| quotient classifier | protocol hash, label-blind features, disjoint training/development/held-out families, finite metrics and clustered intervals | bounded endpoint only; no semantic-discovery or superiority claim without the preregistered margin and independent replay |
| formal correspondence | real Capstone decode, bounded Unicorn trace, Python reference step alignment, modeled flags, and Lean prerequisite | declared transparent subset only; no complete x86-64 theorem |
| compiler transfer | exact local target command, source/object/disassembly hashes, Capstone decode, no object execution, explicit unavailable cells | capability receipt only; no compiler-generalization or application claim |
| sensing costs | deterministic vectors, machine-specific calibration, exact finite dynamic-program objective and gap | finite benchmark certificate only; no universal optimality law |
| replication bundle | public/gold field separation, identifier/hash equality, no-network replay metadata | local harness only until independent replay is observed |
| theorem phase | checked Lean theorem list plus generated bounded counterexample | fail-closed; unresolved claims remain unresolved |

The registry and figure validator apply the same source-bound rule to the five
new visualizations. A caption is not accepted merely because it exists: its
input artifacts, claim IDs, uncertainty method, sample hierarchy, units, and
alt text are generated from the figure registry. Missing, unseen, unsupported,
and unavailable cells must be described rather than drawn as zero. The
compiler, cost, correspondence, quotient, and replication panels are therefore
part of the analysis validation surface, not decorative post-processing.
