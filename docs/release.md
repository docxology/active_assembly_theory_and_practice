# Review package and archival release

The review object is version 0.1.0 of the bounded technical monograph and
computational demonstrator. Publication makes a frozen package citable; it
does not promote its scientific evidence level. Citation metadata is in
[`CITATION.cff`](../CITATION.cff), deposition metadata in
[`.zenodo.json`](../.zenodo.json), and project-owned work is covered by
[`LICENSE`](../LICENSE). Third-party source archives are not relicensed or
bundled into the release; their acquisition manifests retain upstream licenses
and digests.

The production archival identifier for this version is
[10.5281/zenodo.23484108](https://doi.org/10.5281/zenodo.23484108).
Its [Zenodo record](https://zenodo.org/records/23484108) is the authority for
publication state and deposited files. The identifier binds the full research
archive; it does not certify independent replication or scientific validity.

## Reviewer reading route

1. Read the combined PDF for the scientific argument and its limits.
2. Use the HTML manuscript and figure atlas for searchable text, alternatives,
   full-size figures and vector exports. Dense figures should be inspected at
   native size rather than judged from thumbnails.
3. Inspect the executable GNN contract, runtime catalog and source capsules
   before interpreting an inference plot.
4. Compare claim and validation receipts with the stated scientific boundaries.
5. Reproduce with the locked environments and pinned native tools described in
   [getting started](getting_started.md) and [verification](verification.md).

The release archive preserves the following relative paths:

| Surface | Archive path | Interpretation |
| --- | --- | --- |
| Primary monograph | `output/pdf/active_assembly_theory_and_practice_combined.pdf` | Primary review document; accessible tagged PDF. |
| Editable projection | `output/docx/active_assembly_theory_and_practice_combined.docx` | Structurally checked export; Word-specific pagination is not certified. |
| Searchable manuscript | `output/web/index.html` | Rendered text, mathematical notation and responsive tables. |
| Scientific figure atlas | `output/figure_atlas/index.html` | Captions, alternatives, source notes and native PNG/SVG/PDF links. |
| Executable model catalog | `output/gnn/inference/README.md` | Actual model profiles, parsed runtime instances and topology exports. |
| Claim and validation receipts | `output/data/claim_register.json`, `output/data/validation_report.json` | Bounded source-bound checks, with explicit status and limitations. |
| Companion slide editions | `output/slides/` | Automatic document projections with continuation frames; not curated talks. |

The archive checksum manifest binds exact file bytes. A source archive alone
does not contain ignored generated outputs. The review bundle must include
the listed outputs and their supporting figure/model assets, not only a PDF
or a GitHub-generated source ZIP.

## Real GNN and Active Inference

Every inference adapter consumes numeric bindings produced by the actual
configured upstream GNN parser. This does not make every statistical comparator
an Active Inference agent. The [execution contract](gnn_models.md) and
[model specifications](../configs/gnn/README.md) define the distinctions:

| Path | Actual execution | Evidence and boundary |
| --- | --- | --- |
| Passive phase and prospective sensor models | Real `pymdp.Agent` consumes declared A/B/C/D/E tensors; sensing decisions precede acquisition. | `output/gnn/passive_run.json`, `sensor_run.json`, trajectories and intervention controls. Fixture identity is attached only for evaluation. |
| Shared categorical native companion | Real Julia/RxInfer graph consumes the same exported tensors. | `output/gnn/native_comparison.json`; independent scaled-forward smoothing reference. Prefix evidence and stationary iterations remain distinct. |
| Fitted, factorized, hierarchical, regression and calibration diagnostics | Explicit project algorithms consume actual upstream-parsed numeric bindings. | `output/data/inference_model_catalog.json` and retained runtime source capsules. GNN admission is not universal upstream code generation or learned semantics. |
| Gaussian native diagnostic | Real Julia/RxInfer model with source digest and prior checks. | Separate objective/scale; it must not be compared numerically as if it were categorical policy EFE. |

The qualified October 19 build records 14 profiles and 2,299 numeric runtime
instances. Native categorical smoothing differs from the independent reference
by at most `1.1102230246251565e-16`. These are observations about this frozen
build, not tolerances promised for arbitrary configurations. A/B/C/D/E
interventions, hidden-fixture invariance, fixed-policy baselines and malformed
distribution rejection remain part of the executable checks.

## Figures and scholarship

For efficient visual review, inspect `gnn_sensing`,
`rxinfer_categorical_free_energy`, `quotient_classifier`,
`scholarship_evidence_matrix`, `claim_evidence_map` and `validation_summary`
in the atlas. Together these show the prospective decision, the native
inference boundary, a finite benchmark, citation roles, claim lineage and
engineering checks. They do not form six independent scientific confirmations.

Missing and nonfinite metrics fail export instead of becoming zero. Undefined
selective risk and absent replay evidence remain explicit. Scholarship counts
use integer scales and stage totals; sources are not independent replications
or weights of support. Consult [scholarship](scholarship.md),
[statistics](statistics.md) and [visual atlas](visual_atlas.md) for the primary
source review, finite-sample limits and caption contracts. Metadata/link checks
do not amount to full-content peer review of every cited work.

## Qualification and release boundary

The October 19 qualification passed 24/24 stages: 1,294 pre-render Python
tests and two post-render tests passed, with 90.31% combined statement/branch
coverage. One storage-capability test skipped because its required external
volume was unavailable. Strict source-bound validation passed 195/195 checks;
native Julia tests, Lean, PDF syntax and PDF/UA-2 checks passed. Sixteen
pymdp/JAX static-array warnings were retained. The independent gate approved
the bounded demonstrator and primary PDF. No hosted-CI success is inferred.

The full research source includes authored fixtures and owner-side contracts.
It is not the label-blind operator handoff. Protected generated gold labels,
private replay directories, environments, caches, acquired source archives,
temporary files and raw local logs must not enter a public review bundle.
Follow the [separate replay runbook](replay_operator_runbook.md) for a filtered
handoff. Independent physical replication, independent reimplementation and
general application validity remain unearned.

## DOI and publication procedure

Use a production Zenodo record, reserve its DOI before publication, and retain
the server-issued identifier. Add that exact identifier to citation metadata
and release notes; an invented, sandbox or merely reserved DOI is not a
published archive. Zenodo describes [DOI reservation](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/)
and its authenticated [deposit API](https://developers.zenodo.org/).

Bind the release tag to the reviewed source commit and include checksums for
the source snapshot, primary PDF, DOCX, review bundle and verification record.
After publishing, retrieve the record and verify its DOI, version, authors,
license, filenames and server checksums. Preserve repository visibility unless
the owner explicitly requests a visibility change. A private GitHub release
does not itself create a Zenodo DOI or a publicly accessible source archive.
