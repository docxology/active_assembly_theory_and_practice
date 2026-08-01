# RedTeam Findings and Limits {#sec:redteam-limits}

The second-pass RedTeam review records `{{REDTEAM_FINDING_COUNT}}` artifact-level
findings in `{{REDTEAM_REPORT}}`, with `{{REDTEAM_REMEDIATED_COUNT}}`
remediated by validation checks or generated artifacts. The findings target
false-certification paths rather than broad rhetorical objections: source
entries with malformed DOI or URL fields, citations that do not resolve through
`references.bib`, omitted manuscript sections or figure references, blank figure
files, trace observations whose labels no longer match their categorical IDs,
RxInfer companion artifacts that drift from finite free-energy records, and
scholarship entries that fail to distinguish implemented evidence from roadmap
context. The fourth pass adds claim-register closure so manuscript claims must
link to source keys, generated artifacts, validation checks, and boundary text.
The current pass adds artifact-freshness closure: generated outputs must remain
consistent with their recorded file sizes and SHA-256 digests. Critique is
required to appear in the project output surface rather than only in chat.

![RedTeam risk register showing remediated artifact-level findings.]({{REDTEAM_FIGURE}}){#fig:redteam-risk-register}

The remaining scientific limits are substantive and now more precisely bounded.
The RxInfer backend is a Gaussian event-code demonstrator, not yet a categorical
program-phase HMM. The corpus is transparent and small, not a representative
binary corpus, and its ground-truth phase labels are hand-authored rather than
learned. The head-to-head result this corpus produces — the temporal HMM beats
the v0.1 sticky filter but not the memoryless per-step decoder — is now
explained *mechanistically* by the ambiguity ablation (CLM-014): at the
authored, near-diagonal likelihood a single observation already reveals the
phase, so transition structure adds little, and the temporal advantage appears
only as observations become ambiguous. Because that ablation sweeps the
likelihood conditioning synthetically rather than measuring real observation
noise, its verdict is a mechanism demonstration, not an empirical estimate of
real-world ambiguity. Phase segmentation is now *measured* against the
hand-authored ground truth (CLM-015) rather than merely asserted to be
self-consistent. Optional adapters remain provenance slots until they produce
concrete facts.

The project therefore supports a narrow claim: assembly traces can be converted
into auditable observations for explicit variational inference workflows. It
does not support claims of full reverse engineering, malware classification,
semantic decompilation, or autonomous security triage. Natural next steps are a
categorical-HMM RxInfer backend, a larger and deliberately ambiguous corpus, and
parameter learning that replaces the hand-authored beliefs with estimates fit to
data.
