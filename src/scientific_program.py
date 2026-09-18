"""Machine-readable long-horizon scientific program and promotion gates.

The current package is an auditable demonstrator, not a discovery claim.  This
module makes the ambitious successor program executable as a *research
contract*: every proposed advance has a question, a conjecture, a primary
endpoint, a falsifier, an explicit sampling unit, dependencies, and integrity
gates.  The generated artifact is a roadmap receipt, not empirical evidence.

Keeping this contract in ``src/`` is intentional.  A roadmap that cannot be
serialized, checked for dependency errors, and bound to the same reproducible
artifact surface is too easy to rewrite after seeing results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .artifact_contract import write_json_artifact
from .provenance import generator_revision, sha256_path, toolchain_versions
from .receipt import build_receipt_envelope, validate_receipt_envelope
from .scientific_protocol import scientific_protocol_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCIENTIFIC_PROGRAM_SCHEMA_VERSION = "2.0.0"
SCIENTIFIC_PROGRAM_ARTIFACT = "output/data/scientific_program.json"

PROGRAM_STATUSES = frozenset(
    {"implemented_local", "active", "planned", "blocked_by_evidence"}
)


@dataclass(frozen=True)
class ScientificMilestone:
    """One falsifiable milestone in the long-horizon research program."""

    milestone_id: str
    phase: str
    title: str
    status: str
    horizon: str
    scientific_question: str
    conjecture: str
    primary_endpoint: str
    falsifier: str
    sampling_unit: str
    dependencies: tuple[str, ...]
    required_evidence: tuple[str, ...]
    integrity_gates: tuple[str, ...]
    promotion_rule: str
    stop_conditions: tuple[str, ...]
    current_result: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe milestone record."""

        data = asdict(self)
        for key in (
            "dependencies",
            "required_evidence",
            "integrity_gates",
            "stop_conditions",
        ):
            data[key] = list(data[key])
        return data


def _milestones() -> tuple[ScientificMilestone, ...]:
    """Return the canonical staged program.

    The language is deliberately stronger than a feature backlog and weaker
    than a scientific conclusion: future rows are conjectures with defined
    ways to fail, while implemented rows describe only the evidence already
    earned by this repository.
    """

    return (
        ScientificMilestone(
            milestone_id="P0",
            phase="measurement",
            title="Source-bound assembly observation bridge",
            status="implemented_local",
            horizon="current",
            scientific_question=(
                "Can machine-code events be converted into auditable observations "
                "without confusing decoding, execution, and interpretation?"
            ),
            conjecture=(
                "A bounded, provenance-carrying Capstone/Unicorn trace can serve as "
                "a reproducible measurement interface for a generative model."
            ),
            primary_endpoint=(
                "Exact static/dynamic alignment, independent bounded semantic "
                "cross-check, deterministic artifact regeneration, and complete "
                "negative-control reporting."
            ),
            falsifier=(
                "Any unsupported instruction receives an invented meaning, or a "
                "trace/artifact can pass while its source bytes, addresses, or "
                "package provenance disagree."
            ),
            sampling_unit="transparent fixture program, not an instruction step",
            dependencies=(),
            required_evidence=(
                "real Capstone decoding",
                "bounded Unicorn execution",
                "source-bound trace and package provenance",
                "artifact hashes and deterministic regeneration",
            ),
            integrity_gates=(
                "transparent toy bytes only",
                "unsupported operations fail closed",
                "no arbitrary binary execution",
                "static and dynamic evidence remain distinct",
            ),
            promotion_rule=(
                "Keep this as a local measurement contract; do not promote it to "
                "program understanding or general reverse engineering."
            ),
            stop_conditions=(
                "trace alignment is not exact",
                "semantic cross-check disagrees with bounded execution",
            ),
            current_result=(
                "Implemented and validated on the transparent x86-64 fixture "
                "surface; not external validity."
            ),
        ),
        ScientificMilestone(
            milestone_id="P1",
            phase="mechanism",
            title="Temporal inference under controlled ambiguity",
            status="implemented_local",
            horizon="current",
            scientific_question=(
                "When does temporal structure provide information that a memoryless "
                "observation decoder cannot recover?"
            ),
            conjecture=(
                "Transition structure helps only when the observation channel is "
                "sufficiently ambiguous; at a nearly diagonal channel it should "
                "not be credited with a temporal advantage."
            ),
            primary_endpoint=(
                "Predeclared temporal-minus-memoryless accuracy and segmentation "
                "difference across a likelihood-ambiguity intervention."
            ),
            falsifier=(
                "The crossover cannot be reproduced, the strong memoryless baseline "
                "is omitted, or the result depends on post-hoc blend selection."
            ),
            sampling_unit="fixture-level trace under a declared likelihood intervention",
            dependencies=("P0",),
            required_evidence=(
                "memoryless, temporal, and identity-transition comparators",
                "negative controls",
                "controlled ambiguity and transition ablations",
                "descriptive uncertainty with fixture dependence disclosed",
            ),
            integrity_gates=(
                "same A and D when isolating B",
                "no instruction-step pseudoreplication",
                "both positive and null directions reported",
                "intervention grid fixed before headline scoring",
            ),
            promotion_rule=(
                "Treat the result as a mechanism demonstration until it survives "
                "independent programs and representation changes."
            ),
            stop_conditions=(
                "temporal advantage appears only after selecting a favorable grid",
                "negative controls are not materially below the model",
            ),
            current_result=(
                "Implemented: temporal structure wins under controlled ambiguity but "
                "not at the authored near-diagonal likelihood."
            ),
        ),
        ScientificMilestone(
            milestone_id="P2",
            phase="latent-parameter",
            title="Leakage-safe observation-only parameter learning",
            status="implemented_local",
            horizon="current",
            scientific_question=(
                "Can ordered latent-state parameters be estimated from observations "
                "without using held-out phase labels?"
            ),
            conjecture=(
                "Baum--Welch EM can fit an ordered latent-state scaffold, but a "
                "reference initialization and state-order prior may prevent semantic "
                "identifiability."
            ),
            primary_endpoint=(
                "Complete-fixture leave-one-out scores, training evidence histories, "
                "convergence diagnostics, and an executable label-leakage contract."
            ),
            falsifier=(
                "Held-out labels enter fitting, model tensors are stale or malformed, "
                "or the result is reported as semantic discovery despite the ordered "
                "scaffold and negative held-out comparison."
            ),
            sampling_unit="complete fixture held out from both observations and labels",
            dependencies=("P0", "P1"),
            required_evidence=(
                "scaled forward/backward EM implementation",
                "supervised and fixed-reference comparators",
                "training/held-out membership receipt",
                "malformed-input and leakage-negative tests",
            ),
            integrity_gates=(
                "labels are scoring-only",
                "complete fixtures, not correlated steps, are holdouts",
                "convergence and non-convergence are both visible",
                "negative result cannot be rounded into a success claim",
            ),
            promotion_rule=(
                "Use this as a parameter-learning and leakage demonstration; do not "
                "call it unsupervised semantic discovery."
            ),
            stop_conditions=(
                "any training row contains the held-out fixture",
                "state coordinates are silently relabelled or hand-aligned",
            ),
            current_result=(
                "Implemented: observation-only EM underperforms supervised MAP and "
                "the fixed reference on the six-fixture surface."
            ),
        ),
        ScientificMilestone(
            milestone_id="P3",
            phase="identifiability",
            title="Quotient-state identifiability and representation invariance",
            status="implemented_local",
            horizon="next",
            scientific_question=(
                "Can latent computational motifs be defined as equivalence classes of "
                "operational behavior rather than as hand-authored phase names?"
            ),
            conjecture=(
                "Given a separating family of safe interventions, observationally "
                "equivalent programs are identifiable up to quotient-state relabeling, "
                "and semantics-preserving representation changes preserve the posterior "
                "over those quotient states."
            ),
            primary_endpoint=(
                "Pre-registered adjusted mutual information and posterior calibration "
                "after optimal label alignment, measured across source-program, compiler, "
                "architecture, and semantics-preserving transformation holdouts."
            ),
            falsifier=(
                "Non-equivalent motifs remain observationally indistinguishable under "
                "the declared intervention family, or equivalent programs receive "
                "systematically different posterior states after controlling for sensor "
                "coverage and uncertainty."
            ),
            sampling_unit=(
                "source program × compiler/optimization × architecture × "
                "semantics-preserving transformation"
            ),
            dependencies=("P0", "P1", "P2"),
            required_evidence=(
                "formal equivalence classes with executable witnesses",
                "intervention catalog and separating-power analysis",
                "multi-seed state-alignment protocol",
                "cross-domain and transformation holdouts",
                "strong mnemonic and representation baselines",
            ),
            integrity_gates=(
                "quotient labels are generated from semantics, not model output",
                "test programs are frozen before final scoring",
                "splits block source and compiler leakage",
                "uncertainty is clustered at the source-program unit",
                "counterexamples are retained and published",
            ),
            promotion_rule=(
                "Promote only if invariance and identifiability survive preregistered "
                "negative controls, multiple seeds, and an independent reimplementation."
            ),
            stop_conditions=(
                "performance is explained by mnemonic or compiler identity",
                "state alignment requires inspecting held-out labels",
                "equivalence witnesses fail under bounded semantic checking",
            ),
            current_result=(
                "The expanded classifier generator now defines 64 independent source families "
                "with 64 rows per family, balanced equivalent/near-miss roles, and a 32/16/16 "
                "family split. The separate representation-invariance witness remains an eight-family "
                "pilot. This does not establish identifiability or semantic discovery."
            ),
        ),
        ScientificMilestone(
            milestone_id="P4",
            phase="formal-semantics",
            title="Mechanically verified operational semantics",
            status="implemented_local",
            horizon="next",
            scientific_question=(
                "Can the observation bridge and its equivalence relation be stated and "
                "proved in a small mechanized operational-semantics core?"
            ),
            conjecture=(
                "For a declared instruction subset, a machine-checked small-step semantics "
                "can prove trace refinement, bounded interpreter soundness, and invariance "
                "of the observation quotient under approved transformations."
            ),
            primary_endpoint=(
                "A complete proof artifact for the subset, exhaustive bounded-state tests, "
                "and a fail-closed correspondence theorem between semantics and trace rows."
            ),
            falsifier=(
                "A counterexample is found within the declared subset, or the proof relies "
                "on an undocumented axiom equivalent to assuming the desired result."
            ),
            sampling_unit="instruction-sequence semantics instance with bounded machine state",
            dependencies=("P0", "P3"),
            required_evidence=(
                "machine-checked definitions",
                "proof object and replay command",
                "exhaustive bounded enumeration where feasible",
                "independent executable cross-check against Unicorn",
            ),
            integrity_gates=(
                "axioms and trusted kernels are enumerated",
                "unsupported instructions are explicit",
                "proof and executable semantics share versioned fixtures",
                "no proof by agreement with one implementation alone",
            ),
            promotion_rule=(
                "A theorem is promoted only with a public proof artifact, counterexample "
                "search, and an independently replayable kernel/toolchain."
            ),
            stop_conditions=(
                "the formal subset is too narrow to test the conjecture",
                "unproved semantic assumptions are hidden behind an adapter boundary",
            ),
            current_result=(
                "Implemented bounded tranche: transparent fixtures are decoded with real "
                "Capstone, executed with bounded Unicorn, and matched step-by-step by the "
                "Python reference path for the declared fixture subset. A pinned Lean project "
                "also compiles a kernel-checked Word64 fragment with equivalence and "
                "counterexample theorems. This is not a broad x86-64 mechanization or a theorem "
                "that the Python path and Lean path are identical."
            ),
        ),
        ScientificMilestone(
            milestone_id="P5",
            phase="external-validity",
            title="Cross-architecture and cross-compiler invariance benchmark",
            status="implemented_local",
            horizon="long",
            scientific_question=(
                "Does the proposed latent representation survive changes in ISA, compiler, "
                "optimization, and instruction-level encoding?"
            ),
            conjecture=(
                "A semantics-grounded quotient representation transfers better than mnemonic "
                "or surface-feature baselines across x86-64, AArch64, RISC-V, and a declared "
                "portable intermediate representation."
            ),
            primary_endpoint=(
                "Pre-registered cross-domain generalization, calibration, invariance gap, "
                "and failure-stratified performance on source-disjoint programs."
            ),
            falsifier=(
                "The proposed representation fails to beat surface baselines after strict "
                "source/compiler splits, or its apparent transfer is explained by duplicated "
                "templates, instruction vocabulary, or metadata leakage."
            ),
            sampling_unit="source-level program family, held out across compiler and architecture",
            dependencies=("P3", "P4"),
            required_evidence=(
                "publicly documented corpus construction",
                "architecture-specific semantics adapters",
                "source/compiler/architecture-disjoint splits",
                "surface, learned, and semantics-based baselines",
                "external data audit and license/provenance receipt",
            ),
            integrity_gates=(
                "no train/test near-duplicate programs",
                "architecture metadata is not a hidden label",
                "safe non-execution or sandbox contract for external binaries",
                "all missing/unsupported effects remain explicit",
            ),
            promotion_rule=(
                "External-validity language is permitted only after the benchmark is public, "
                "frozen, and independently audited."
            ),
            stop_conditions=(
                "source or compiler leakage is detected",
                "license/provenance cannot be reconstructed",
                "external binaries require unsafe execution to obtain the claimed result",
            ),
            current_result=(
                "The compiler transfer capability receipt freezes a matrix over five "
                "toolchains, ten source families, thirteen target triples, Mach-O and ELF "
                "object formats, and per-compiler optimization vocabularies (Clang -O0..-Oz, "
                "NASM unoptimized, rustc opt-level 0/1/2/3/s/z, Zig "
                "Debug/ReleaseSafe/ReleaseFast/ReleaseSmall declared). On the recorded machine "
                "128 of 184 cells validated: Apple Clang 21 across x86_64/aarch64 Mach-O and "
                "x86_64-unknown-linux-gnu/aarch64-unknown-linux-gnu ELF, NASM 3.02 in macho64 "
                "and elf64, and rustc 1.95 as a no_std staticlib on the host aarch64 target. "
                "Every non-validated cell is an explicit unavailable row carrying the requested "
                "triple and a reason (RISC-V unsupported by the local Clang, the "
                "x86_64-apple-darwin rustc standard library not installed, and go/zig declared "
                "but not compiled); no toolchain, target, or architecture was downloaded or "
                "substituted, and no emitted object was ever executed."
            ),
        ),
        ScientificMilestone(
            milestone_id="P6",
            phase="active-sensing",
            title="Information-per-cost law for active program observation",
            status="implemented_local",
            horizon="long",
            scientific_question=(
                "Can active sensing of program evidence be characterized by a theorem or "
                "tight bound rather than by a policy-specific score?"
            ),
            conjecture=(
                "For a specified observation channel and intervention cost, an active policy "
                "can approach a non-trivial information-per-cost frontier while fixed and "
                "random policies obey a measurable lower-performance bound."
            ),
            primary_endpoint=(
                "Regret to an optimal or certified near-optimal policy, information gained per "
                "measured cost, and confidence intervals clustered by independent source program."
            ),
            falsifier=(
                "The policy does not beat strong fixed/random controls under measured costs, "
                "or the claimed bound is violated by a reproducible counterexample."
            ),
            sampling_unit="independent source program under a fixed sensor budget",
            dependencies=("P3", "P5"),
            required_evidence=(
                "measured acquisition costs",
                "admissible action set and sensor model",
                "optimality certificate or bounded exhaustive policy search",
                "oracle retained only as an inadmissible upper bound",
                "budget and calibration sensitivity analysis",
            ),
            integrity_gates=(
                "actions cannot inspect unrevealed values",
                "costs are not normalized proxies when runtime claims are made",
                "policy selection is frozen before held-out scoring",
                "oracle cannot be presented as deployable",
            ),
            promotion_rule=(
                "Report a scientific law only if the result survives alternative cost models, "
                "strong baselines, and independent replication."
            ),
            stop_conditions=(
                "the sensor model is not physically or computationally measurable",
                "active selection changes the data-generating process in an unreported way",
            ),
            current_result=(
                "Implemented finite tranche: deterministic eight-component sensing vectors, "
                "local wall-clock calibration, and exact dynamic-programming certificates are "
                "generated for the frozen finite sensor space. The tolerance and zero-gap "
                "certificates are scoped to that finite benchmark; no general information-"
                "per-cost law or universal optimality claim is made."
            ),
        ),
        ScientificMilestone(
            milestone_id="P7",
            phase="replication",
            title="Independent implementation and open challenge",
            status="implemented_local",
            horizon="long",
            scientific_question=(
                "Does the central result survive an independently written implementation, "
                "frozen data, and adversarial analysis?"
            ),
            conjecture=(
                "The key invariance/identifiability and active-sensing results are not artifacts "
                "of one codebase, one author, or one analysis path."
            ),
            primary_endpoint=(
                "Blind reproduction of preregistered primary endpoints and failure cases by an "
                "independent team using only the public protocol and frozen benchmark."
            ),
            falsifier=(
                "An independent implementation cannot reproduce the primary endpoint, or the "
                "result changes materially after access to the original analysis."
            ),
            sampling_unit="independent implementation × frozen source-program benchmark",
            dependencies=("P3", "P4", "P5", "P6"),
            required_evidence=(
                "preregistration and frozen test set",
                "independent code and environment",
                "adversarial review, counterexamples, and falsification tests",
                "full null and failure-case publication",
            ),
            integrity_gates=(
                "original authors do not edit the independent result",
                "primary endpoints cannot be changed after unblinding",
                "all deviations are logged",
                "negative replication is publishable and preserved",
            ),
            promotion_rule=(
                "No broad scientific claim, award-level narrative, or deployment recommendation "
                "is permitted before independent replication."
            ),
            stop_conditions=(
                "the benchmark cannot be frozen reproducibly",
                "independent reviewers cannot reconstruct the data lineage",
            ),
            current_result=(
                "Implemented local tranche: a frozen CLI generator, label-separated public and "
                "gold manifests, hash validation, and no-network replay contract are available. "
                "A same-host separately controlled Lima replay has also been observed and protected-scored; "
                "the harness is not an independent implementation, physical-machine replication, or completed public challenge."
            ),
        ),
        ScientificMilestone(
            milestone_id="P8",
            phase="theory",
            title="General theorem or decisive countertheorem",
            status="implemented_local",
            horizon="long",
            scientific_question=(
                "What mathematically general statement connects operational semantics, "
                "observation quotients, identifiability, and active information acquisition?"
            ),
            conjecture=(
                "There exists a non-trivial theorem characterizing when a finite intervention "
                "family identifies computational motifs up to semantic equivalence and when an "
                "active policy is information-theoretically efficient."
            ),
            primary_endpoint=(
                "A peer-reviewed theorem with explicit assumptions, proof, tightness or lower "
                "bound where possible, counterexamples outside the assumptions, and independent "
                "mechanized or line-by-line verification."
            ),
            falsifier=(
                "A counterexample invalidates the conjecture, the theorem is only a restatement "
                "of definitions, or the assumptions are so strong that the result has no reach."
            ),
            sampling_unit="mathematical model class; empirical examples are validation, not proof",
            dependencies=("P3", "P4", "P5", "P6", "P7"),
            required_evidence=(
                "formal definitions and theorem statement",
                "proof and counterexample suite",
                "mechanized proof or independently audited proof artifact",
                "empirical demonstrations across domains",
            ),
            integrity_gates=(
                "the theorem is separated from empirical fit claims",
                "all assumptions are stated before consequence claims",
                "negative and boundary cases are part of the main result",
                "mathematicians outside the project review the proof",
            ),
            promotion_rule=(
                "Only a genuinely broad theorem, or a decisive countertheorem that changes the "
                "field's understanding, can support Fields-level mathematical ambition."
            ),
            stop_conditions=(
                "the result is merely an engineering benchmark improvement",
                "the proof depends on undocumented computational behavior",
                "empirical examples are substituted for the theorem itself",
            ),
            current_result=(
                "Implemented fail-closed tranche: the candidate theorem phase records a checked "
                "closed-world Lean fragment, a finite computational validation, and an automatic "
                "single-intervention counterexample. The broad theorem remains unresolved and no "
                "theorem-shaped manuscript claim is promoted."
            ),
        ),
        ScientificMilestone(
            milestone_id="P9",
            phase="identifiability",
            title="Finite intervention power and separating sets",
            status="implemented_local",
            horizon="next",
            scientific_question=(
                "How many declared initial-state interventions are required to separate the "
                "finite quotient benchmark's equivalent and one-change near-miss pairs?"
            ),
            conjecture=(
                "Within a fixed finite catalog, separating-set cardinality is a measurable "
                "property of the intervention-outcome matrix, but it need not imply semantic "
                "identifiability outside that catalog."
            ),
            primary_endpoint=(
                "Exact enumeration of every non-empty intervention subset, its equivalent-pair "
                "consistency and near-miss separation, and the minimum-cardinality fully separating "
                "sets, including an explicitly non-separating negative control."
            ),
            falsifier=(
                "A subset is omitted or duplicated, the outcome matrix cannot be regenerated from "
                "the formal benchmark, a negative control is silently removed, or a broad "
                "identifiability claim is inferred from the finite result."
            ),
            sampling_unit="declared transformation pair × intervention subset in the finite catalog",
            dependencies=("P3", "P4"),
            required_evidence=(
                "source-bound intervention-power receipt",
                "all non-empty subset rows and pairwise separators",
                "single-state negative-control analysis",
                "fail-closed promotion status and reproducible digest",
            ),
            integrity_gates=(
                "subsets are enumerated from the formal outcome matrix, not model predictions",
                "equivalent and near-miss labels remain separate from classifier inputs",
                "unresolved controls remain visible",
                "finite separation is not called universal identifiability",
            ),
            promotion_rule=(
                "Promote only a finite benchmark diagnostic unless a stronger theorem supplies "
                "assumptions and independent evidence for generalization."
            ),
            stop_conditions=(
                "the declared intervention catalog changes after outcomes are inspected",
                "a separator depends on a hidden label or transformation identifier",
                "the finite matrix is incomplete or irreproducible",
            ),
            current_result=(
                "Implemented exact finite analysis: all 15 non-empty subsets of four interventions "
                "are enumerated. Three singleton interventions fully separate the current 16-pair "
                "catalog, while the all-zero negative control leaves three near-miss families "
                "unresolved. The receipt explicitly blocks promotion to semantic identifiability."
            ),
        ),
        ScientificMilestone(
            milestone_id="P10",
            phase="latent-parameter-robustness",
            title="Initialization robustness of observation-only parameter learning",
            status="implemented_local",
            horizon="next",
            scientific_question=(
                "How sensitive is observation-only ordered-HMM parameter learning to deterministic "
                "initialization on complete-fixture holdouts?"
            ),
            conjecture=(
                "A single observation-only fit is insufficient evidence of a stable latent-parameter "
                "estimate; multi-start agreement and held-out performance spread must be reported "
                "before any mechanistic interpretation."
            ),
            primary_endpoint=(
                "Held-out accuracy, convergence diagnostics, and pairwise label agreement across a "
                "frozen reference, uniform, and three deterministic random initializations for every "
                "complete-fixture holdout."
            ),
            falsifier=(
                "Held-out phase labels enter fitting, initialization choices are tuned on the test "
                "fixture, unstable starts are hidden, or the multi-start rows cannot be reproduced "
                "from the protocol and training complements."
            ),
            sampling_unit="held-out transparent fixture × deterministic initialization",
            dependencies=("P2",),
            required_evidence=(
                "multi-start parameter-learning receipt",
                "fit-input leakage guard",
                "per-initialization held-out scores and diagnostics",
                "seed sensitivity summary separated from sampling uncertainty",
            ),
            integrity_gates=(
                "phase labels are used only for scoring",
                "development and held-out fixtures remain outside fitting",
                "all declared starts are reported, including unstable or weak starts",
                "robustness is not presented as latent-semantic discovery",
            ),
            promotion_rule=(
                "Report initialization sensitivity as a robustness result; do not promote a model "
                "to a scientific explanation without independent data and external validation."
            ),
            stop_conditions=(
                "a requested start fails without an explicit underpowered or unsupported state",
                "the seed ledger or fit-input contract is incomplete",
                "model selection uses held-out labels or post-hoc start selection",
            ),
            current_result=(
                "Implemented multi-start tranche: 30 observation-only fits across six holdouts and "
                "five fixed starts are retained. The current accuracy range and modest pairwise label "
                "agreement expose material initialization sensitivity, so no stability or semantic "
                "promotion is made."
            ),
        ),
        ScientificMilestone(
            milestone_id="P11",
            phase="formal-semantics",
            title="Complete normalized machine-state correspondence",
            status="active",
            horizon="current",
            scientific_question=(
                "Can the declared x86-64 integer subset be represented by one explicit "
                "64-bit state and matched step-by-step across Python, Capstone, Unicorn, and Lean?"
            ),
            conjecture=(
                "For the frozen bounded instruction subset, a normalized IR and explicit "
                "machine state admit deterministic trace refinement with no hidden register "
                "or flag semantics."
            ),
            primary_endpoint=(
                "Stepwise agreement of decoded instruction, pre/post state, modeled flags, "
                "memory effects, and control-flow target across the declared fixtures and "
                "exhaustive reduced-state tests."
            ),
            falsifier=(
                "A supported instruction requires a synthetic state encoding, disagrees with "
                "Unicorn on a declared effect, or cannot be represented in the Lean state."
            ),
            sampling_unit="instruction family × bounded machine state × fixture",
            dependencies=("P4",),
            required_evidence=(
                "normalized IR and machine-state schemas",
                "Python/Lean serialized fixture bundle",
                "real Capstone/Unicorn stepwise comparison",
                "exhaustive boundary-value and unsupported tests",
            ),
            integrity_gates=(
                "64-bit arithmetic and operand widths are explicit",
                "INC and DEC preserve CF",
                "no hidden register or negative memory sentinel exists",
                "undefined and unsupported effects fail closed",
            ),
            promotion_rule=(
                "Promote only the declared subset and theorem-backed properties; never "
                "call this complete x86-64 semantics."
            ),
            stop_conditions=(
                "reference and Unicorn traces disagree",
                "Lean and Python serialization diverges",
                "unsupported effects receive invented values",
            ),
            current_result=(
                "An explicit 16-GPR, width-aware bounded Python state now matches the transparent "
                "Capstone/Unicorn fixtures step by step, including flags, memory effects, and branch "
                "targets. Lean compiles a normalized-state schema and boundary theorems; a generated "
                "cross-language proof bridge and exhaustive full-subset coverage remain unresolved."
            ),
        ),
        ScientificMilestone(
            milestone_id="P12",
            phase="transfer-and-identifiability",
            title="Scaled quotient and transfer benchmark",
            status="active",
            horizon="next",
            scientific_question=(
                "Does quotient-conditioned inference remain calibrated when source families, "
                "compiler configurations, optimizations, and architectures are held out?"
            ),
            conjecture=(
                "A semantics-grounded observation channel can preserve more equivalence and "
                "near-miss information than surface baselines under strict source-disjoint splits."
            ),
            primary_endpoint=(
                "Clustered balanced accuracy, macro-F1, adjusted mutual information, calibration, "
                "invariance gap, and near-miss separation on at least 64 independent families."
            ),
            falsifier=(
                "Performance is explained by compiler, architecture, mnemonic, byte, or family "
                "identity, or the expanded benchmark cannot be reproduced from its manifest."
            ),
            sampling_unit="source family × compiler/optimization × architecture × transformation",
            dependencies=("P3", "P5", "P11"),
            required_evidence=(
                "hashed source-family manifest",
                "source/compiler/architecture-disjoint splits",
                "co-primary model and leakage portfolio",
                "clustered statistics and negative controls",
            ),
            integrity_gates=(
                "test families are frozen before final scoring",
                "labels are formal-output-derived and absent from features",
                "unavailable targets remain explicit",
                "underpowered strata cannot be promoted",
            ),
            promotion_rule=(
                "Promote only bounded transfer evidence that meets the preregistered family, "
                "split, uncertainty, and practical-margin gates."
            ),
            stop_conditions=(
                "duplicate or near-duplicate family leakage",
                "a baseline or model cannot be computed",
                "independent-family target is not reached",
            ),
            current_result=(
                "The 64-family, 4096-row classifier benchmark is now generated with the declared "
                "co-primary models and family-clustered statistics. The compiler/architecture matrix "
                "and external-source track remain capability or blocked receipts, so broad transfer "
                "and identifiability promotion remain unearned until the expanded scoring receipt passes."
            ),
        ),
        ScientificMilestone(
            milestone_id="P13",
            phase="external-validity",
            title="Licensed-source static external validation",
            status="blocked_by_evidence",
            horizon="long",
            scientific_question=(
                "Do the source-bound observations and quotient contracts survive labeled, "
                "license-cleared real-source builds without unsafe binary execution?"
            ),
            conjecture=(
                "A carefully documented source-level label contract can support static emitted-object "
                "coverage and project-held-out transfer diagnostics."
            ),
            primary_endpoint=(
                "Project-held-out descriptive and inferential transfer results with complete license, "
                "revision, build, label-contract, and static-safety provenance."
            ),
            falsifier=(
                "License or revision provenance is incomplete, builds cannot be reproduced, labels "
                "depend on unsafe execution, or source overlap contaminates evaluation."
            ),
            sampling_unit="licensed project × revision × source function × compiler target",
            dependencies=("P5", "P11", "P12"),
            required_evidence=(
                "pinned external-source manifest",
                "at least five projects and 100 labelable examples for promotion",
                "static-only compilation and disassembly receipts",
                "independent acquisition/build replay",
            ),
            integrity_gates=(
                "network retrieval is manifest-controlled",
                "external binaries are never executed",
                "unsupported cases remain visible",
                "application claims remain blocked below the gate",
            ),
            promotion_rule=(
                "Keep external results descriptive until every external-validity gate passes; "
                "then promote only the measured endpoint actually supported."
            ),
            stop_conditions=(
                "ambiguous license or source lineage",
                "build commands require uncontrolled network or execution",
                "fewer than the preregistered projects or examples are labelable",
            ),
            current_result=(
                "The repository currently has a local static compiler capability receipt and no "
                "licensed external-source effectiveness result."
            ),
        ),
        ScientificMilestone(
            milestone_id="P14",
            phase="active-sensing",
            title="Measured active-sensing robustness",
            status="active",
            horizon="next",
            scientific_question=(
                "Do active policies retain an information-per-measured-cost advantage under "
                "alternative budgets, calibration runs, and non-oracle controls?"
            ),
            conjecture=(
                "Finite policy certificates remain informative when deterministic instrumentation "
                "costs are supplemented by machine-specific calibrated acquisition measurements."
            ),
            primary_endpoint=(
                "Regret, information per deterministic and calibrated cost, and finite optimality "
                "gap across held-out programs and declared policy baselines."
            ),
            falsifier=(
                "The active policy does not beat fixed/random controls under measured costs or the "
                "certificate is invalidated by a reproducible finite counterexample."
            ),
            sampling_unit="source program × sensor budget × policy family",
            dependencies=("P6", "P12"),
            required_evidence=(
                "eight-component deterministic cost vectors",
                "warm-up and repeated wall-clock calibration",
                "exact dynamic-program reference",
                "policy stress and residual diagnostics",
            ),
            integrity_gates=(
                "actions cannot inspect unrevealed values",
                "oracle remains an inadmissible reference",
                "cross-machine comparisons require matching receipts",
                "finite certificates are not universal laws",
            ),
            promotion_rule=(
                "Promote only finite measured policy evidence unless a separate theorem and "
                "independent replication support a broader statement."
            ),
            stop_conditions=(
                "runtime language is used without a matching local calibration receipt",
                "policy selection inspects held-out values",
                "calibration residuals or action costs are not reproducible",
            ),
            current_result=(
                "The current finite policy and local calibration receipts are valid bounded "
                "diagnostics; broader active-sensing law remains unearned."
            ),
        ),
        ScientificMilestone(
            milestone_id="P15",
            phase="replication",
            title="Blind replication and adversarial audit",
            status="blocked_by_evidence",
            horizon="long",
            scientific_question=(
                "Do the primary quotient, transfer, and sensing endpoints survive a frozen public "
                "manifest, no-network replay, and independently controlled analysis?"
            ),
            conjecture=(
                "The source-bound receipts and primary endpoint definitions are sufficiently explicit "
                "for an independent implementation to reproduce or falsify them."
            ),
            primary_endpoint=(
                "Blind reproduction of preregistered primary endpoints, negative controls, and "
                "failure cases with no original-code access to private labels."
            ),
            falsifier=(
                "An independent analysis cannot reconstruct the data lineage, public hashes do not "
                "freeze the benchmark, or the endpoint changes after unblinding."
            ),
            sampling_unit="independent implementation × frozen benchmark × primary endpoint",
            dependencies=("P7", "P11", "P12", "P14"),
            required_evidence=(
                "public/private manifest separation",
                "offline replay receipt",
                "independent implementation or external replay record",
                "adversarial leakage and counterexample report",
            ),
            integrity_gates=(
                "gold labels remain outside public inputs",
                "negative replication is preserved",
                "original authors cannot edit the external result",
                "replay status is not inferred from local reruns",
            ),
            promotion_rule=(
                "Do not promote broad scientific or deployment claims without an external "
                "replication receipt."
            ),
            stop_conditions=(
                "public inputs contain recoverable labels",
                "source or protocol hash changes do not invalidate receipts",
                "the benchmark cannot be reconstructed from public materials",
            ),
            current_result=(
                "A frozen local no-network harness exists, and a same-host separately controlled "
                "Lima guest replay has been observed with protected scoring. Independent "
                "implementation or independent physical/CI replay remains unobserved, so the "
                "broad P15 gate remains blocked."
            ),
        ),
        ScientificMilestone(
            milestone_id="P16",
            phase="theory-and-publication",
            title="Theorem/countertheorem synthesis and promotion",
            status="planned",
            horizon="long",
            scientific_question=(
                "Which theorem, impossibility result, or bounded countertheorem is genuinely "
                "supported after the formal, empirical, transfer, sensing, and replication work?"
            ),
            conjecture=(
                "The integrated evidence will either support a reusable theorem about bounded "
                "observational quotients and sensing or produce a decisive counterexample that "
                "changes the scientific interpretation of the program."
            ),
            primary_endpoint=(
                "A checked theorem, explicit countertheorem, or unresolved result with complete "
                "assumptions, proof/counterexample artifact, independent review status, and claim map."
            ),
            falsifier=(
                "The proposed theorem is definitional, the assumptions hide the desired result, "
                "or empirical success is substituted for formal proof."
            ),
            sampling_unit="formal model class and independently reviewed evidence bundle",
            dependencies=("P11", "P12", "P13", "P14", "P15"),
            required_evidence=(
                "Lean or independently audited proof/counterexample",
                "complete assumption and axiom report",
                "external review receipt",
                "source-bound publication artifact",
            ),
            integrity_gates=(
                "theorem and empirical evidence remain separate",
                "counterexamples are first-class outputs",
                "publication language is generated from status receipts",
                "recognition language is not used as evidence",
            ),
            promotion_rule=(
                "Promote only the narrowest checked conclusion; retain unresolved or falsified "
                "status where the evidence does not support a stronger claim."
            ),
            stop_conditions=(
                "the result is only an engineering benchmark",
                "proof review is unavailable",
                "the evidence bundle is stale or source-unbound",
            ),
            current_result=(
                "The existing theorem phase is fail-closed with a local Lean fragment and bounded "
                "counterexample; this synthesis milestone is not yet started."
            ),
        ),
        ScientificMilestone(
            milestone_id="P17",
            phase="reproducibility-infrastructure",
            title="Protocol-v2 artifact and manuscript closure",
            status="implemented_local",
            horizon="current",
            scientific_question=(
                "Can every project-owned generated data surface be bound to one versioned "
                "protocol and then carried, without hand edits, through figures, manuscript "
                "variables, template hydration, and accessible document outputs?"
            ),
            conjecture=(
                "Separating domain schema versions from a common receipt schema makes stale, "
                "mixed-protocol, or bypassed artifacts detectable before they can become claims "
                "or rendered evidence."
            ),
            primary_endpoint=(
                "A protocol-v2 envelope on every project-owned JSON data artifact, atomic writes, "
                "fail-closed surface validation, source-generated variable/figure closure, and "
                "post-render provenance checks."
            ),
            falsifier=(
                "A generated JSON artifact bypasses the envelope, changes protocol without "
                "invalidating dependent receipts, leaves unresolved manuscript variables, or "
                "renders a figure without registered inputs, alternatives, and checks."
            ),
            sampling_unit="project-owned generated artifact and downstream manuscript consumer",
            dependencies=("P0", "P7", "P11"),
            required_evidence=(
                "protocol-v2 artifact receipt contract",
                "atomic producer writes and legacy migration diagnostics",
                "artifact-surface and mixed-protocol negative controls",
                "source-bound figure, variable, DOCX, and tagged-PDF closure",
            ),
            integrity_gates=(
                "domain schema and receipt schema remain distinct",
                "migration is explicit and never hidden in validation",
                "blocked and underpowered states remain visible",
                "manuscript values and captions are generated from receipts",
            ),
            promotion_rule=(
                "Treat this as a reproducibility and publication-integrity result; it cannot "
                "promote an empirical, formal, external-validity, or replication claim by itself."
            ),
            stop_conditions=(
                "a data artifact lacks required receipt fields",
                "a stale protocol hash is accepted by a dependent receipt",
                "a rendered artifact is not traceable to the final source and registry surface",
            ),
            current_result=(
                "Implemented locally: the shared receipt layer, atomic writers, explicit legacy "
                "migration command, artifact audit command, source-generated closure figure, and "
                "template-bound manuscript variables are now exercised by the producer pipeline. "
                "Independent review and public artifact evaluation remain outside this milestone."
            ),
        ),
    )


def scientific_program_payload() -> dict[str, object]:
    """Return the canonical research-program contract."""

    milestones = _milestones()
    status_counts: dict[str, int] = {}
    for milestone in milestones:
        status_counts[milestone.status] = status_counts.get(milestone.status, 0) + 1
    protocol = scientific_protocol_payload()
    envelope = build_receipt_envelope(
        protocol=protocol,
        generator_revision=generator_revision(Path(__file__)),
        input_hashes={"src/scientific_program.py": sha256_path(Path(__file__))},
        toolchain_versions=toolchain_versions(()),
        status="implemented_local",
        limitations=(
            "This artifact is a falsifiable research contract, not evidence that future milestones have been achieved.",
            "Promotion remains gated by source-bound receipts, negative controls, and independent evidence.",
        ),
        unsupported_cases=(
            "The roadmap does not substitute for a complete x86-64 theorem, universal identifiability, or external validity.",
        ),
        underpowered_cases=(
            "P13, P15, and P16 remain blocked or planned until their evidence gates pass; P17 is a local publication-integrity contract, not scientific promotion.",
        ),
    )
    return {
        **envelope,
        "schema_version": SCIENTIFIC_PROGRAM_SCHEMA_VERSION,
        "artifact": SCIENTIFIC_PROGRAM_ARTIFACT,
        "program_id": "active-assembly-quotient-inference",
        "program_title": "Operationally invariant inference for computational motifs",
        "central_question": (
            "Can computational motifs be defined as quotient objects of operational "
            "program semantics and recovered with calibrated uncertainty from active, "
            "architecture-invariant observations?"
        ),
        "central_conjecture": (
            "Under a separating family of safe interventions, observationally equivalent "
            "motifs are identifiable up to quotient-state relabeling, while an active policy "
            "approaches a non-trivial information-per-cost frontier."
        ),
        "recognition_reality": {
            "fields": (
                "Fields-level ambition would require a new, broadly important mathematical "
                "theorem; software scale, benchmark size, and visual polish are not substitutes."
            ),
            "nobel": (
                "There is no Nobel category for software or computer science. A Nobel-relevant "
                "trajectory would require a contribution that genuinely belongs in an eligible "
                "scientific field; this repository makes no such claim."
            ),
        },
        "status_counts": status_counts,
        "current_evidence_boundary": (
            "The repository demonstrates a source-bound x86-64 observation bridge, controlled "
            "temporal mechanisms, typed/action-conditioned diagnostics, leakage-safe ordered-HMM "
            "parameter learning on six transparent fixtures, and bounded P3-P10 implementation "
            "receipts with active P11-P17 promotion contracts. The extensions remain local or finite: they do not establish semantic "
            "identifiability, broad cross-architecture/compiler invariance, external validity, a "
            "general theorem, or independent replication."
        ),
        "program_invariants": [
            "Preregister primary endpoints before inspecting the final test set.",
            "Split by complete source program and block compiler/architecture near-duplicates.",
            "Treat source programs, not instruction steps, as the inferential sampling unit.",
            "Publish nulls, counterexamples, non-convergence, and failed replications.",
            "Keep implementation, empirical evidence, scholarship, and roadmap context partitioned.",
            "Never execute arbitrary binaries in the canonical demonstrator without a reviewed safety contract.",
            "Do not turn a roadmap milestone into a claim until its promotion gate passes.",
        ],
        "promotion_protocol": (
            "A milestone moves from planned or active to implemented only when its primary endpoint, "
            "falsifier checks, artifact lineage, negative controls, uncertainty definition, and claim "
            "boundary are all present. A broad conclusion additionally requires independent replication."
        ),
        "milestones": [milestone.to_dict() for milestone in milestones],
    }


def validate_scientific_program_payload(
    payload: dict[str, object],
) -> tuple[bool, str]:
    """Validate schema, content, and dependency integrity of a program receipt."""

    if payload.get("schema_version") != SCIENTIFIC_PROGRAM_SCHEMA_VERSION:
        return False, f"schema={payload.get('schema_version')}"
    protocol = scientific_protocol_payload()
    receipt_valid, receipt_detail = validate_receipt_envelope(
        payload, expected_protocol=protocol
    )
    if not receipt_valid:
        return False, f"scientific program receipt: {receipt_detail}"
    if payload.get("status") != "implemented_local":
        return False, "scientific program evidence status is invalid"
    milestones = payload.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        return False, "milestones must be a non-empty list"

    required = {
        "milestone_id",
        "phase",
        "title",
        "status",
        "horizon",
        "scientific_question",
        "conjecture",
        "primary_endpoint",
        "falsifier",
        "sampling_unit",
        "dependencies",
        "required_evidence",
        "integrity_gates",
        "promotion_rule",
        "stop_conditions",
        "current_result",
    }
    ids: list[str] = []
    errors: list[str] = []
    rows_by_id: dict[str, dict[str, object]] = {}
    for row in milestones:
        if not isinstance(row, dict):
            errors.append("non-object milestone")
            continue
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"{row.get('milestone_id', '?')}:missing={missing}")
            continue
        milestone_id = str(row["milestone_id"])
        ids.append(milestone_id)
        rows_by_id[milestone_id] = row
        if row.get("status") not in PROGRAM_STATUSES:
            errors.append(f"{milestone_id}:invalid-status={row.get('status')}")
        for field in (
            "scientific_question",
            "conjecture",
            "primary_endpoint",
            "falsifier",
            "sampling_unit",
            "promotion_rule",
            "current_result",
        ):
            if not str(row.get(field, "")).strip():
                errors.append(f"{milestone_id}:empty-{field}")
        for field in (
            "dependencies",
            "required_evidence",
            "integrity_gates",
            "stop_conditions",
        ):
            value = row.get(field)
            if (
                not isinstance(value, list)
                or (field != "dependencies" and not value)
                or any(not str(item).strip() for item in value)
            ):
                errors.append(f"{milestone_id}:invalid-{field}")
    if len(ids) != len(set(ids)):
        errors.append("duplicate milestone IDs")

    known_ids = set(ids)
    for milestone_id, row in rows_by_id.items():
        dependencies = row.get("dependencies", [])
        if not isinstance(dependencies, list):
            continue
        unknown = sorted(set(str(value) for value in dependencies) - known_ids)
        if unknown:
            errors.append(f"{milestone_id}:unknown-dependencies={unknown}")
        if milestone_id in dependencies:
            errors.append(f"{milestone_id}:self-dependency")

    # Detect cycles with a small deterministic depth-first traversal.
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(milestone_id: str) -> None:
        if milestone_id in visiting:
            errors.append(f"dependency-cycle-at={milestone_id}")
            return
        if milestone_id in visited:
            return
        visiting.add(milestone_id)
        row = rows_by_id.get(milestone_id, {})
        dependencies = row.get("dependencies", [])
        if isinstance(dependencies, list):
            for dependency in dependencies:
                if str(dependency) in rows_by_id:
                    visit(str(dependency))
        visiting.remove(milestone_id)
        visited.add(milestone_id)

    for milestone_id in ids:
        visit(milestone_id)

    status_counts = payload.get("status_counts")
    expected_counts: dict[str, int] = {}
    for row in milestones:
        if isinstance(row, dict):
            status = str(row.get("status"))
            expected_counts[status] = expected_counts.get(status, 0) + 1
    if status_counts != expected_counts:
        errors.append("status_counts do not match milestone rows")
    if not str(payload.get("central_question", "")).strip():
        errors.append("central_question is empty")
    if not str(payload.get("central_conjecture", "")).strip():
        errors.append("central_conjecture is empty")
    program_invariants = payload.get("program_invariants")
    if not isinstance(program_invariants, list) or len(program_invariants) < 5:
        errors.append("program_invariants are incomplete")
    if errors:
        return False, "; ".join(errors[:8])
    return True, f"milestones={len(milestones)} statuses={expected_counts}"


def write_scientific_program(project_root: Path = PROJECT_ROOT) -> str:
    """Write the canonical program receipt and return its project-relative path."""

    path = project_root / SCIENTIFIC_PROGRAM_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = scientific_program_payload()
    valid, detail = validate_scientific_program_payload(payload)
    if not valid:
        raise ValueError(f"canonical scientific program is invalid: {detail}")
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
    )
    return SCIENTIFIC_PROGRAM_ARTIFACT


__all__ = [
    "PROGRAM_STATUSES",
    "SCIENTIFIC_PROGRAM_ARTIFACT",
    "SCIENTIFIC_PROGRAM_SCHEMA_VERSION",
    "ScientificMilestone",
    "scientific_program_payload",
    "validate_scientific_program_payload",
    "write_scientific_program",
]
