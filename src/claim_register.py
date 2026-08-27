"""Machine-readable claim register for manuscript and artifact validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .artifact_contract import write_json_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class ClaimRecord:
    """One bounded project claim linked to sources, artifacts, and checks."""

    claim_id: str
    claim_type: str
    statement: str
    support_keys: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    validation_checks: tuple[str, ...]
    boundary: str
    status: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe claim row."""

        data = asdict(self)
        data["support_keys"] = list(self.support_keys)
        data["artifact_paths"] = list(self.artifact_paths)
        data["validation_checks"] = list(self.validation_checks)
        return data


CLAIM_RECORDS: tuple[ClaimRecord, ...] = (
    ClaimRecord(
        claim_id="CLM-001",
        claim_type="implemented_trace",
        statement="The fixture bytes are decoded into auditable x86-64 instruction events.",
        support_keys=("capstone", "quynh2014capstone", "intel_sdm"),
        artifact_paths=(
            "output/data/disassembly_trace.json",
            "output/data/assembly_listing.json",
            "output/data/trace_bundle.json",
        ),
        validation_checks=(
            "assembly_listing_contract",
            "trace_required_package_provenance",
            "trace_instruction_observation_alignment",
        ),
        boundary="Fixture-specific disassembly is not semantic decompilation.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-002",
        claim_type="implemented_trace",
        statement="Bounded emulation produces a reproducible register-state trace for the fixture.",
        support_keys=("unicorn", "quynh2015unicorn"),
        artifact_paths=(
            "output/data/emulation_trace.json",
            "output/data/trace_bundle.json",
        ),
        validation_checks=("trace_dynamic_register_contract", "trace_binary_hash"),
        boundary="The bounded fixture trace is not a sandbox for arbitrary binaries.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-003",
        claim_type="observation_model",
        statement="Decoded assembly events are encoded as categorical observations for a generative model.",
        support_keys=(
            "friston2009freeenergy",
            "friston2017activeinference",
            "heins2022pymdp",
        ),
        artifact_paths=(
            "output/data/assembly_listing.json",
            "output/data/trace_bundle.json",
            "output/data/inference_result.json",
        ),
        validation_checks=(
            "assembly_listing_contract",
            "trace_observation_semantics",
            "scholarship_claim_boundaries",
        ),
        boundary="The observation vocabulary is hand-authored and does not prove program understanding.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-004",
        claim_type="rxinfer_backend",
        statement="RxInfer emits the authoritative Gaussian Bethe Free Energy trajectory and a validated, namespaced categorical phase-HMM companion.",
        support_keys=(
            "bagaev2023rxinfer",
            "rxinfer_static_docs",
            "rxinfer_python_docs",
        ),
        artifact_paths=(
            "output/rxinfer/free_energy_trace.jsonl",
            "output/rxinfer/diagnostics.json",
            "output/rxinfer/posteriors.json",
            "output/rxinfer_categorical/categorical_free_energy_trace.jsonl",
            "output/rxinfer_categorical/categorical_diagnostics.json",
            "output/rxinfer_categorical/categorical_posteriors.json",
        ),
        validation_checks=(
            "rxinfer_free_energy_records",
            "rxinfer_companion_artifacts",
            "rxinfer_categorical_free_energy_records",
            "rxinfer_categorical_companion_artifacts",
            "rxinfer_categorical_diagnostics_consistent",
        ),
        boundary="The Gaussian trajectory remains the authoritative project spine; the categorical phase-HMM is a fixed-parameter companion with model-specific free-energy scale and is not evidence of semantic understanding or a learned reverse-engineering system.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-005",
        claim_type="comparison_baseline",
        statement="The pymdp path is a comparison baseline rather than a source of RxInfer Bethe Free Energy.",
        support_keys=("heins2022pymdp",),
        artifact_paths=(
            "output/data/inference_result.json",
            "output/figures/inference_diagnostics.png",
        ),
        validation_checks=(
            "figure_inference_diagnostics",
            "scholarship_claim_boundaries",
        ),
        boundary="pymdp diagnostics and RxInfer Bethe Free Energy are not interchangeable.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-006",
        claim_type="roadmap_scope",
        statement="The broader assembly-analysis package surface is roadmap provenance unless adapters emit facts.",
        support_keys=(
            "angr",
            "rizin",
            "radare2",
            "llvm_objdump",
            "keystone",
            "nasm",
            "brumley2011bap",
            "schulte2019gtirb",
            "ghidra",
            "qiling",
            "miasm",
            "dyninst",
            "retdec",
            "mcsema",
        ),
        artifact_paths=(
            "output/data/trace_bundle.json",
            "output/data/scholarship_sources.json",
        ),
        validation_checks=(
            "trace_optional_roadmap_provenance",
            "scholarship_evidence_partition",
        ),
        boundary="Roadmap package citations are not evidence that those tools ran in the current fixture pipeline.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-008",
        claim_type="scholarship_boundary",
        statement="Citations are partitioned by evidence tier and project stage to prevent support laundering.",
        support_keys=tuple(),
        artifact_paths=(
            "output/data/scholarship_sources.json",
            "output/figures/scholarship_evidence_matrix.png",
        ),
        validation_checks=(
            "scholarship_source_quality",
            "scholarship_evidence_partition",
        ),
        boundary="A valid ledger documents support boundaries; it is not empirical proof by itself.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-009",
        claim_type="artifact_reproducibility",
        statement="Generated analysis artifacts are recorded with file sizes and SHA-256 hashes.",
        support_keys=tuple(),
        artifact_paths=(
            "output/data/reproducibility_manifest.json",
            "output/figures/artifact_hash_manifest.png",
        ),
        validation_checks=(
            "reproducibility_artifact_manifest",
            "figure_artifact_hash_manifest",
        ),
        boundary="A manifest detects stale or replaced artifacts; it does not certify scientific validity by itself.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-010",
        claim_type="temporal_inference",
        statement="A left-to-right transition matrix yields a temporal HMM whose forward-filter decoding substantially outperforms the identity-transition comparator on the labelled corpus.",
        support_keys=(
            "friston2017activeinference",
            "heins2022pymdp",
            "rabiner1989tutorial",
        ),
        artifact_paths=(
            "output/data/temporal_inference.json",
            "output/data/evaluation_report.json",
        ),
        validation_checks=(
            "advanced_temporal_artifacts_present",
            "advanced_temporal_beats_identity_transition",
            "advanced_temporal_isolates_transition",
            "advanced_memoryless_baseline_honest",
        ),
        boundary="The claim is bounded to beating the identity-transition comparator only: on this short-program corpus a simpler memoryless per-step decoder (argmax D·A) outperforms the temporal HMM, so temporal structure shows no net advantage at the authored likelihood and this is recorded explicitly. The identity-transition and temporal decoders share hand-authored A and D (only B differs); a separate supervised held-out fitting path is reported as a distinct generalization diagnostic, and the exit phase is observation-underdetermined.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-011",
        claim_type="evaluation_controls",
        statement="Phase-decoding accuracy is discriminative: shuffled-label and uniform-belief negative controls sit near chance, far below the temporal model.",
        support_keys=("heins2022pymdp",),
        artifact_paths=("output/data/evaluation_report.json",),
        validation_checks=(
            "advanced_negative_controls_discriminate",
            "advanced_evaluation_freshness",
        ),
        boundary="Controls show the metric is not vacuous; they do not establish external validity beyond the corpus.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-012",
        claim_type="phase_segmentation",
        statement="Smoothed beliefs segment each trace into contiguous program phases that exactly partition the instruction steps.",
        support_keys=("friston2017activeinference",),
        artifact_paths=("output/data/phase_segments.json",),
        validation_checks=("advanced_phase_segmentation_partition",),
        boundary="Segmentation reflects the hand-authored generative model, not a discovered ground-truth control-flow structure.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-013",
        claim_type="active_selection",
        statement="An expected-free-energy active-disassembly policy is implemented; on the short-program corpus it is never worse than a fixed scan and ties because one observation saturates the phase belief.",
        support_keys=("friston2017activeinference",),
        artifact_paths=("output/data/active_selection.json",),
        validation_checks=(
            "advanced_active_never_worse",
            "advanced_active_policy_functional",
        ),
        boundary="Expected free energy genuinely varies across candidates (functional policy, not collapsed), so the corpus tie is a real null — no measurable advantage at this fixture scale — not a non-working policy; the active advantage is shown only as a mechanism in unit tests.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-014",
        claim_type="ambiguity_ablation",
        statement="The temporal HMM gains a measurable decoding advantage over the memoryless per-step decoder exactly when observations become ambiguous (the authored likelihood is blended toward uniform), and has no net advantage at the authored, near-diagonal likelihood.",
        support_keys=("friston2017activeinference", "parr2022activeinference"),
        artifact_paths=(
            "output/data/evaluation_report.json",
            "output/figures/ambiguity_ablation.png",
        ),
        validation_checks=(
            "advanced_ambiguity_ablation_shows_advantage",
            "advanced_evaluation_freshness",
        ),
        boundary="The ablation changes only the likelihood conditioning (A) over the same byte corpus; it demonstrates the mechanism behind the head-to-head result, not a claim about real-world binary observation noise. Both directions are measured: no advantage at the authored likelihood, positive advantage at high ambiguity.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-015",
        claim_type="segmentation_evaluation",
        statement="The smoothed forward-backward beliefs segment traces into program phases with a measured, non-trivial boundary F1 and step accuracy against the hand-authored ground truth.",
        support_keys=("friston2017activeinference",),
        artifact_paths=(
            "output/data/phase_segments.json",
            "output/data/evaluation_report.json",
        ),
        validation_checks=(
            "advanced_segmentation_measured",
            "advanced_phase_segmentation_partition",
        ),
        boundary="Segmentation reflects the hand-authored generative model, not a discovered ground-truth control-flow structure; the boundary-recovery metric quantifies how well the model's smoothed beliefs align with the authored phase boundaries.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-016",
        claim_type="decoder_calibration",
        statement="Decoder calibration tracks accuracy on this corpus: the memoryless per-step decoder is the best calibrated (lowest ECE), the temporal model is middle, and the identity-transition comparator is worst.",
        support_keys=("friston2017activeinference", "parr2022activeinference"),
        artifact_paths=(
            "output/data/evaluation_report.json",
            "output/figures/decoder_calibration.png",
        ),
        validation_checks=(
            "advanced_decoder_calibration_ordering",
            "advanced_evaluation_freshness",
        ),
        boundary="Calibration is computed from each decoder's own per-step posteriors on the small hand-authored corpus; it reports honest ordering (the temporal model is neither the most accurate nor the best calibrated), not a claim about real-world binary observation noise.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-017",
        claim_type="model_evidence",
        statement="The temporal HMM has the highest Bayesian log marginal likelihood of the observed traces (best generative model) even though the memoryless decoder wins on raw per-step accuracy.",
        support_keys=("friston2017activeinference", "rabiner1989tutorial"),
        artifact_paths=(
            "output/data/evaluation_report.json",
            "output/figures/model_evidence.png",
        ),
        validation_checks=(
            "advanced_model_evidence_temporal_best",
            "advanced_evaluation_freshness",
        ),
        boundary="Log marginal likelihood is computed from the forward-filter evidence on the small hand-authored corpus; it reports the generative (model-selection) comparison — the temporal model best explains the data even where it is not the best per-step classifier.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-018",
        claim_type="held_out_estimation",
        statement="A smoothed supervised HMM fitting path estimates A, B, and D on training fixtures only and evaluates each excluded fixture in a deterministic leave-one-fixture-out design.",
        support_keys=("rabiner1989tutorial",),
        artifact_paths=("output/data/held_out_model.json",),
        validation_checks=("held_out_model_contract", "held_out_model_freshness"),
        boundary="This is supervised estimation on the transparent toy-fixture corpus, not unsupervised phase discovery, population inference, or evidence for general binary-analysis performance.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-019",
        claim_type="bounded_semantics",
        statement="The implemented transfer-function subset produces an auditable symbolic trace whose pre/post RAX values agree stepwise with independent Unicorn observations, and whose explicitly modeled EFLAGS bits agree on the supported fixtures.",
        support_keys=(
            "capstone",
            "intel_sdm",
            "unicorn",
            "dasgupta2019x86semantics",
        ),
        artifact_paths=(
            "output/data/symbolic_execution.json",
            "output/data/trace_bundle.json",
            "output/figures/trace_refinement.png",
        ),
        validation_checks=(
            "symbolic_semantics_contract",
            "trace_symbolic_facts_contract",
            "symbolic_stepwise_trace_refinement",
            "figure_trace_refinement",
        ),
        boundary="The semantics are explicitly bounded to the documented immediate-register subset and modeled flag set; unsupported instructions fail closed and this is not a complete x86 lifter, SMT proof, alias analysis, or decompiler.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-020",
        claim_type="generative_adequacy",
        statement="A fixed-parameter HMM generative adequacy check compares observed fixture phase structure with same-length simulations and reports which observed metrics fall inside central simulation intervals.",
        support_keys=("rabiner1989tutorial", "gelman2013bda"),
        artifact_paths=(
            "output/data/generative_adequacy.json",
            "output/data/evaluation_report.json",
            "output/figures/generative_adequacy.png",
        ),
        validation_checks=(
            "generative_adequacy_contract",
            "generative_adequacy_freshness",
            "figure_generative_adequacy",
        ),
        boundary="The check uses the fixed hand-authored A/B/D model and transparent fixture lengths; simulation intervals are descriptive model-checking intervals, not posterior uncertainty, population inference, or evidence that the model has discovered phases.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-021",
        claim_type="typed_observation_model",
        statement="A label-blind typed observation schema retains static Capstone fields and bounded dynamic status, and its factorized and hierarchical companion scores are evaluated with fixture-level holdouts, field-family ablations, and an explicit missing-field policy.",
        support_keys=(
            "intel_sdm",
            "capstone",
            "unicorn",
            "rabiner1989tutorial",
            "gelman2013bda",
        ),
        artifact_paths=(
            "output/data/typed_observations.json",
            "output/data/factorized_observation_model.json",
            "output/figures/typed_observation_matrix.png",
            "output/figures/factorized_observation_comparison.png",
            "output/data/trace_bundle.json",
        ),
        validation_checks=(
            "trace_typed_observation_alignment",
            "typed_observation_schema_contract",
            "typed_observation_schema_freshness",
            "factorized_observation_contract",
            "factorized_observation_freshness",
            "factorized_observation_leakage_guard",
            "figure_typed_observation_matrix",
            "figure_factorized_observation_comparison",
        ),
        boundary="The typed and factorized layers are a supervised, fixture-level companion to the collapsed categorical HMM; they do not establish independent semantic discovery, conditional independence of fields, external validity, or improved performance on arbitrary binaries.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-022",
        claim_type="corpus_design_coverage",
        statement="The expanded transparent fixture corpus contains source-bound variation in bounded memory operands, repeated authored phases, and static-versus-dynamic branch paths, and the coverage inventory records that variation before any richer model is promoted.",
        support_keys=("capstone", "unicorn", "rabiner1989tutorial"),
        artifact_paths=(
            "output/data/corpus_coverage.json",
            "output/figures/corpus_coverage.png",
        ),
        validation_checks=(
            "corpus_coverage_contract",
            "corpus_coverage_freshness",
            "figure_corpus_coverage",
        ),
        boundary="Coverage is a design diagnostic over hand-authored toy fixtures; it is not independent binary sampling, external validity, or evidence that any typed field is intrinsic program semantics.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-023",
        claim_type="action_conditioned_sensing",
        statement="A bounded action-conditioned policy can choose typed field queries from training-only predictive likelihoods before revealing held-out values, and its acquisition budget can be compared with fixed, seeded-random, and explicitly inadmissible oracle policies.",
        support_keys=(
            "friston2017activeinference",
            "parr2022activeinference",
            "rabiner1989tutorial",
        ),
        artifact_paths=(
            "output/data/action_sensing.json",
            "output/figures/action_sensing.png",
        ),
        validation_checks=(
            "action_sensing_contract",
            "action_sensing_freshness",
            "action_sensing_leakage_guard",
            "figure_action_sensing",
        ),
        boundary="The experiment uses equal normalized proxy costs and transparent toy fixtures; it is a sensor-choice mechanism test, not evidence of autonomous sensing, real execution cost, or general reverse-engineering performance.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-024",
        claim_type="observation_only_parameter_learning",
        statement="An ordered categorical HMM can be fit by observation-only Baum-Welch expectation-maximization on training-fixture observation IDs, with phase labels reserved for held-out scoring and a complete-fixture comparison against supervised MAP and the fixed reference model.",
        support_keys=("dempster1977em", "rabiner1989tutorial"),
        artifact_paths=(
            "output/data/parameter_learning.json",
            "output/figures/parameter_learning.png",
        ),
        validation_checks=(
            "parameter_learning_contract",
            "parameter_learning_freshness",
            "parameter_learning_leakage_guard",
            "figure_parameter_learning",
        ),
        boundary="The estimator uses an ordered-state scaffold, a reference-model initialization, a positive transition-support prior, and only six transparent toy fixtures; its held-out accuracy is a finite implementation diagnostic, not evidence of unsupervised semantic discovery, identifiable phase meanings, or general binary-analysis performance.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-025",
        claim_type="scientific_program_contract",
        statement=(
            "The long-horizon research program is recorded as a machine-readable contract "
            "with explicit conjectures, primary endpoints, falsifiers, dependencies, "
            "sampling units, promotion gates, and stop conditions."
        ),
        support_keys=tuple(),
        artifact_paths=("output/data/scientific_program.json",),
        validation_checks=(
            "scientific_program_contract",
            "scientific_program_freshness",
        ),
        boundary=(
            "A roadmap receipt is a design and integrity contract, not evidence that "
            "the planned identifiability, theorem, external-validity, or replication "
            "milestones have been achieved."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-026",
        claim_type="quotient_protocol",
        statement=(
            "A bounded operational-equivalence protocol emits real Capstone/Unicorn "
            "witnesses for transparent semantics-preserving transformation pairs and "
            "a deliberately separated near-miss pair."
        ),
        support_keys=("capstone", "unicorn", "intel_sdm"),
        artifact_paths=("output/data/quotient_invariance_protocol.json",),
        validation_checks=(
            "quotient_invariance_contract",
            "quotient_invariance_freshness",
        ),
        boundary=(
            "This is the first bounded P3 protocol and witness set; it is not latent "
            "semantic identifiability, posterior invariance, complete-ISA equivalence, "
            "cross-architecture transfer, or an external-validity result."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-027",
        claim_type="representation_invariance_benchmark",
        statement=(
            "A source-family-disjoint transparent x86-64 benchmark records "
            "equivalent and near-miss transformation variants, declares its "
            "primary P3 endpoints before scoring, and computes bounded protocol "
            "diagnostics from real Capstone/Unicorn witnesses."
        ),
        support_keys=("capstone", "unicorn", "intel_sdm"),
        artifact_paths=("output/data/representation_invariance_benchmark.json",),
        validation_checks=(
            "representation_invariance_contract",
            "representation_invariance_freshness",
        ),
        boundary=(
            "The benchmark is a source-family-disjoint protocol and descriptive "
            "witness receipt, not a fitted semantic model, an identifiability "
            "result, posterior invariance, cross-architecture transfer, or "
            "external-validity claim. Its equivalent/near-miss labels come from "
            "the declared bounded output relation."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-028",
        claim_type="intervention_separation_control",
        statement=(
            "The P3 benchmark retains a deliberately insufficient single-state "
            "intervention catalog as a negative control: it leaves some declared "
            "near misses unresolved while preserving equivalent-pair consistency."
        ),
        support_keys=("capstone", "unicorn", "intel_sdm"),
        artifact_paths=("output/data/representation_invariance_benchmark.json",),
        validation_checks=(
            "representation_invariance_contract",
            "representation_invariance_freshness",
        ),
        boundary=(
            "This is a protocol sensitivity control showing that separating power "
            "depends on the intervention catalog. It is not a fitted-model failure "
            "analysis, semantic identifiability result, or evidence that the primary "
            "catalog is complete."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-029",
        claim_type="cross_architecture_runtime_benchmark",
        statement=(
            "The same transparent numeric accumulator operation is decoded by "
            "Capstone and boundedly executed by Unicorn on x86-64, AArch64, and "
            "RISC-V64 fixtures, with the expected accumulator result on all three."
        ),
        support_keys=("capstone", "unicorn"),
        artifact_paths=("output/data/cross_architecture_fixtures.json",),
        validation_checks=(
            "cross_architecture_contract",
            "cross_architecture_freshness",
        ),
        boundary=(
            "This is a local adapter and fixture benchmark for one numeric "
            "operation. It is not cross-ISA semantic equivalence, compiler "
            "invariance, or external validity."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-030",
        claim_type="formal_verification_receipt",
        statement=(
            "A pinned Lean 4.28.0 project kernel-checks modular accumulator "
            "equivalence and a near-miss counterexample for the declared closed "
            "operational fragment."
        ),
        support_keys=("lean_reference",),
        artifact_paths=("output/data/formal_verification.json",),
        validation_checks=(
            "formal_verification_contract",
            "formal_verification_freshness",
        ),
        boundary=(
            "The proof covers the declared Word64 fragment only; it is not a "
            "complete x86 semantics, lifter correctness theorem, or learned "
            "semantic identifiability result."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-031",
        claim_type="exact_information_policy",
        statement=(
            "A finite label-blind typed-field sensor family is analyzed by exact "
            "finite-horizon dynamic programming, producing an executable "
            "information-per-cost frontier without Monte Carlo policy evaluation."
        ),
        support_keys=("friston2017activeinference", "parr2022activeinference"),
        artifact_paths=(
            "output/data/active_information_frontier.json",
            "output/figures/active_information_frontier.png",
        ),
        validation_checks=(
            "active_information_contract",
            "active_information_freshness",
            "figure_active_information_frontier",
        ),
        boundary=(
            "The sensor channels and costs are finite, model-derived, and "
            "fixture-bound. The frontier is not a deployment cost law, an "
            "autonomous binary-analysis policy, or a theorem for all sensors."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-032",
        claim_type="blind_replication_harness",
        statement=(
            "A frozen local CLI harness separates public feature rows from gold "
            "labels and binds generation, scoring, and replay receipts to the "
            "source-bound protocol and input hashes."
        ),
        support_keys=("wilkinson2016fair",),
        artifact_paths=(
            "output/data/replication_manifest.json",
            "output/replication/public/public_manifest.json",
            "output/replication/private/gold_labels.json",
        ),
        validation_checks=("replication_contract", "replication_freshness"),
        boundary=(
            "This is a deterministic no-network local reproduction interface. It "
            "does not constitute independent replay, a public challenge, or an "
            "external-validity result until a separately controlled implementation "
            "reproduces the receipt."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-033",
        claim_type="theorem_or_countertheorem_phase",
        statement=(
            "The theorem phase distinguishes a kernel-checked closed Lean result, "
            "a finite computational certificate, and an automatically generated "
            "bounded counterexample to an overstrong single-intervention claim."
        ),
        support_keys=("demoura2021lean4", "lean_reference"),
        artifact_paths=(
            "output/data/theorem_or_countertheorem.json",
            "output/data/formal_verification.json",
        ),
        validation_checks=(
            "theorem_countertheorem_contract",
            "theorem_countertheorem_freshness",
        ),
        boundary=(
            "The receipt does not prove a broad x86-64 correspondence, transfer "
            "guarantee, or identifiability theorem. The promotion state remains "
            "fail closed, and unresolved theorem classes are first-class outputs."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-034",
        claim_type="finite_intervention_power_diagnostic",
        statement=(
            "The finite quotient benchmark enumerates every non-empty subset of its "
            "four declared initial-state interventions and reports the minimum "
            "separating sets for the current equivalent and near-miss outcome matrix."
        ),
        support_keys=("capstone", "unicorn", "intel_sdm"),
        artifact_paths=(
            "output/data/intervention_power.json",
            "output/figures/intervention_power.png",
        ),
        validation_checks=(
            "intervention_power_contract",
            "intervention_power_freshness",
            "figure_intervention_power",
        ),
        boundary=(
            "This is an exact finite combinatorial diagnostic over a source-bound "
            "outcome matrix. Singleton separators and a negative control do not "
            "establish semantic identifiability, completeness of the intervention "
            "catalog, or a universal separation law."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-035",
        claim_type="observation_only_initialization_robustness",
        statement=(
            "Observation-only ordered-HMM fitting is evaluated across five frozen "
            "initializations for each of six complete-fixture holdouts, with every "
            "held-out score and pairwise label-agreement diagnostic retained."
        ),
        support_keys=("dempster1977em", "rabiner1989tutorial"),
        artifact_paths=(
            "output/data/parameter_learning.json",
            "output/figures/parameter_learning.png",
        ),
        validation_checks=(
            "parameter_learning_contract",
            "parameter_learning_freshness",
            "parameter_learning_leakage_guard",
            "figure_parameter_learning",
        ),
        boundary=(
            "The multi-start result is a finite seed-sensitivity and robustness "
            "analysis. It does not establish stable latent parameters, unsupervised "
            "semantic discovery, or population-level performance; the observed spread "
            "is retained as a limitation."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-036",
        claim_type="licensed_external_validation_boundary",
        statement=(
            "The external-validation track uses an immutable manifest with license, "
            "revision, archive-hash, compiler, and static-only safety fields; the "
            "pinned source rows and their underpowered label/build gates remain "
            "visible rather than becoming a zero-valued application result."
        ),
        support_keys=("wilkinson2016fair",),
        artifact_paths=(
            "configs/external_sources.yaml",
            "output/data/external_source_validation.json",
        ),
        validation_checks=(
            "external_source_contract",
            "external_source_freshness",
            "figure_external_source_provenance",
        ),
        boundary=(
            "This receipt records a controlled acquisition and static-analysis "
            "interface. It is not external validity or application effectiveness "
            "until the project, revision, label-count, build, and independent-replay "
            "gates pass."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-038",
        claim_type="licensed_static_source_coverage",
        statement=(
            "Pinned permissively licensed source rows are acquired only through "
            "explicit network opt-in, checked against archive and license digests, "
            "compiled with an allowlisted static argument vector, and decoded with "
            "object/disassembly hashes while emitted objects remain unexecuted."
        ),
        support_keys=("wilkinson2016fair",),
        artifact_paths=(
            "configs/external_sources.yaml",
            "output/data/external_capabilities.json",
            "output/data/external_static_validation.json",
        ),
        validation_checks=(
            "external_static_contract",
            "external_static_freshness",
            "external_capability_contract",
            "external_capability_freshness",
        ),
        boundary=(
            "Static compilation and decoding establish coverage and provenance only. "
            "They do not establish source equivalence, project-held-out accuracy, "
            "compiler correctness, or application effectiveness."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-039",
        claim_type="separately_controlled_replay_boundary",
        statement=(
            "The frozen public replay bundle carries protocol, toolchain, instruction, "
            "and prediction hashes, while the protected ingestion path requires an "
            "explicit separately controlled-run attestation before recording observed "
            "independent replay."
        ),
        support_keys=("wilkinson2016fair", "acm2020artifact"),
        artifact_paths=(
            "output/data/replication_manifest.json",
            "output/replication/public/protocol_manifest.json",
            "output/replication/public/toolchain_receipt.json",
            "output/replication/public/REPLAY_INSTRUCTIONS.md",
        ),
        validation_checks=(
            "replication_contract",
            "replication_freshness",
            "independent_replay_contract",
            "independent_replay_freshness",
        ),
        boundary=(
            "The repository can prepare and ingest a receipt but cannot fabricate "
            "the separate run. Independent replay, when observed, is not independent "
            "reimplementation and does not establish new external validity."
        ),
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-037",
        claim_type="protocol_artifact_closure",
        statement=(
            "Project-owned generated JSON artifacts carry a shared protocol-v2 receipt "
            "layer distinct from their domain schema, and the producer pipeline binds "
            "that surface to manuscript and document-generation checks."
        ),
        support_keys=(
            "w3cprov2013",
            "rocrate2023",
            "sandve2013repro",
            "acm2020artifact",
        ),
        artifact_paths=(
            "output/data/research_protocol.json",
            "output/data/validation_report.json",
            "output/figures/artifact_receipt_closure.png",
        ),
        validation_checks=(
            "artifact_receipt_surface",
            "scientific_protocol_contract",
            "figure_artifact_receipt_closure",
        ),
        boundary=(
            "The receipt contract detects stale, mixed, or bypassed generated artifacts; "
            "it does not establish semantic identifiability, external validity, independent "
            "replication, or a formal theorem."
        ),
        status="validated",
    ),
)


def claim_register_payload() -> dict[str, object]:
    """Return the complete claim register payload."""

    status_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for claim in CLAIM_RECORDS:
        status_counts[claim.status] = status_counts.get(claim.status, 0) + 1
        type_counts[claim.claim_type] = type_counts.get(claim.claim_type, 0) + 1
    return {
        "claim_count": len(CLAIM_RECORDS),
        "status_counts": status_counts,
        "type_counts": type_counts,
        "claims": [claim.to_dict() for claim in CLAIM_RECORDS],
    }


def write_claim_register(output_dir: Path = OUTPUT_DIR) -> str:
    """Write claim register artifacts and return the JSON path relative to the project."""

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "claim_register.json"
    payload = claim_register_payload()
    write_json_artifact(
        json_path,
        payload,
        project_root=output_dir.parent,
        producer_path=Path(__file__),
    )

    markdown_path = data_dir / "claim_register.md"
    rows = [
        "# Claim Register",
        "",
        "| ID | Type | Status | Claim | Boundary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for claim in CLAIM_RECORDS:
        rows.append(
            f"| `{claim.claim_id}` | {claim.claim_type} | {claim.status} | "
            f"{claim.statement} | {claim.boundary} |"
        )
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    from .figures import plot_claim_evidence_map

    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    plot_claim_evidence_map(
        [claim.to_dict() for claim in CLAIM_RECORDS],
        output_dir / "figures" / "claim_evidence_map.png",
    )
    return str(json_path.relative_to(PROJECT_ROOT))
