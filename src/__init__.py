"""Assembly-code observations for Active Inference demonstrations.

The package exposes a small convenience API while keeping submodule imports
lightweight. In particular, importing a focused utility such as
``src.pdf_accessibility`` must not eagerly initialize the optional
Active-Inference/JAX stack. The public names below are resolved on first access
and then cached in the module namespace, preserving the previous
``from src import ...`` API without imposing its heaviest dependency cost on
every consumer.
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AssemblyObservation": ("features", "AssemblyObservation"),
    "ArchitectureFixture": ("multi_arch", "ArchitectureFixture"),
    "SensingAction": ("action_sensing", "SensingAction"),
    "FactorizedObservationModel": (
        "factorized_model",
        "FactorizedObservationModel",
    ),
    "FORMAL_TOOLCHAIN": ("formal_verification", "FORMAL_TOOLCHAIN"),
    "assembly_listing_markdown": ("assembly_formalism", "assembly_listing_markdown"),
    "assembly_listing_payload": ("assembly_formalism", "assembly_listing_payload"),
    "ArtifactManifestRow": ("reproducibility", "ArtifactManifestRow"),
    "BasicBlock": ("assembly_packages", "BasicBlock"),
    "CfgEdge": ("assembly_packages", "CfgEdge"),
    "ClaimRecord": ("claim_register", "ClaimRecord"),
    "CorpusFixture": ("corpus", "CorpusFixture"),
    "DOCX_ARTIFACT": ("docx_validation", "DOCX_ARTIFACT"),
    "DOCX_RECEIPT_ARTIFACT": ("docx_validation", "DOCX_RECEIPT_ARTIFACT"),
    "corpus_coverage_payload": ("corpus_coverage", "corpus_coverage_payload"),
    "EmulationTrace": ("emulation", "EmulationTrace"),
    "ExpectedFreeEnergy": ("active_selection", "ExpectedFreeEnergy"),
    "ExperimentResult": ("experiments", "ExperimentResult"),
    "FilterResult": ("temporal_model", "FilterResult"),
    "GenerativeModelSpec": ("generative_model", "GenerativeModelSpec"),
    "ExactPolicyResult": ("information_policy", "ExactPolicyResult"),
    "FiniteSensor": ("information_policy", "FiniteSensor"),
    "InferenceResult": ("active_inference_bridge", "InferenceResult"),
    "InstructionEvent": ("disassembly", "InstructionEvent"),
    "PackageProvenance": ("assembly_packages", "PackageProvenance"),
    "RxInferRunResult": ("rxinfer_bridge", "RxInferRunResult"),
    "ScholarshipSource": ("scholarship", "ScholarshipSource"),
    "Segment": ("phase_segmentation", "Segment"),
    "SelectionResult": ("active_selection", "SelectionResult"),
    "ScientificMilestone": ("scientific_program", "ScientificMilestone"),
    "SmoothingResult": ("temporal_model", "SmoothingResult"),
    "SymbolicExecutionReport": ("symbolic_semantics", "SymbolicExecutionReport"),
    "TemporalGenerativeModel": ("temporal_model", "TemporalGenerativeModel"),
    "TraceBundle": ("assembly_packages", "TraceBundle"),
    "TransformationFamily": (
        "representation_invariance",
        "TransformationFamily",
    ),
    "TypedObservation": ("typed_observations", "TypedObservation"),
    "ValidationReport": ("validation", "ValidationReport"),
    "FigureSpec": ("figure_contract", "FigureSpec"),
    "active_observation_order": ("active_selection", "active_observation_order"),
    "action_conditioned_sensing_payload": (
        "action_sensing",
        "action_conditioned_sensing_payload",
    ),
    "build_assembly_listing": ("assembly_formalism", "build_assembly_listing"),
    "build_corpus": ("corpus", "build_corpus"),
    "build_figure_specs": ("figure_contract", "build_figure_specs"),
    "build_generative_model": ("generative_model", "build_generative_model"),
    "cross_architecture_payload": ("multi_arch", "cross_architecture_payload"),
    "build_static_cfg": ("assembly_packages", "build_static_cfg"),
    "build_temporal_model": ("temporal_model", "build_temporal_model"),
    "build_typed_observation_corpus": (
        "factorized_model",
        "build_typed_observation_corpus",
    ),
    "compare_models": ("evaluation", "compare_models"),
    "detect_changepoints": ("phase_segmentation", "detect_changepoints"),
    "disassemble_sample": ("disassembly", "disassemble_sample"),
    "docx_render_payload": ("docx_validation", "docx_render_payload"),
    "emulate_sample": ("emulation", "emulate_sample"),
    "encode_typed_observations": (
        "typed_observations",
        "encode_typed_observations",
    ),
    "evaluate_decoding": ("evaluation", "evaluate_decoding"),
    "evaluation_payload": ("evaluation", "evaluation_payload"),
    "active_information_payload": (
        "information_policy",
        "active_information_payload",
    ),
    "expected_free_energy": ("active_selection", "expected_free_energy"),
    "expected_information_gain": ("active_selection", "expected_information_gain"),
    "solve_exact_policy": ("information_policy", "solve_exact_policy"),
    "fixed_order_steps_to_confidence": (
        "active_selection",
        "fixed_order_steps_to_confidence",
    ),
    "fixture_observation_ids": ("corpus", "fixture_observation_ids"),
    "fit_factorized_observation_model": (
        "factorized_model",
        "fit_factorized_observation_model",
    ),
    "fit_temporal_model": ("model_fitting", "fit_temporal_model"),
    "fit_ordered_hmm_em": ("parameter_learning", "fit_ordered_hmm_em"),
    "formal_verification_payload": (
        "formal_verification",
        "formal_verification_payload",
    ),
    "forward_backward": ("temporal_model", "forward_backward"),
    "forward_filter": ("temporal_model", "forward_filter"),
    "generative_adequacy": ("evaluation", "generative_adequacy"),
    "infer_hidden_states": ("active_inference_bridge", "infer_hidden_states"),
    "leave_one_fixture_out_evaluation": (
        "model_fitting",
        "leave_one_fixture_out_evaluation",
    ),
    "leave_one_fixture_out_factorized_evaluation": (
        "factorized_model",
        "leave_one_fixture_out_factorized_evaluation",
    ),
    "predictive_information_contribution": (
        "active_selection",
        "predictive_information_contribution",
    ),
    "parameter_learning_payload": (
        "parameter_learning",
        "parameter_learning_payload",
    ),
    "quotient_invariance_payload": (
        "quotient_invariance",
        "quotient_invariance_payload",
    ),
    "representation_invariance_payload": (
        "representation_invariance",
        "representation_invariance_payload",
    ),
    "run_advanced_inference": ("advanced_inference", "run_advanced_inference"),
    "run_experiment": ("experiments", "run_experiment"),
    "write_formal_verification": (
        "formal_verification",
        "write_formal_verification",
    ),
    "run_rxinfer_backend": ("rxinfer_bridge", "run_rxinfer_backend"),
    "run_rxinfer_categorical_backend": (
        "rxinfer_bridge",
        "run_rxinfer_categorical_backend",
    ),
    "write_quotient_invariance_protocol": (
        "quotient_invariance",
        "write_quotient_invariance_protocol",
    ),
    "scientific_program_payload": (
        "scientific_program",
        "scientific_program_payload",
    ),
    "segment_phases": ("phase_segmentation", "segment_phases"),
    "select_next_observation": ("active_selection", "select_next_observation"),
    "symbolic_execute": ("symbolic_semantics", "symbolic_execute"),
    "validate_symbolic_report": (
        "symbolic_semantics",
        "validate_symbolic_report",
    ),
    "validate_scientific_program_payload": (
        "scientific_program",
        "validate_scientific_program_payload",
    ),
    "validate_quotient_invariance_payload": (
        "quotient_invariance",
        "validate_quotient_invariance_payload",
    ),
    "validate_representation_invariance_payload": (
        "representation_invariance",
        "validate_representation_invariance_payload",
    ),
    "validate_cross_architecture_payload": (
        "multi_arch",
        "validate_cross_architecture_payload",
    ),
    "validate_julia_manifest": (
        "rxinfer_bridge",
        "validate_julia_manifest",
    ),
    "validate_trace_file": ("validation", "validate_trace_file"),
    "validate_docx_render_payload": (
        "docx_validation",
        "validate_docx_render_payload",
    ),
    "validate_docx_render_receipt": (
        "docx_validation",
        "validate_docx_render_receipt",
    ),
    "validate_figure_bundle": ("figure_contract", "validate_figure_bundle"),
    "validate_active_information_payload": (
        "information_policy",
        "validate_active_information_payload",
    ),
    "viterbi_decode": ("temporal_model", "viterbi_decode"),
    "write_advanced_figures": ("advanced_figures", "write_advanced_figures"),
    "write_figure_bundle": ("figure_contract", "write_figure_bundle"),
    "write_claim_register": ("claim_register", "write_claim_register"),
    "write_docx_render_receipt": (
        "docx_validation",
        "write_docx_render_receipt",
    ),
    "write_reproducibility_manifest": (
        "reproducibility",
        "write_reproducibility_manifest",
    ),
    "write_scientific_program": ("scientific_program", "write_scientific_program"),
    "write_representation_invariance_benchmark": (
        "representation_invariance",
        "write_representation_invariance_benchmark",
    ),
    "write_validation_report": ("validation", "write_validation_report"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve and cache one public convenience export on first access."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to IDEs and interactive discovery."""

    return sorted(set(globals()) | set(__all__))
