"""Assembly-code observations for Active Inference demonstrations."""

from .active_inference_bridge import InferenceResult, infer_hidden_states
from .active_selection import (
    ExpectedFreeEnergy,
    SelectionResult,
    active_observation_order,
    expected_free_energy,
    fixed_order_steps_to_confidence,
    select_next_observation,
)
from .advanced_figures import write_advanced_figures
from .advanced_inference import run_advanced_inference
from .assembly_packages import BasicBlock, CfgEdge, PackageProvenance, TraceBundle
from .claim_register import ClaimRecord, write_claim_register
from .corpus import CorpusFixture, build_corpus, fixture_observation_ids
from .disassembly import InstructionEvent, disassemble_sample
from .emulation import EmulationTrace, emulate_sample
from .evaluation import compare_models, evaluate_decoding, evaluation_payload
from .experiments import ExperimentResult, run_experiment
from .features import AssemblyObservation
from .generative_model import GenerativeModelSpec, build_generative_model
from .phase_segmentation import Segment, detect_changepoints, segment_phases
from .redteam_report import RedTeamFinding, write_redteam_report
from .reproducibility import ArtifactManifestRow, write_reproducibility_manifest
from .rxinfer_bridge import RxInferRunResult, run_rxinfer_backend
from .scholarship import ScholarshipSource
from .temporal_model import (
    FilterResult,
    SmoothingResult,
    TemporalGenerativeModel,
    build_temporal_model,
    forward_backward,
    forward_filter,
    viterbi_decode,
)
from .validation import ValidationReport, validate_trace_file, write_validation_report

__all__ = [
    "AssemblyObservation",
    "ArtifactManifestRow",
    "BasicBlock",
    "CfgEdge",
    "ClaimRecord",
    "CorpusFixture",
    "EmulationTrace",
    "ExpectedFreeEnergy",
    "ExperimentResult",
    "FilterResult",
    "GenerativeModelSpec",
    "InferenceResult",
    "InstructionEvent",
    "PackageProvenance",
    "RedTeamFinding",
    "RxInferRunResult",
    "ScholarshipSource",
    "Segment",
    "SelectionResult",
    "SmoothingResult",
    "TemporalGenerativeModel",
    "TraceBundle",
    "ValidationReport",
    "active_observation_order",
    "build_corpus",
    "build_generative_model",
    "build_temporal_model",
    "compare_models",
    "detect_changepoints",
    "disassemble_sample",
    "emulate_sample",
    "evaluate_decoding",
    "evaluation_payload",
    "expected_free_energy",
    "fixed_order_steps_to_confidence",
    "fixture_observation_ids",
    "forward_backward",
    "forward_filter",
    "infer_hidden_states",
    "run_advanced_inference",
    "run_experiment",
    "run_rxinfer_backend",
    "segment_phases",
    "select_next_observation",
    "validate_trace_file",
    "viterbi_decode",
    "write_claim_register",
    "write_advanced_figures",
    "write_redteam_report",
    "write_reproducibility_manifest",
    "write_validation_report",
]
