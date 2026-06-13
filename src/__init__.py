"""Assembly-code observations for Active Inference demonstrations."""

from .active_inference_bridge import InferenceResult, infer_hidden_states
from .assembly_packages import BasicBlock, CfgEdge, PackageProvenance, TraceBundle
from .claim_register import ClaimRecord, write_claim_register
from .disassembly import InstructionEvent, disassemble_sample
from .emulation import EmulationTrace, emulate_sample
from .experiments import ExperimentResult, run_experiment
from .features import AssemblyObservation
from .generative_model import GenerativeModelSpec, build_generative_model
from .redteam_report import RedTeamFinding, write_redteam_report
from .reproducibility import ArtifactManifestRow, write_reproducibility_manifest
from .rxinfer_bridge import RxInferRunResult, run_rxinfer_backend
from .scholarship import ScholarshipSource
from .validation import ValidationReport, validate_trace_file, write_validation_report

__all__ = [
    "AssemblyObservation",
    "ArtifactManifestRow",
    "BasicBlock",
    "CfgEdge",
    "ClaimRecord",
    "EmulationTrace",
    "ExperimentResult",
    "GenerativeModelSpec",
    "InferenceResult",
    "InstructionEvent",
    "PackageProvenance",
    "RedTeamFinding",
    "RxInferRunResult",
    "ScholarshipSource",
    "TraceBundle",
    "ValidationReport",
    "build_generative_model",
    "disassemble_sample",
    "emulate_sample",
    "infer_hidden_states",
    "run_experiment",
    "run_rxinfer_backend",
    "validate_trace_file",
    "write_claim_register",
    "write_redteam_report",
    "write_reproducibility_manifest",
    "write_validation_report",
]
