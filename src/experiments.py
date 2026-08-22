"""End-to-end assembly observation and Active Inference experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .active_inference_bridge import InferenceResult, infer_hidden_states
from .assembly_formalism import assembly_listing_payload
from .assembly_packages import TraceBundle, build_trace_bundle, trace_bundle_to_json
from .assembly_packages.capstone_adapter import disassemble_with_capstone
from .assembly_packages.optional_adapters import probe_optional_packages
from .assembly_packages.unicorn_adapter import emulate_with_unicorn
from .disassembly import InstructionEvent, instructions_to_json
from .emulation import EmulationTrace
from .features import AssemblyObservation, encode_observations, observation_counts
from .typed_observations import TypedObservation, encode_typed_observations
from .figures import (
    plot_observation_pipeline,
    plot_package_provenance,
    plot_posterior_beliefs,
    plot_trace_refinement,
    plot_trace_feature_matrix,
)
from .generative_model import GenerativeModelSpec, build_generative_model
from .symbolic_semantics import (
    SymbolicExecutionReport,
    symbolic_execute,
    validate_symbolic_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class ExperimentResult:
    """Complete artifact summary for one sample experiment."""

    sample_name: str
    instructions: tuple[InstructionEvent, ...]
    trace: EmulationTrace
    observations: tuple[AssemblyObservation, ...]
    typed_observations: tuple[TypedObservation, ...]
    symbolic_execution: SymbolicExecutionReport
    model: GenerativeModelSpec
    inference: InferenceResult
    trace_bundle: TraceBundle
    data_paths: dict[str, str]
    figure_paths: dict[str, str]

    def summary(self) -> dict[str, object]:
        """Return a compact summary for reports and manuscript variables."""

        return {
            "sample_name": self.sample_name,
            "instruction_count": len(self.instructions),
            "observation_count": len(self.observations),
            "symbolic_execution": self.symbolic_execution.to_dict(),
            "symbolic_final_rax_matches": self.symbolic_execution.final_rax_matches,
            "final_rax": self.trace.final_rax,
            "final_state": self.inference.final_state,
            "observation_counts": observation_counts(self.observations),
            "trace_schema_version": self.trace_bundle.schema_version,
            "binary_sha256": self.trace_bundle.binary_sha256,
            "package_count": len(self.trace_bundle.package_provenance),
            "available_package_count": sum(
                provenance.available
                for provenance in self.trace_bundle.package_provenance
            ),
            "data_paths": self.data_paths,
            "figure_paths": self.figure_paths,
        }


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(path.relative_to(PROJECT_ROOT))


def _model_payload(model: GenerativeModelSpec) -> dict[str, object]:
    return {
        "A": model.A.tolist(),
        "D": model.D.tolist(),
        "observation_labels": list(model.observation_labels),
        "hidden_state_labels": list(model.hidden_state_labels),
    }


def run_experiment(sample_name: str = "x86_accumulator") -> ExperimentResult:
    """Run disassembly, emulation, feature encoding, inference, and exports."""

    decoded_instructions, capstone_provenance = disassemble_with_capstone(sample_name)
    emulation_trace, unicorn_provenance = emulate_with_unicorn(sample_name)
    instructions = tuple(decoded_instructions)
    trace = emulation_trace
    observations = tuple(encode_observations(instructions, trace))
    typed_observations = tuple(encode_typed_observations(instructions, trace))
    symbolic_execution = symbolic_execute(instructions, trace)
    symbolic_valid, symbolic_detail = validate_symbolic_report(symbolic_execution)
    if not symbolic_valid:
        raise RuntimeError(
            f"bounded symbolic semantics validation failed: {symbolic_detail}"
        )
    model = build_generative_model()
    inference = infer_hidden_states(list(observations), model)
    trace_bundle = build_trace_bundle(
        sample_name=sample_name,
        instructions=instructions,
        trace=trace,
        observations=observations,
        typed_observations=typed_observations,
        package_provenance=(
            capstone_provenance,
            unicorn_provenance,
            *tuple(probe_optional_packages()),
        ),
        symbolic_facts=tuple(event.to_dict() for event in symbolic_execution.events),
    )

    data_dir = OUTPUT_DIR / "data"
    figure_dir = OUTPUT_DIR / "figures"
    data_paths = {
        "disassembly_trace": _write_json(
            data_dir / "disassembly_trace.json",
            instructions_to_json(instructions),
        ),
        "emulation_trace": _write_json(
            data_dir / "emulation_trace.json",
            trace.to_dict(),
        ),
        "symbolic_execution": _write_json(
            data_dir / "symbolic_execution.json",
            symbolic_execution.to_dict(),
        ),
        "inference_result": _write_json(
            data_dir / "inference_result.json",
            {
                "model": _model_payload(model),
                "observations": [observation.to_dict() for observation in observations],
                "inference": inference.to_dict(),
            },
        ),
        "trace_bundle": _write_json(
            data_dir / "trace_bundle.json",
            trace_bundle_to_json(trace_bundle),
        ),
        "assembly_listing": _write_json(
            data_dir / "assembly_listing.json",
            assembly_listing_payload(sample_name, instructions, observations, trace),
        ),
    }
    figure_paths = {
        "observation_pipeline": str(
            plot_observation_pipeline(
                observations, figure_dir / "observation_pipeline.png"
            ).relative_to(PROJECT_ROOT)
        ),
        "posterior_beliefs": str(
            plot_posterior_beliefs(
                inference, model, figure_dir / "posterior_beliefs.png"
            ).relative_to(PROJECT_ROOT)
        ),
        "package_provenance": str(
            plot_package_provenance(
                [
                    provenance.to_dict()
                    for provenance in trace_bundle.package_provenance
                ],
                figure_dir / "package_provenance.png",
            ).relative_to(PROJECT_ROOT)
        ),
        "trace_feature_matrix": str(
            plot_trace_feature_matrix(
                list(observations),
                figure_dir / "trace_feature_matrix.png",
            ).relative_to(PROJECT_ROOT)
        ),
        "trace_refinement": str(
            plot_trace_refinement(
                symbolic_execution.to_dict(),
                figure_dir / "trace_refinement.png",
            ).relative_to(PROJECT_ROOT)
        ),
    }
    result = ExperimentResult(
        sample_name=sample_name,
        instructions=instructions,
        trace=trace,
        observations=observations,
        typed_observations=typed_observations,
        symbolic_execution=symbolic_execution,
        model=model,
        inference=inference,
        trace_bundle=trace_bundle,
        data_paths=data_paths,
        figure_paths=figure_paths,
    )
    _write_json(data_dir / "experiment_summary.json", result.summary())
    return result
