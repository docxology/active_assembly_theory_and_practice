"""Machine-readable claim register for manuscript and artifact validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

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
        support_keys=("capstone",),
        artifact_paths=("output/data/disassembly_trace.json", "output/data/trace_bundle.json"),
        validation_checks=(
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
        support_keys=("unicorn",),
        artifact_paths=("output/data/emulation_trace.json", "output/data/trace_bundle.json"),
        validation_checks=("trace_dynamic_register_contract", "trace_binary_hash"),
        boundary="The bounded fixture trace is not a sandbox for arbitrary binaries.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-003",
        claim_type="observation_model",
        statement="Decoded assembly events are encoded as categorical observations for a generative model.",
        support_keys=("friston2009freeenergy", "friston2017activeinference", "heins2022pymdp"),
        artifact_paths=("output/data/trace_bundle.json", "output/data/inference_result.json"),
        validation_checks=("trace_observation_semantics", "scholarship_claim_boundaries"),
        boundary="The observation vocabulary is hand-authored and does not prove program understanding.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-004",
        claim_type="rxinfer_backend",
        statement="RxInfer emits the authoritative Bethe Free Energy trajectory for this project.",
        support_keys=("bagaev2023rxinfer", "rxinfer_static_docs", "rxinfer_python_docs"),
        artifact_paths=(
            "output/rxinfer/free_energy_trace.jsonl",
            "output/rxinfer/diagnostics.json",
            "output/rxinfer/posteriors.json",
        ),
        validation_checks=("rxinfer_free_energy_records", "rxinfer_companion_artifacts"),
        boundary="The current model is a Gaussian event-code demonstrator, not the planned categorical HMM.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-005",
        claim_type="comparison_baseline",
        statement="The pymdp path is a comparison baseline rather than a source of RxInfer Bethe Free Energy.",
        support_keys=("heins2022pymdp",),
        artifact_paths=("output/data/inference_result.json", "output/figures/inference_diagnostics.png"),
        validation_checks=("figure_inference_diagnostics", "scholarship_claim_boundaries"),
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
        artifact_paths=("output/data/trace_bundle.json", "output/data/scholarship_sources.json"),
        validation_checks=("trace_optional_roadmap_provenance", "scholarship_evidence_partition"),
        boundary="Roadmap package citations are not evidence that those tools ran in v0.1.",
        status="validated",
    ),
    ClaimRecord(
        claim_id="CLM-007",
        claim_type="adversarial_validation",
        statement="RedTeam findings are durable generated artifacts rather than conversation-only critique.",
        support_keys=(),
        artifact_paths=("output/data/redteam_report.json", "output/figures/redteam_risk_register.png"),
        validation_checks=("redteam_findings_remediated", "figure_redteam_risk_register"),
        boundary="Remediated findings strengthen project contracts but do not replace peer review.",
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
        validation_checks=("scholarship_source_quality", "scholarship_evidence_partition"),
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
        statement="A left-to-right transition matrix yields a temporal HMM whose forward-filter decoding substantially outperforms the v0.1 sticky identity-transition recursive filter on the labelled corpus.",
        support_keys=("friston2017activeinference", "heins2022pymdp"),
        artifact_paths=(
            "output/data/temporal_inference.json",
            "output/data/evaluation_report.json",
        ),
        validation_checks=(
            "advanced_temporal_artifacts_present",
            "advanced_temporal_beats_static",
            "advanced_temporal_isolates_transition",
            "advanced_memoryless_baseline_honest",
        ),
        boundary="The claim is bounded to beating the v0.1 STICKY filter only: on this short-program corpus a simpler MEMORYLESS per-step decoder (argmax D·A) outperforms the temporal HMM, so temporal structure shows no NET advantage here and this is recorded explicitly. The static and temporal decoders share A and D (only B differs), parameters are hand-authored not fit, and the exit phase is observation-underdetermined.",
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
        validation_checks=("advanced_active_never_worse", "advanced_active_policy_functional"),
        boundary="Expected free energy genuinely varies across candidates (functional policy, not collapsed), so the corpus tie is a real null — no measurable advantage at this fixture scale — not a non-working policy; the active advantage is shown only as a mechanism in unit tests.",
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
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
