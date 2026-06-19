"""Durable RedTeam findings for the active assembly scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class RedTeamFinding:
    """One concrete adversarial finding and the implemented remediation."""

    finding_id: str
    vector: str
    severity: str
    location: str
    false_certification_path: str
    negative_control: str
    remediation: str
    validation_check: str
    status: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe finding row."""

        return asdict(self)


REDTEAM_FINDINGS: tuple[RedTeamFinding, ...] = (
    RedTeamFinding(
        finding_id="RT-001",
        vector="source_shape_quality",
        severity="critical",
        location="src/scholarship.py",
        false_certification_path="A source ledger entry could have an empty role, weak URL, or malformed DOI while bibliography coverage still passed.",
        negative_control="Add a source with doi='not-a-doi' or url='http://example.invalid' and require validation to fail.",
        remediation="Source-quality checks now validate DOI shape, HTTPS URLs, source-type vocabulary, role text, and claim-boundary text.",
        validation_check="scholarship_source_quality",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-002",
        vector="citation_closure",
        severity="critical",
        location="manuscript/*.md and manuscript/references.bib",
        false_certification_path="The manuscript could cite a key not present in references.bib while source-ledger coverage still passed.",
        negative_control="Add [@missing_key] to a manuscript section and require validation to fail.",
        remediation="Citation closure now checks that every manuscript citation key resolves to a bibliography entry.",
        validation_check="scholarship_citation_closure",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-003",
        vector="render_surface_contract",
        severity="investigate",
        location="manuscript/ and output/manuscript/",
        false_certification_path="A render could omit a section or leave a figure reference out while script-level hydration still reported success.",
        negative_control="Remove manuscript/05_validation.md or a figure reference and require validation to report an incomplete manuscript surface.",
        remediation="Manuscript-source checks now require the expected section set and expected figure references.",
        validation_check="manuscript_source_surface",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-004",
        vector="visualization_semantics",
        severity="investigate",
        location="output/figures/",
        false_certification_path="A zero-byte or blank PNG could exist and satisfy a path-only figure check.",
        negative_control="Overwrite one expected PNG with a constant image and require validation to fail.",
        remediation="Figure checks now test file size and pixel variance for every expected visualization.",
        validation_check="figure_*",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-005",
        vector="critique_durability",
        severity="investigate",
        location="output/data/redteam_report.json",
        false_certification_path="RedTeam conclusions could live only in chat and disappear from the project artifact set.",
        negative_control="Delete output/data/redteam_report.json or set a finding status to open and require validation to fail.",
        remediation="RedTeam findings are now generated as JSON/Markdown and checked by the validation report.",
        validation_check="redteam_findings_remediated",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-006",
        vector="trace_semantic_contract",
        severity="critical",
        location="output/data/trace_bundle.json",
        false_certification_path="A trace bundle could preserve required keys while observation IDs, categories, labels, or instruction addresses no longer agreed.",
        negative_control="Change one observation category without changing its observation_id and require validation to fail.",
        remediation="Trace validation now checks observation IDs against the category vocabulary, address alignment, label vocabulary, and dynamic register trace shape.",
        validation_check="trace_observation_semantics",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-007",
        vector="rxinfer_artifact_consistency",
        severity="critical",
        location="output/rxinfer/",
        false_certification_path="A finite free-energy JSONL file could pass while posteriors, model configuration, or execution logs were missing or stale.",
        negative_control="Delete posteriors.json or mismatch its run_id and require validation to fail.",
        remediation="RxInfer validation now checks companion artifacts, posterior iteration count, model observation values, logs, and diagnostic run consistency.",
        validation_check="rxinfer_companion_artifacts",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-008",
        vector="diagnostic_visualization",
        severity="investigate",
        location="output/figures/inference_diagnostics.png",
        false_certification_path="The manuscript could discuss separate pymdp and RxInfer diagnostics without a figure showing that they are distinct objective traces.",
        negative_control="Remove inference_diagnostics.png or replace it with a blank image and require validation to fail.",
        remediation="The pipeline now writes a two-panel diagnostic figure and validates it as a nonblank generated output.",
        validation_check="figure_inference_diagnostics",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-009",
        vector="scholarship_stage_laundering",
        severity="critical",
        location="src/scholarship.py and output/data/scholarship_sources.json",
        false_certification_path="A source could be cited as if it supported implementation evidence while it was only roadmap or theory context.",
        negative_control="Remove project_stage or evidence_tier from a source row and require validation to fail.",
        remediation="Scholarship sources now carry project_stage and evidence_tier fields, and validation checks the implemented/comparison/roadmap/context partition.",
        validation_check="scholarship_evidence_partition",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-010",
        vector="package_roadmap_overclaim",
        severity="investigate",
        location="docs/assembly_packages.md and manuscript/01_introduction.md",
        false_certification_path="A richer package roadmap could be read as evidence that Ghidra, BAP, GTIRB, Qiling, Miasm, Dyninst, RetDec, or McSema already ran.",
        negative_control="Move a roadmap-only package into the implemented stage without an adapter artifact and require scholarship validation to fail.",
        remediation="The manuscript and docs now state that expanded packages are future adapter surfaces, and the trace bundle records them as optional roadmap provenance rows.",
        validation_check="trace_optional_roadmap_provenance",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-011",
        vector="claim_register_closure",
        severity="critical",
        location="src/claim_register.py and output/data/claim_register.json",
        false_certification_path="The manuscript could add a confident claim that appears source-aware but is not tied to an artifact path or executable validation check.",
        negative_control="Add a claim with an unknown source key, missing artifact path, or nonexistent validation check and require validation to fail.",
        remediation="Claims are now generated as JSON/Markdown with source keys, artifact paths, validation checks, boundaries, and a nonblank claim-evidence figure.",
        validation_check="claim_register_contract",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-012",
        vector="artifact_reproducibility_manifest",
        severity="critical",
        location="output/data/reproducibility_manifest.json",
        false_certification_path="Generated artifacts could be stale, replaced, or silently edited while path-existence checks and prose still reported a validated analysis.",
        negative_control="Edit one manifest-tracked artifact after manifest generation and require validation to fail on a hash mismatch.",
        remediation="The pipeline now writes a reproducibility manifest with file sizes and SHA-256 hashes, and validation recomputes each tracked artifact hash.",
        validation_check="reproducibility_artifact_manifest",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-013",
        vector="temporal_superiority_overclaim",
        severity="critical",
        location="src/temporal_model.py and output/data/evaluation_report.json",
        false_certification_path="The temporal HMM could be asserted as superior to the original static filter without a head-to-head measurement, or the metric could be vacuous (every decoder scores the same).",
        negative_control="Replace the transition matrix with the identity (B=I) so the temporal filter collapses to the static one, or shuffle the ground-truth labels; require the temporal-beats-static and negative-control checks to fail.",
        remediation="Evaluation recomputes static, temporal, and Viterbi decoding accuracy from the labelled corpus and validation asserts mean temporal accuracy exceeds static and beats shuffled/uniform controls by a margin.",
        validation_check="advanced_temporal_beats_static",
        status="remediated",
    ),
    RedTeamFinding(
        finding_id="RT-014",
        vector="active_inference_cherry_pick",
        severity="critical",
        location="src/active_selection.py and output/data/active_selection.json",
        false_certification_path="Active selection could be advertised as 'helping' by cherry-picking a confidence threshold or a single favourable fixture, laundering a mechanism demo into a corpus-scale claim.",
        negative_control="Record the honest per-fixture active-vs-fixed step counts; require validation to fail if active is ever strictly worse than the fixed scan, and confine the active-advantage claim to a unit test of the mechanism.",
        remediation="The active artifact records the measured corpus outcome including ties; validation asserts active is never worse than the fixed scan, and the claim register marks the corpus advantage explicitly as a non-result.",
        validation_check="advanced_active_never_worse",
        status="remediated",
    ),
)


def redteam_payload() -> dict[str, object]:
    """Return the complete RedTeam report payload."""

    remediated = sum(finding.status == "remediated" for finding in REDTEAM_FINDINGS)
    return {
        "finding_count": len(REDTEAM_FINDINGS),
        "remediated_count": remediated,
        "open_count": len(REDTEAM_FINDINGS) - remediated,
        "execution_mode": "internal_specialist_panel",
        "target_classification": "structured artifact with verifier-first review",
        "findings": [finding.to_dict() for finding in REDTEAM_FINDINGS],
    }


def write_redteam_report(output_dir: Path = OUTPUT_DIR) -> str:
    """Write RedTeam report artifacts and return the JSON path relative to the project."""

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "redteam_report.json"
    payload = redteam_payload()
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown_path = data_dir / "redteam_report.md"
    rows = [
        "# RedTeam Findings",
        "",
        "| ID | Vector | Severity | Status | Validation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for finding in REDTEAM_FINDINGS:
        rows.append(
            f"| `{finding.finding_id}` | {finding.vector} | {finding.severity} | "
            f"{finding.status} | `{finding.validation_check}` |"
        )
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    from .figures import plot_redteam_risk_register

    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    plot_redteam_risk_register(
        [finding.to_dict() for finding in REDTEAM_FINDINGS],
        output_dir / "figures" / "redteam_risk_register.png",
    )
    return str(json_path.relative_to(PROJECT_ROOT))
