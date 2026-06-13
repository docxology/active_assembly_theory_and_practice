"""Analysis validation report for scholarship, traces, inference, and figures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

import matplotlib.image as mpimg
import numpy as np

from .assembly_packages.schema import TRACE_SCHEMA_VERSION, binary_hash
from .claim_register import CLAIM_RECORDS
from .features import HIDDEN_STATE_LABELS, OBSERVATION_LABELS
from .figures import plot_validation_summary
from .redteam_report import REDTEAM_FINDINGS
from .reproducibility import MANIFEST_TARGETS, verify_reproducibility_manifest
from .rxinfer_bridge import file_sha256, load_free_energy_records, validate_free_energy_records
from .scholarship import scholarship_keys, scholarship_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class ValidationCheck:
    """One validation assertion with a compact diagnostic string."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe check row."""

        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    """Complete validation report for generated analysis outputs."""

    generated_at: str
    checks: tuple[ValidationCheck, ...]
    summary: dict[str, Any]
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report."""

        return {
            "generated_at": self.generated_at,
            "checks": [check.to_dict() for check in self.checks],
            "summary": dict(self.summary),
            "artifact_paths": dict(self.artifact_paths),
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, condition: bool, detail: str) -> ValidationCheck:
    return ValidationCheck(name=name, passed=bool(condition), detail=detail)


def _trace_checks(trace_path: Path) -> list[ValidationCheck]:
    payload = _load_json(trace_path)
    required = {
        "schema_version",
        "sample_id",
        "architecture",
        "binary_sha256",
        "static_instruction_events",
        "observations",
        "package_provenance",
        "basic_blocks",
    }
    missing = sorted(required - set(payload))
    observations = payload.get("observations", [])
    instructions = payload.get("static_instruction_events", [])
    provenance = {row["name"]: row for row in payload.get("package_provenance", [])}
    required_packages = {"capstone", "unicorn"}
    optional_roadmap_packages = {
        "angr",
        "rizin",
        "radare2",
        "llvm-objdump",
        "ghidra",
        "bap",
        "gtirb",
        "qiling",
        "miasm",
        "dyninst",
        "retdec",
        "mcsema",
        "keystone",
        "nasm",
    }
    try:
        expected_binary_hash = binary_hash(str(payload.get("sample_id")))
    except ValueError:
        expected_binary_hash = None
    observation_semantics_ok = bool(observations) and bool(instructions)
    if observation_semantics_ok:
        for observation in observations:
            index = int(observation.get("index", -1))
            category = str(observation.get("category", ""))
            observation_id = int(observation.get("observation_id", -1))
            if (
                index < 0
                or index >= len(instructions)
                or category not in OBSERVATION_LABELS
                or observation_id != OBSERVATION_LABELS.index(category)
                or observation.get("address") != instructions[index].get("address")
            ):
                observation_semantics_ok = False
                break
    label_vocab_ok = payload.get("observation_labels") == list(OBSERVATION_LABELS) and payload.get(
        "hidden_state_labels"
    ) == list(HIDDEN_STATE_LABELS)
    dynamic_trace = payload.get("dynamic_register_trace", [])
    dynamic_trace_ok = (
        len(dynamic_trace) >= 2
        and dynamic_trace[0].get("event") == "initial"
        and dynamic_trace[-1].get("event") == "final"
        and int(dynamic_trace[-1].get("instruction_count", -1)) == len(instructions)
    )
    return [
        _check("trace_required_keys", not missing, f"missing={missing}"),
        _check(
            "trace_schema_version",
            payload.get("schema_version") == TRACE_SCHEMA_VERSION,
            f"schema={payload.get('schema_version')}",
        ),
        _check(
            "trace_binary_hash",
            payload.get("binary_sha256") == expected_binary_hash,
            f"sha256={payload.get('binary_sha256')}",
        ),
        _check(
            "trace_instruction_observation_alignment",
            len(instructions) == len(observations) and len(instructions) > 0,
            f"instructions={len(instructions)} observations={len(observations)}",
        ),
        _check(
            "trace_required_package_provenance",
            required_packages.issubset(provenance)
            and all(provenance[name]["available"] for name in required_packages),
            f"packages={sorted(provenance)}",
        ),
        _check(
            "trace_optional_roadmap_provenance",
            optional_roadmap_packages.issubset(provenance)
            and all(provenance[name]["optional"] for name in optional_roadmap_packages),
            f"optional_packages={sorted(optional_roadmap_packages & set(provenance))}",
        ),
        _check(
            "trace_basic_block_coverage",
            bool(payload.get("basic_blocks"))
            and payload["basic_blocks"][0].get("instruction_indexes") == list(range(len(instructions))),
            f"blocks={len(payload.get('basic_blocks', []))}",
        ),
        _check(
            "trace_observation_semantics",
            observation_semantics_ok and label_vocab_ok,
            f"observations={len(observations)} labels_ok={label_vocab_ok}",
        ),
        _check(
            "trace_dynamic_register_contract",
            dynamic_trace_ok,
            f"dynamic_register_events={len(dynamic_trace)}",
        ),
    ]


def validate_trace_file(trace_path: Path) -> list[ValidationCheck]:
    """Validate one trace bundle file and return check rows."""

    return _trace_checks(trace_path)


def _rxinfer_checks(trace_path: Path, rxinfer_dir: Path) -> list[ValidationCheck]:
    records = load_free_energy_records(rxinfer_dir / "free_energy_trace.jsonl")
    diagnostics = _load_json(rxinfer_dir / "diagnostics.json")
    posteriors = _load_json(rxinfer_dir / "posteriors.json")
    model_config = _load_json(rxinfer_dir / "model_config.json")
    expected_data_hash = file_sha256(trace_path)
    checks: list[ValidationCheck] = []
    try:
        validate_free_energy_records(records, expected_data_hash=expected_data_hash)
    except Exception as exc:  # noqa: BLE001 - validation report must serialize failure text
        checks.append(_check("rxinfer_free_energy_records", False, str(exc)))
    else:
        checks.append(
            _check(
                "rxinfer_free_energy_records",
                True,
                f"records={len(records)} final={records[-1]['value']}",
            )
        )
    checks.append(
        _check(
            "rxinfer_diagnostics_consistent",
            diagnostics.get("iteration_count") == len(records)
            and diagnostics.get("all_free_energy_finite") is True
            and diagnostics.get("data_hash") == expected_data_hash,
            f"iterations={diagnostics.get('iteration_count')} all_finite={diagnostics.get('all_free_energy_finite')}",
        )
    )
    companion_files = ("posteriors.json", "model_config.json", "stdout.log", "stderr.log")
    companion_files_ok = all((rxinfer_dir / filename).exists() for filename in companion_files)
    posterior_rows = posteriors.get("posteriors", {}).get("μ", [])
    config_values = model_config.get("observation_values", [])
    checks.append(
        _check(
            "rxinfer_companion_artifacts",
            companion_files_ok
            and posteriors.get("run_id") == diagnostics.get("run_id")
            and len(posterior_rows) == diagnostics.get("iteration_count")
            and len(config_values) > 0,
            f"companions={companion_files_ok} posterior_rows={len(posterior_rows)}",
        )
    )
    return checks


def _figure_checks(figure_dir: Path) -> list[ValidationCheck]:
    expected = (
        "observation_pipeline.png",
        "posterior_beliefs.png",
        "package_provenance.png",
        "trace_feature_matrix.png",
        "rxinfer_free_energy.png",
        "inference_diagnostics.png",
        "scholarship_source_roles.png",
        "scholarship_evidence_matrix.png",
        "claim_evidence_map.png",
        "redteam_risk_register.png",
        "artifact_hash_manifest.png",
    )
    checks: list[ValidationCheck] = []
    for filename in expected:
        path = figure_dir / filename
        if not path.exists():
            checks.append(_check(f"figure_{path.stem}", False, "missing"))
            continue
        image = np.asarray(mpimg.imread(path), dtype=float)
        checks.append(
            _check(
                f"figure_{path.stem}",
                path.stat().st_size > 0 and image.size > 0 and float(np.var(image)) > 0.00001,
                f"bytes={path.stat().st_size} variance={float(np.var(image)):.6f}",
            )
        )
    return checks


def _scholarship_checks(project_root: Path) -> list[ValidationCheck]:
    references = (project_root / "manuscript" / "references.bib").read_text(encoding="utf-8")
    manuscript_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((project_root / "manuscript").glob("*.md"))
    )
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", references))
    cited_keys = {
        key
        for match in re.findall(r"@([A-Za-z0-9_:-]+)", manuscript_text)
        for key in [match.rstrip(".,;]")]
        if not key.startswith(("fig:", "tab:", "sec:", "eq:"))
    }
    ledger_keys = scholarship_keys()
    missing_bib = sorted(ledger_keys - bib_keys)
    missing_usage = sorted((ledger_keys & bib_keys) - cited_keys)
    unresolved_citations = sorted(cited_keys - bib_keys)
    payload = scholarship_payload()
    sources = payload["sources"]
    allowed_source_types = {
        "peer_reviewed_theory",
        "software_paper",
        "software_documentation",
        "peer_reviewed_binary_analysis",
        "software_repository",
        "technical_report",
    }
    allowed_evidence_tiers = {
        "theory",
        "implemented_backend",
        "implemented_dependency",
        "integration_guidance",
        "comparison_baseline",
        "binary_analysis_context",
        "roadmap_adapter",
    }
    allowed_project_stages = {
        "implemented",
        "comparison",
        "roadmap",
        "context",
    }
    doi_pattern = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
    source_quality_ok = all(
        source["source_type"] in allowed_source_types
        and source["evidence_tier"] in allowed_evidence_tiers
        and source["project_stage"] in allowed_project_stages
        and str(source["url"]).startswith("https://")
        and (source["doi"] is None or bool(doi_pattern.match(str(source["doi"]))))
        and len(str(source["project_role"])) >= 20
        and len(str(source["claim_boundary"])) >= 20
        for source in sources
    )
    project_stage_counts = payload["project_stage_counts"]
    evidence_tier_counts = payload["evidence_tier_counts"]
    evidence_partition_ok = (
        set(project_stage_counts).issubset(allowed_project_stages)
        and set(evidence_tier_counts).issubset(allowed_evidence_tiers)
        and int(project_stage_counts.get("implemented", 0)) >= 4
        and int(project_stage_counts.get("roadmap", 0)) >= 10
        and int(project_stage_counts.get("context", 0)) >= 3
        and int(project_stage_counts.get("comparison", 0)) >= 1
    )
    return [
        _check(
            "scholarship_source_quality",
            source_quality_ok and int(payload["source_count"]) >= 20,
            f"sources={payload['source_count']} types={sorted(payload['source_type_counts'])}",
        ),
        _check(
            "scholarship_evidence_partition",
            evidence_partition_ok,
            f"stages={project_stage_counts} tiers={evidence_tier_counts}",
        ),
        _check(
            "scholarship_bibliography_coverage",
            not missing_bib,
            f"missing_bib={missing_bib}",
        ),
        _check(
            "scholarship_citation_closure",
            not unresolved_citations,
            f"unresolved_citations={unresolved_citations}",
        ),
        _check(
            "scholarship_manuscript_usage",
            not missing_usage,
            f"uncited_ledger_keys={missing_usage}",
        ),
        _check(
            "scholarship_claim_boundaries",
            "not a general reverse-engineering" in manuscript_text
            and "not yet the planned categorical hidden Markov model" in manuscript_text,
            "cautious claim boundaries present",
        ),
    ]


def _manuscript_source_checks(project_root: Path) -> list[ValidationCheck]:
    manuscript_dir = project_root / "manuscript"
    expected_sections = {
        "00_abstract.md",
        "01_introduction.md",
        "02_methodology.md",
        "03_results.md",
        "04_discussion.md",
        "05_validation.md",
        "06_redteam_limits.md",
        "99_references.md",
    }
    files = {path.name for path in manuscript_dir.glob("*.md")}
    text = "\n".join(path.read_text(encoding="utf-8") for path in manuscript_dir.glob("*.md"))
    expected_figures = {
        "OBSERVATION_FIGURE",
        "PACKAGE_PROVENANCE_FIGURE",
        "TRACE_FEATURE_FIGURE",
        "POSTERIOR_FIGURE",
        "RXINFER_FREE_ENERGY_FIGURE",
        "INFERENCE_DIAGNOSTIC_FIGURE",
        "SCHOLARSHIP_FIGURE",
        "SCHOLARSHIP_MATRIX_FIGURE",
        "CLAIM_FIGURE",
        "VALIDATION_FIGURE",
        "REDTEAM_FIGURE",
        "REPRODUCIBILITY_FIGURE",
    }
    missing_sections = sorted(expected_sections - files)
    missing_figures = sorted(token for token in expected_figures if f"{{{{{token}}}}}" not in text)
    return [
        _check(
            "manuscript_source_surface",
            not missing_sections and not missing_figures,
            f"missing_sections={missing_sections} missing_figures={missing_figures}",
        )
    ]


def _manuscript_token_checks(output_dir: Path) -> list[ValidationCheck]:
    manuscript_dir = output_dir / "manuscript"
    if not manuscript_dir.exists():
        return [_check("rendered_manuscript_tokens", True, "not hydrated yet")]
    unresolved: list[str] = []
    for path in manuscript_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\{\{[A-Z0-9_]+\}\}", text):
            unresolved.append(path.name)
    return [
        _check(
            "rendered_manuscript_tokens",
            not unresolved,
            f"unresolved_files={unresolved}",
        )
    ]


def _redteam_checks(output_dir: Path) -> list[ValidationCheck]:
    report_path = output_dir / "data" / "redteam_report.json"
    if not report_path.exists():
        return [_check("redteam_findings_remediated", False, "missing redteam_report.json")]
    payload = _load_json(report_path)
    findings = payload.get("findings", [])
    expected_ids = {finding.finding_id for finding in REDTEAM_FINDINGS}
    observed_ids = {finding.get("finding_id") for finding in findings}
    all_remediated = all(finding.get("status") == "remediated" for finding in findings)
    validation_names = {finding.get("validation_check") for finding in findings}
    return [
        _check(
            "redteam_findings_remediated",
            payload.get("finding_count") == len(REDTEAM_FINDINGS)
            and expected_ids == observed_ids
            and all_remediated
            and len(validation_names) == len(findings),
            f"findings={payload.get('finding_count')} remediated={payload.get('remediated_count')}",
        )
    ]


def _claim_register_checks(
    project_root: Path,
    observed_check_names: set[str],
) -> list[ValidationCheck]:
    report_path = project_root / "output" / "data" / "claim_register.json"
    if not report_path.exists():
        return [_check("claim_register_contract", False, "missing claim_register.json")]
    payload = _load_json(report_path)
    claims = payload.get("claims", [])
    expected_ids = {claim.claim_id for claim in CLAIM_RECORDS}
    observed_ids = {claim.get("claim_id") for claim in claims}
    ledger_keys = scholarship_keys()
    all_sources_known = all(
        set(claim.get("support_keys", [])).issubset(ledger_keys)
        for claim in claims
    )
    all_artifacts_present = all(
        all((project_root / str(path)).exists() for path in claim.get("artifact_paths", []))
        for claim in claims
    )
    all_checks_known = all(
        set(claim.get("validation_checks", [])).issubset(observed_check_names)
        for claim in claims
    )
    all_validated = all(claim.get("status") == "validated" for claim in claims)
    return [
        _check(
            "claim_register_contract",
            payload.get("claim_count") == len(CLAIM_RECORDS)
            and observed_ids == expected_ids
            and all_sources_known
            and all_artifacts_present
            and all_checks_known
            and all_validated,
            (
                f"claims={payload.get('claim_count')} sources={all_sources_known} "
                f"artifacts={all_artifacts_present} checks={all_checks_known}"
            ),
        )
    ]


def _reproducibility_checks(project_root: Path) -> list[ValidationCheck]:
    manifest_path = project_root / "output" / "data" / "reproducibility_manifest.json"
    if not manifest_path.exists():
        return [_check("reproducibility_artifact_manifest", False, "missing manifest")]
    payload = _load_json(manifest_path)
    errors = verify_reproducibility_manifest(project_root)
    rows = payload.get("artifacts", [])
    hashes_present = all(
        isinstance(row.get("sha256"), str)
        and len(str(row.get("sha256"))) == 64
        and int(row.get("bytes", 0)) > 0
        for row in rows
    )
    return [
        _check(
            "reproducibility_artifact_manifest",
            not errors
            and payload.get("artifact_count") == len(MANIFEST_TARGETS)
            and payload.get("all_required_present") is True
            and hashes_present,
            f"artifacts={payload.get('artifact_count')} errors={errors[:3]}",
        )
    ]


def _documentation_checks(project_root: Path) -> list[ValidationCheck]:
    docs_dir = project_root / "docs"
    expected_docs = {
        "README.md",
        "analysis_validation.md",
        "architecture.md",
        "assembly_packages.md",
        "claim_map.md",
        "claim_register.md",
        "redteam.md",
        "reproducibility.md",
        "rxinfer_roadmap.md",
        "scholarship.md",
        "verification.md",
    }
    observed_docs = {path.name for path in docs_dir.glob("*.md")}
    missing_docs = sorted(expected_docs - observed_docs)
    claim_map_path = docs_dir / "claim_map.md"
    claim_map_text = claim_map_path.read_text(encoding="utf-8") if claim_map_path.exists() else ""
    return [
        _check(
            "documentation_contract_surface",
            not missing_docs
            and "output/rxinfer/free_energy_trace.jsonl" in claim_map_text
            and "output/data/reproducibility_manifest.json" in claim_map_text
            and "Passing validation is a project contract" in claim_map_text,
            f"missing_docs={missing_docs}",
        )
    ]


def build_validation_report(project_root: Path = PROJECT_ROOT) -> ValidationReport:
    """Validate generated analysis artifacts and return a structured report."""

    output_dir = project_root / "output"
    trace_path = output_dir / "data" / "trace_bundle.json"
    rxinfer_dir = output_dir / "rxinfer"
    checks: list[ValidationCheck] = []
    checks.extend(_trace_checks(trace_path))
    checks.extend(_rxinfer_checks(trace_path, rxinfer_dir))
    checks.extend(_figure_checks(output_dir / "figures"))
    checks.extend(_scholarship_checks(project_root))
    checks.extend(_manuscript_source_checks(project_root))
    checks.extend(_manuscript_token_checks(output_dir))
    checks.extend(_redteam_checks(output_dir))
    checks.extend(_reproducibility_checks(project_root))
    checks.extend(_claim_register_checks(project_root, {check.name for check in checks}))
    checks.extend(_documentation_checks(project_root))
    passed = sum(check.passed for check in checks)
    failed = len(checks) - passed
    report_path = output_dir / "data" / "validation_report.json"
    figure_path = output_dir / "figures" / "validation_summary.png"
    return ValidationReport(
        generated_at=datetime.now(UTC).isoformat(),
        checks=tuple(checks),
        summary={
            "passed": passed,
            "failed": failed,
            "total": len(checks),
            "all_passed": failed == 0,
        },
        artifact_paths={
            "validation_report": str(report_path.relative_to(project_root)),
            "validation_summary": str(figure_path.relative_to(project_root)),
        },
    )


def write_validation_report(project_root: Path = PROJECT_ROOT) -> ValidationReport:
    """Write validation JSON and summary figure."""

    report = build_validation_report(project_root)
    output_dir = project_root / "output"
    report_path = output_dir / "data" / "validation_report.json"
    figure_path = output_dir / "figures" / "validation_summary.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_validation_summary([check.to_dict() for check in report.checks], figure_path)
    return report
