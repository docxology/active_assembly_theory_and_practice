"""Source-bound registry for manuscript variables and template hydration.

The manuscript variable JSON is intentionally a small rendering interface: it
contains strings because Pandoc/template hydration consumes strings.  This
module records the typed interpretation, canonical inputs, formatting rule,
source consumers, repository revisions, and the exact template contract beside
that interface so a green render cannot hide a stale or orphaned value.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from .artifact_contract import write_json_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_SCHEMA_VERSION = "1.0"
MANUSCRIPT_VARIABLE_REGISTRY = "output/data/manuscript_variable_registry.json"
TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
SOURCE_EXCLUDES = {"AGENTS.md", "README.md", "SYNTAX.md"}
INTERVAL_RE = re.compile(
    r"^\[\s*(-?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d+)?|\.\d+))\s*\]$"
)
NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d+)?|\.\d+)$")

_PATH_SUFFIXES = (
    "_FIGURE",
    "_ARTIFACT",
    "_LEDGER",
    "_REGISTER",
    "_REPORT",
    "_MANIFEST",
)
_COUNT_SUFFIXES = ("_COUNT", "_STEPS", "_PERCENT")
_PROPORTION_HINTS = (
    "ACCURACY",
    "BALANCED",
    "ECE",
    "F1",
    "FRACTION",
    "RECALL",
    "RATE",
    "CONSISTENCY",
    "SEPARATION",
    "SPAN",
)

VARIABLE_DEFINITIONS: dict[str, str] = {
    "DOCX_RENDER_ARTIFACT": "Standard template-generated combined DOCX manuscript artifact.",
    "DOCX_RENDER_RECEIPT": "Post-render DOCX integrity and source-lineage receipt.",
    "DOCX_RENDER_STATUS": "Compact structural status of the current template-generated DOCX artifact; the dedicated receipt is the freshness gate.",
    "DOCX_RENDER_FIGURE_COUNT": "Number of figure images embedded in the current combined DOCX, or pending before template rendering.",
    "DOCX_RENDER_CAPTION_COUNT": "Number of figure-caption paragraphs in the current combined DOCX, or pending before template rendering.",
    "CHANCE_ACCURACY": "Reciprocal of the number of hidden phase classes.",
    "CI_LEVEL_PERCENT": "Nominal Wilson score interval confidence level in percent.",
    "CALIBRATION_BIN_COUNT": "Number of equal-width bins used for reliability and ECE calculations.",
    "ACTIVE_NEVER_WORSE": "Whether active selection is no worse than the fixed scan at every ablation blend.",
    "DECODER_BEST_CALIBRATED": "Decoder with the smallest expected calibration error on the labelled corpus.",
    "BEST_MODEL": "Model with the largest pooled fixed-parameter log marginal likelihood.",
    "MOST_INFORMATIVE_OBS": "Observation category with the largest realized information gain.",
    "LEAST_INFORMATIVE_OBS": "Observation category with the smallest realized information gain.",
    "MEMORY_DYNAMIC_EVENT_COUNT": "Number of bounded Unicorn memory-hook events in the x86_memory provenance fixture.",
    "MEMORY_READ_COUNT": "Number of dynamic read events in the x86_memory provenance fixture.",
    "MEMORY_WRITE_COUNT": "Number of dynamic write events in the x86_memory provenance fixture.",
    "CORPUS_MEMORY_FIXTURE_COUNT": "Number of corpus fixtures containing explicit memory addressing modes.",
    "CORPUS_ALTERNATE_PATH_FIXTURE_COUNT": "Number of corpus fixtures whose bounded dynamic path omits at least one statically decoded instruction.",
    "CORPUS_REPEATED_PHASE_FIXTURE_COUNT": "Number of corpus fixtures containing a repeated authored phase label.",
    "CORPUS_UNIQUE_INSTRUCTION_FAMILY_COUNT": "Number of distinct typed instruction families in the transparent corpus.",
    "CORPUS_UNIQUE_MEMORY_MODE_COUNT": "Number of distinct non-none memory addressing modes in the transparent corpus.",
    "CORPUS_UNIQUE_BRANCH_CLASS_COUNT": "Number of distinct non-none branch target classes in the transparent corpus.",
    "CORPUS_COVERAGE_ARTIFACT": "Source-bound fixture-design coverage inventory generated from Capstone and bounded Unicorn records.",
    "ACTION_SENSING_ARTIFACT": "Held-out action-conditioned typed-field sensing artifact with predictive, fixed, random, and oracle policy records.",
    "ACTION_SENSING_FIGURE": "Figure comparing the censored action budget and final confidence of the bounded sensing policies.",
    "ACTION_SENSING_LEAKAGE_FREE": "Whether every held-out sensing policy excludes its fixture from fitted field likelihoods.",
    "ACTION_SENSING_ACTIVE_MEAN_ACTIONS": "Mean number of active typed-field queries used before the configured confidence threshold, censored at the common budget.",
    "ACTION_SENSING_FIXED_MEAN_ACTIONS": "Mean number of fixed-order typed-field queries used before the configured confidence threshold, censored at the common budget.",
    "ACTION_SENSING_RANDOM_CONFIDENCE_RATE": "Fraction of held-out fixtures reaching the confidence threshold under the seeded-random query baseline.",
    "ACTION_SENSING_ACTIVE_MINUS_FIXED_ACTIONS": "Active minus fixed mean action count under the common bounded sensing budget.",
    "ACTION_SENSING_COST_WEIGHT": "Fixed normalized action-cost weight used in the action-conditioned expected-free-energy score.",
    "EXTERNAL_SOURCE_ARTIFACT": "Offline source-bound receipt for the pinned licensed-source validation track.",
    "EXTERNAL_SOURCE_STATUS": "Evidence status of the licensed-source validation track; blocked or unavailable is not a zero result.",
    "EXTERNAL_SOURCE_PROJECT_COUNT": "Number of distinct source families recorded in the external-source manifest.",
    "EXTERNAL_SOURCE_LABELABLE_EXAMPLE_COUNT": "Number of mechanically labelable external evaluation examples in the external-source receipt.",
    "EXTERNAL_LABEL_ARTIFACT": "Source-bound external label-contract receipt containing mechanically generated pairs and explicit exclusions.",
    "EXTERNAL_LABEL_STATUS": "Evidence status of the source-bound label-contract producer; lexical witnesses and near-miss controls are not universal semantic proofs.",
    "EXTERNAL_LABEL_ROW_COUNT": "Number of accepted source-bound external pair rows after static compilation and Capstone decoding.",
    "EXTERNAL_LABEL_EQUIVALENT_COUNT": "Number of accepted lexical-equivalence witness rows in the source-bound external contract.",
    "EXTERNAL_LABEL_NEAR_MISS_COUNT": "Number of accepted contract-directed one-change near-miss rows in the source-bound external contract.",
    "EXTERNAL_LABEL_NEGATIVE_COUNT": "Number of accepted non-separating identity-control rows in the source-bound external contract.",
    "EXTERNAL_LABEL_EXCLUSION_COUNT": "Number of source-bound external candidate rows retained as explicit exclusions rather than silently filtered.",
    "EXTERNAL_LABEL_PROJECT_COUNT": "Number of source projects represented by accepted source-bound external label rows.",
    "EXTERNAL_LABEL_REVISION_COUNT": "Number of project revisions represented by accepted source-bound external label rows.",
    "EXTERNAL_LABEL_TEST_PROJECTS": "Project identifiers reserved for held-out source-bound external evaluation.",
    "EXTERNAL_LABEL_COVERAGE_FIGURE": "Source-bound census of accepted external label roles and project/revision split allocation; it is not a model-performance figure.",
    "EXTERNAL_STATIC_ARTIFACT": "Static-only external compilation and Capstone-decoding receipt; it does not contain source-equivalence labels.",
    "EXTERNAL_STATIC_STATUS": "Evidence status of the static-only external compilation receipt.",
    "EXTERNAL_STATIC_ROW_COUNT": "Number of pinned external source rows retained in the static-only receipt, including failed and unavailable rows.",
    "EXTERNAL_STATIC_VALIDATED_COUNT": "Number of pinned external source rows compiled and decoded successfully without executing emitted objects.",
    "EXTERNAL_CAPABILITIES_ARTIFACT": "Locally measured compiler and target capability receipt; advertised targets are not counted unless a static object is produced.",
    "EXTERNAL_CAPABILITIES_STATUS": "Evidence status of the local compiler capability receipt.",
    "EXTERNAL_CAPABILITY_TARGET_COUNT": "Number of compiler-target probes that produced a static object successfully.",
    "INDEPENDENT_REPLAY_STATUS": "Current separately controlled replay endpoint status; this does not imply an independent physical machine or independent reimplementation.",
    "INDEPENDENT_REPLAY_CONTROL_SCOPE": "Control scope recorded by the replay receipt, such as same-host separately controlled guest; this is not a physical-machine or independent-reimplementation claim.",
    "INDEPENDENT_REPLAY_MODEL": "Deterministic model used by the separately controlled replay receipt.",
    "INDEPENDENT_REPLAY_ROW_COUNT": "Number of protected-score rows in the separately controlled replay receipt.",
    "INDEPENDENT_REPLAY_POSITIVE_ROW_COUNT": "Number of positive/equivalence rows in the protected replay score.",
    "INDEPENDENT_REPLAY_NEGATIVE_ROW_COUNT": "Number of negative/near-miss rows in the protected replay score.",
    "INDEPENDENT_REPLAY_BALANCED_ACCURACY": "Balanced accuracy from the protected score of the separately controlled replay; a single replay result, not a new scientific sample.",
    "INDEPENDENT_REPLAY_MACRO_F1": "Macro-F1 from the protected score of the separately controlled replay; a single replay result, not a new scientific sample.",
    "INDEPENDENT_REPLAY_BRIER": "Brier score from the protected score of the separately controlled replay; lower is better and the result is bounded to the frozen bundle.",
    "INDEPENDENT_REPLAY_ECE": "Fixed-bin expected calibration error from the protected score of the separately controlled replay; lower is better and the result is bounded to the frozen bundle.",
    "INDEPENDENT_REPLAY_FALSE_EQUIVALENCE_RATE": "False-equivalence rate from the protected score of the separately controlled replay.",
    "INDEPENDENT_REPLAY_NEAR_MISS_SEPARATION": "Near-miss separation rate from the protected score of the separately controlled replay.",
    "INDEPENDENT_REPLAY_NETWORK_DISABLED": "Whether the separately controlled replay receipt records a network-disabled execution boundary.",
    "REPLICATION_PUBLIC_BUNDLE_SHA256": "SHA-256 digest of the public replay files, excluding private labels and protected scoring outputs.",
    "ARTIFACT_RECEIPT_SCHEMA_VERSION": "Common protocol-v2 receipt schema version layered above each domain-specific artifact schema.",
    "ARTIFACT_RECEIPT_REQUIRED_FIELD_COUNT": "Number of required common receipt fields in the protocol-v2 artifact envelope.",
    "ARTIFACT_RECEIPT_COUNT": "Number of project-owned JSON data artifacts present in the generated output surface.",
    "ARTIFACT_RECEIPT_ERROR_COUNT": "Number of project-owned JSON data artifacts failing the common receipt-envelope validator.",
    "ARTIFACT_RECEIPT_AUDIT_COMMAND": "Canonical offline command for validating the project-owned artifact receipt surface.",
    "ARTIFACT_RECEIPT_CLOSURE_FIGURE": "Source-generated provenance visualization of the protocol, artifact, validation, and document-production boundary.",
    "QUOTIENT_CLASSIFIER_INTERVENTION_COUNT": "Number of initial-state interventions used by the co-primary quotient classifier benchmark.",
    "PARAMETER_LEARNING_ARTIFACT": "Held-out observation-only ordered-HMM EM artifact with training complements, fitted tensors, diagnostics, and comparator scores.",
    "PARAMETER_LEARNING_FIGURE": "Figure comparing observation-only ordered EM, supervised MAP, and fixed-reference held-out scores with training evidence trajectories.",
    "PARAMETER_LEARNING_HOLDOUT_COUNT": "Number of complete-fixture holdouts used in the observation-only parameter-learning comparison.",
    "PARAMETER_LEARNING_MAX_ITERATIONS": "Maximum Baum-Welch iterations permitted for each training-complement fit.",
    "PARAMETER_LEARNING_EM_MEAN_ACCURACY": "Mean held-out temporal accuracy of the observation-only ordered-HMM EM fit.",
    "PARAMETER_LEARNING_SUPERVISED_MEAN_ACCURACY": "Mean held-out temporal accuracy of the supervised MAP comparator fitted on the same training complements.",
    "PARAMETER_LEARNING_REFERENCE_MEAN_ACCURACY": "Mean held-out temporal accuracy of the fixed hand-authored reference comparator.",
    "PARAMETER_LEARNING_EM_MINUS_SUPERVISED": "Paired mean accuracy difference between observation-only ordered EM and supervised MAP on complete-fixture holdouts.",
    "PARAMETER_LEARNING_EM_MINUS_REFERENCE": "Paired mean accuracy difference between observation-only ordered EM and the fixed reference on complete-fixture holdouts.",
    "PARAMETER_LEARNING_CONVERGED_HOLDOUT_COUNT": "Number of held-out fits whose EM parameter delta reached the configured tolerance before the iteration cap.",
    "PARAMETER_LEARNING_LEAKAGE_FREE": "Whether phase labels are excluded from every parameter-learning fit and held-out fixtures are excluded from training observations.",
    "PARAMETER_LEARNING_MULTI_START_COUNT": "Number of retained observation-only fits across all holdouts and declared initializations.",
    "PARAMETER_LEARNING_MULTI_START_MEAN_ACCURACY": "Mean held-out temporal accuracy pooled across the declared multi-start fits.",
    "PARAMETER_LEARNING_MULTI_START_MEAN_RANGE": "Mean within-holdout accuracy range across the declared multi-start initializations.",
    "PARAMETER_LEARNING_MULTI_START_MEAN_AGREEMENT": "Mean pairwise agreement among multi-start predicted phase sequences within each held-out fixture.",
    "SCIENTIFIC_PROGRAM_ARTIFACT": "Machine-readable long-horizon scientific program with conjectures, falsifiers, dependencies, promotion gates, and stop conditions.",
    "SCIENTIFIC_PROGRAM_MILESTONE_COUNT": "Number of milestones in the machine-readable long-horizon scientific program.",
    "SCIENTIFIC_PROGRAM_IMPLEMENTED_COUNT": "Number of program milestones currently implemented only on the local source-bound surface.",
    "SCIENTIFIC_PROGRAM_ACTIVE_COUNT": "Number of program milestones currently marked active.",
    "SCIENTIFIC_PROGRAM_PLANNED_COUNT": "Number of program milestones still planned and not yet earned by evidence.",
    "QUOTIENT_PROTOCOL_ARTIFACT": "First bounded P3 quotient/invariance protocol with real equivalent-pair and near-miss Capstone/Unicorn witnesses.",
    "REPRESENTATION_BENCHMARK_ARTIFACT": "Source-family-disjoint P3 representation-invariance benchmark with preregistered endpoints and bounded protocol diagnostics.",
    "REPRESENTATION_BENCHMARK_FIGURE": "Figure comparing the primary and deliberately insufficient negative-control P3 endpoints with descriptive family-level Wilson intervals.",
    "REPRESENTATION_BENCHMARK_FAMILY_COUNT": "Number of source-program transformation families in the P3 representation-invariance benchmark.",
    "REPRESENTATION_BENCHMARK_PAIR_COUNT": "Number of equivalent and near-miss transformation pairs in the P3 representation-invariance benchmark.",
    "REPRESENTATION_BENCHMARK_EQUIVALENCE_CONSISTENCY": "Observed final-RAX equivalence consistency over declared equivalent pairs; protocol diagnostic only.",
    "REPRESENTATION_BENCHMARK_NEAR_MISS_SEPARATION": "Observed final-RAX separation rate over declared near-miss pairs; protocol diagnostic only.",
    "REPRESENTATION_BENCHMARK_SURFACE_BLIND_NEAR_MISS_COUNT": "Number of declared near-miss pairs whose coarse observation-category surface does not change.",
    "REPRESENTATION_BENCHMARK_NEGATIVE_CONTROL_SEPARATION": "Observed final-RAX separation rate under the deliberately insufficient single-state intervention catalog; negative-control diagnostic only.",
    "REPRESENTATION_BENCHMARK_NEGATIVE_CONTROL_UNSEPARATED_COUNT": "Number of near-miss pairs that remain unresolved under the deliberately insufficient single-state intervention catalog.",
    "INTERVENTION_POWER_ARTIFACT": "Exact finite intervention-subset enumeration and separating-set receipt derived from the representation-invariance outcome matrix.",
    "INTERVENTION_POWER_FIGURE": "Figure showing exact separation power across all finite intervention subsets and singleton interventions.",
    "INTERVENTION_POWER_SUBSET_COUNT": "Number of non-empty intervention subsets enumerated in the finite quotient catalog.",
    "INTERVENTION_POWER_PAIR_COUNT": "Number of equivalent and near-miss pair outcomes in the intervention-power matrix.",
    "INTERVENTION_POWER_MINIMAL_CARDINALITY": "Minimum number of declared interventions in any fully separating subset, or an explicit unavailable marker.",
    "INTERVENTION_POWER_MINIMAL_SET_COUNT": "Number of tied minimum-cardinality fully separating intervention subsets.",
    "INTERVENTION_POWER_SINGLE_STATE_SEPARATION": "Exact near-miss separation proportion for the deliberately insufficient one-state intervention control.",
    "INTERVENTION_POWER_SINGLE_STATE_UNRESOLVED_COUNT": "Number of near-miss families unresolved by the one-state intervention control.",
    "SYMBOLIC_STEPWISE_RAX_AGREEMENT_COUNT": "Number of executed instruction steps whose symbolic post-step RAX equals the independent Unicorn snapshot.",
    "SYMBOLIC_STEPWISE_RAX_COMPARABLE_COUNT": "Number of executed instruction steps with an unambiguous Unicorn architectural snapshot available for symbolic comparison.",
    "SYMBOLIC_STEPWISE_FLAG_AGREEMENT_COUNT": "Number of explicitly modeled EFLAGS bit comparisons that agree with post-step Unicorn flags.",
    "SYMBOLIC_STEPWISE_FLAG_COMPARISON_COUNT": "Number of EFLAGS bit comparisons defined by the bounded transfer functions and checked against Unicorn.",
    "TRACE_REFINEMENT_FIGURE": "Stepwise symbolic-versus-Unicorn RAX overlay and modeled-EFLAGS comparison figure.",
    "RXINFER_CATEGORICAL_FINAL_FREE_ENERGY": "Final Bethe Free Energy from the namespaced categorical RxInfer companion; model-specific scale.",
}


def _git_value(root: Path, *args: str) -> str:
    """Return a read-only git value or an explicit unavailable marker."""

    try:
        result = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    value = result.stdout.strip()
    return value or "unavailable"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_diff_sha256(root: Path) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(root),
                "diff",
                "HEAD",
                "--no-ext-diff",
                "--binary",
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def manuscript_source_paths(project_root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    """Return the renderable authored manuscript source files."""

    return tuple(
        sorted(
            path
            for path in (project_root / "manuscript").glob("*.md")
            if path.name not in SOURCE_EXCLUDES
        )
    )


def source_token_consumers(
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Return token consumers and occurrence counts from authored source."""

    consumers: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for path in manuscript_source_paths(project_root):
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(project_root))
        for match in TOKEN_RE.finditer(text):
            token = match.group(1)
            counts[token] = counts.get(token, 0) + 1
            rows = consumers.setdefault(token, [])
            if not rows or rows[-1]["path"] != relative:
                rows.append({"path": relative, "occurrences": 0})
            rows[-1]["occurrences"] += 1
    return consumers, counts


def _raw_value(token: str, rendered: str) -> tuple[str, object]:
    """Infer a JSON-safe typed view of the already-computed value."""

    if rendered.lower() in {"true", "false"}:
        return "boolean", rendered.lower() == "true"
    interval = INTERVAL_RE.fullmatch(rendered)
    if interval:
        return "interval", {
            "lower": float(interval.group(1)),
            "upper": float(interval.group(2)),
        }
    if NUMBER_RE.fullmatch(rendered):
        if rendered.lstrip("-").isdigit():
            return "integer", int(rendered)
        return "number", float(rendered)
    if token.endswith(_PATH_SUFFIXES) or rendered.startswith(("../", "output/")):
        return "path", rendered
    return "text", rendered


def _unit_for(token: str, raw_type: str) -> str | None:
    if token.endswith("_PERCENT"):
        return "percent"
    if raw_type == "interval" and "ACCURACY" in token:
        return "proportion"
    if any(hint in token for hint in _PROPORTION_HINTS):
        return "proportion"
    if raw_type == "integer" and (
        token.endswith(_COUNT_SUFFIXES) or "COUNT" in token or "STEPS" in token
    ):
        return "count"
    return None


def _precision(rendered: str) -> int | None:
    match = re.search(r"\.(\d+)", rendered)
    return len(match.group(1)) if match else None


def _canonical_source(token: str) -> str:
    if token.endswith("_FIGURE"):
        return "src.figures / src.advanced_figures"
    if token.endswith(_PATH_SUFFIXES):
        return "src.manuscript_variables.generate_variables"
    if token.startswith("RXINFER_CATEGORICAL"):
        return "src.rxinfer_bridge.run_rxinfer_categorical_backend"
    if token.startswith(("RXINFER_", "JULIA_")):
        return "src.rxinfer_bridge.run_rxinfer_backend"
    if token.startswith(
        (
            "ACTIVE_",
            "AMBIGUITY_",
            "SEG_",
            "PHASE_",
            "GENERATIVE_ADEQUACY_",
            "PARAMETER_LEARNING_",
        )
    ):
        return "src.advanced_inference.run_advanced_inference"
    if token.startswith(
        (
            "TEMPORAL_",
            "IDENTITY_TRANSITION_",
            "MEMORYLESS_",
            "SHUFFLE_",
            "UNIFORM_",
        )
    ):
        return "src.evaluation.evaluation_payload"
    if token.startswith(("SCHOLARSHIP_", "CLAIM_")):
        return "src.scholarship / src.claim_register"
    if token.startswith(("HIDDEN_", "OBSERVATION_", "CHANCE_")):
        return "src.features"
    return "src.manuscript_variables.generate_variables"


def _template_root(project_root: Path) -> Path | None:
    explicit = os.environ.get("TEMPLATE_REPO_ROOT")
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend(
        parent / "template" for parent in (project_root.parent, *project_root.parents)
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "infrastructure/rendering/manuscript_injection.py").is_file():
            return resolved
    return None


def _template_hydration(project_root: Path) -> dict[str, Any]:
    root = _template_root(project_root)
    output_dir = project_root / "output" / "manuscript"
    unresolved: list[str] = []
    if output_dir.is_dir():
        for path in sorted(output_dir.glob("*.md")):
            if TOKEN_RE.search(path.read_text(encoding="utf-8")):
                unresolved.append(str(path.relative_to(project_root)))
    return {
        "status": (
            "hydrated" if output_dir.is_dir() and not unresolved else "not_yet_hydrated"
        ),
        "template_repo_root": str(root) if root else "unavailable",
        "template_revision": _git_value(root, "rev-parse", "HEAD")
        if root
        else "unavailable",
        "template_contract": (
            "infrastructure.rendering.manuscript_injection."
            "write_resolved_manuscript_tree"
        ),
        "output_directory": str(output_dir.relative_to(project_root)),
        "source_files_expected": [
            path.name for path in manuscript_source_paths(project_root)
        ],
        "output_files_observed": sorted(path.name for path in output_dir.glob("*.md"))
        if output_dir.is_dir()
        else [],
        "unresolved_tokens": unresolved,
    }


def _input_hashes(
    project_root: Path, token: str, consumers: list[dict[str, Any]]
) -> dict[str, str]:
    paths = [
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        project_root / "src" / "manuscript_variables.py",
        project_root / "src" / "manuscript_registry.py",
    ]
    paths.extend(project_root / row["path"] for row in consumers)
    if token.endswith("_FIGURE"):
        figure_stem = {
            "ASSEMBLY_MEMORY_FIGURE": "assembly_memory_timeline",
        }.get(token, token[:-7].lower())
        paths.append(project_root / "output" / "figures" / f"{figure_stem}.png")
    artifact_paths = {
        "evaluation_report": project_root / "output/data/evaluation_report.json",
        "active_selection": project_root / "output/data/active_selection.json",
        "trace_bundle": project_root / "output/data/trace_bundle.json",
        "temporal_inference": project_root / "output/data/temporal_inference.json",
    }
    if token.startswith(
        (
            "TEMPORAL_",
            "IDENTITY_TRANSITION_",
            "MEMORYLESS_",
            "SHUFFLE_",
            "UNIFORM_",
            "DECODER_",
            "BEST_MODEL",
        )
    ):
        paths.append(artifact_paths["evaluation_report"])
    elif token.startswith("ACTIVE_"):
        paths.append(artifact_paths["active_selection"])
    elif token.startswith("ACTION_SENSING_"):
        paths.append(project_root / "output" / "data" / "action_sensing.json")
    elif token in {
        "INSTRUCTION_COUNT",
        "OBSERVATION_COUNT",
        "FINAL_RAX",
        "BINARY_SHA256",
    }:
        paths.append(artifact_paths["trace_bundle"])
    elif token.startswith("PHASE_") or token.startswith("SEG_"):
        paths.append(artifact_paths["temporal_inference"])
    elif token.startswith("MEMORY_"):
        paths.append(project_root / "output" / "data" / "memory_evidence.json")
    elif token.startswith("CORPUS_"):
        paths.append(project_root / "output" / "data" / "corpus_coverage.json")
    elif token.startswith("PARAMETER_LEARNING_"):
        paths.append(project_root / "output" / "data" / "parameter_learning.json")
    elif token.startswith("INTERVENTION_POWER_"):
        paths.append(project_root / "output" / "data" / "intervention_power.json")
    elif token.startswith("QUOTIENT_CLASSIFIER_"):
        paths.append(project_root / "output" / "data" / "quotient_classifier.json")
    elif token.startswith("EXTERNAL_SOURCE_"):
        paths.append(
            project_root / "output" / "data" / "external_source_validation.json"
        )
    elif token.startswith("EXTERNAL_LABEL_"):
        paths.append(project_root / "output" / "data" / "external_label_contract.json")
    elif token.startswith("EXTERNAL_STATIC_"):
        paths.append(
            project_root / "output" / "data" / "external_static_validation.json"
        )
    elif token.startswith("EXTERNAL_CAPABILITIES_") or token.startswith(
        "EXTERNAL_CAPABILITY_"
    ):
        paths.append(project_root / "output" / "data" / "external_capabilities.json")
    elif token.startswith("INDEPENDENT_REPLAY_"):
        paths.append(project_root / "output" / "data" / "replication_manifest.json")
        # If a separately controlled operator supplies a receipt, its bytes
        # become part of the variable lineage. Absence remains the explicit
        # not_yet_observed state and does not block local variable generation.
        paths.append(
            project_root / "output" / "data" / "independent_replay_receipt.json"
        )
    elif token == "REPLICATION_PUBLIC_BUNDLE_SHA256":
        paths.append(project_root / "output" / "data" / "replication_manifest.json")
    elif token.startswith("RXINFER_CATEGORICAL"):
        paths.append(
            project_root
            / "output"
            / "rxinfer_categorical"
            / "categorical_diagnostics.json"
        )
    result: dict[str, str] = {}
    for path in paths:
        digest = _sha256(path)
        if digest:
            result[str(path.relative_to(project_root))] = digest
    return dict(sorted(result.items()))


def build_variable_registry(
    variables: dict[str, str], project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Build the machine-readable source/format/provenance registry."""

    consumers, counts = source_token_consumers(project_root)
    source_tokens = sorted(consumers)
    generated_tokens = sorted(variables)
    records: list[dict[str, Any]] = []
    for token in generated_tokens:
        rendered = str(variables[token])
        raw_type, raw_value = _raw_value(token, rendered)
        token_consumers = consumers.get(token, [])
        records.append(
            {
                "token": token,
                "semantic_definition": VARIABLE_DEFINITIONS.get(
                    token, f"Source-bound generated value for {token}."
                ),
                "canonical_source": _canonical_source(token),
                "producer": "src.manuscript_variables.generate_variables",
                "raw_type": raw_type,
                "raw_value": raw_value,
                "rendered_type": "string",
                "rendered_value": rendered,
                "unit": _unit_for(token, raw_type),
                "precision": _precision(rendered),
                "format_rule": "canonical producer-rendered manuscript string",
                "missing_value_policy": "fail_closed",
                "input_hashes": _input_hashes(project_root, token, token_consumers),
                "consumers": token_consumers,
            }
        )
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "target_repository": {
            "root": str(project_root),
            "revision": _git_value(project_root, "rev-parse", "HEAD"),
            "worktree_status": _git_value(project_root, "status", "--short"),
            "worktree_diff_sha256": _git_diff_sha256(project_root),
        },
        "template_hydration": _template_hydration(project_root),
        "source_tokens": source_tokens,
        "source_token_counts": dict(sorted(counts.items())),
        "generated_tokens": generated_tokens,
        "missing_tokens": sorted(set(source_tokens) - set(generated_tokens)),
        "unused_generated_tokens": sorted(set(generated_tokens) - set(source_tokens)),
        "duplicate_registry_tokens": [],
        "records": records,
        "summary": {
            "source_token_count": len(source_tokens),
            "generated_token_count": len(generated_tokens),
            "record_count": len(records),
            "all_source_tokens_generated": set(source_tokens).issubset(variables),
            "no_unused_generated_tokens": set(generated_tokens).issubset(source_tokens),
            "no_pending_values": not any(
                str(value).lower() == "pending" for value in variables.values()
            ),
        },
    }


def write_variable_registry(
    variables: dict[str, str], project_root: Path = PROJECT_ROOT
) -> Path:
    """Write and return the source-bound variable registry path."""

    path = project_root / MANUSCRIPT_VARIABLE_REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_variable_registry(variables, project_root)
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if isinstance(previous, dict):
            previous_generated_at = previous.get("generated_at")
            previous_without_time = dict(previous)
            previous_without_time.pop("generated_at", None)
            current_without_time = dict(payload)
            current_without_time.pop("generated_at", None)
            if previous_without_time == current_without_time and isinstance(
                previous_generated_at, str
            ):
                payload["generated_at"] = previous_generated_at
    write_json_artifact(
        path,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
    )
    return path
