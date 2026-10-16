"""Canonical source-bound figure specifications and offline figure atlas.

The project historically kept a caption dictionary and plotting functions in
``src.figures``.  This module is the format-neutral contract layered above
that compatibility API.  It binds every visual to its source artifacts and
hashes, declares the population and uncertainty semantics, and emits the
same metadata to the project registry, template registry, vector exports, and
the offline accessible atlas.

The contract is intentionally conservative.  A missing source receipt, stale
input hash, missing alternative text, or unparsable vector file is a hard
validation failure.  A visual can describe a blocked or underpowered result,
but it cannot turn that status into a numeric zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html import escape
import base64
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt

from .artifact_contract import write_json_artifact, write_json_atomic
from .figures import (
    FIGURE_CAPTIONS,
    FIGURE_CITATIONS,
    _caption_context,
    _figure_registry_rows,
    _png_qa_core,
    read_png_image,
    write_figure_captions,
)
from .visual_atlas import NARRATIVE_FIGURE_IDS, write_visual_atlas_figures

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_REGISTRY_SCHEMA_VERSION = "2.1"
FIGURE_REGISTRY_COMPATIBLE_SCHEMA_VERSIONS = ("2.0",)
FIGURE_REGISTRY_ARTIFACT = "output/data/figure_registry.json"

# Validator-summary artifacts are consumers of the figure registry rather than
# scientific inputs to the substantive figures.  Including them in the
# registry's aggregate input-hash projection creates a circular freshness
# dependency: validating the report rewrites the summary figure, which rewrites
# the registry, which then makes a rendered document appear stale.  Their own
# figure rows remain present and are validated independently; only the
# aggregate input projection omits these two validator-produced artifacts.
_VALIDATOR_PRODUCED_INPUTS = frozenset(
    {
        "output/data/reproducibility_manifest.json",
        "output/data/validation_report.json",
    }
)


@dataclass(frozen=True)
class FigureSpec:
    """Complete metadata for one substantive visual."""

    figure_id: str
    label: str
    section: str
    placement: str
    analytical_question: str
    title: str
    subtitle: str
    chart_family: str
    generator: str
    generator_revision: str
    input_artifacts: tuple[str, ...]
    input_hashes: dict[str, str]
    population: str
    sample_hierarchy: str
    split: str
    metrics: str
    uncertainty: str
    missing_semantics: str
    evidence_status: str
    receipt_status: str
    claim_ids: tuple[str, ...]
    validation_checks: tuple[str, ...]
    scholarship_keys: tuple[str, ...]
    caption: str
    alt_text: str
    long_description: str
    slide_caption: str
    png_path: str
    svg_path: str
    pdf_path: str
    atlas_path: str
    visual_qa: dict[str, Any]
    caption_word_count: int
    alt_text_word_count: int
    long_description_word_count: int
    palette_contract: str
    table_companion_path: str
    source_note: str
    visual_qa_status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe registry row."""

        value = asdict(self)
        for key in ("input_artifacts", "claim_ids", "scholarship_keys"):
            value[key] = list(value[key])
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_for(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if resolved.name == "output":
        return resolved.parent
    if (resolved / "output").is_dir():
        return resolved
    return PROJECT_ROOT


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_value(value: object, default: str = "not observed") -> str:
    if value is None:
        return default
    return str(value)


def _format_number(value: object, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "not observed"
    return f"{float(value):.{digits}f}"


def _hash_inputs(root: Path, artifacts: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for artifact in sorted(set(artifacts)):
        path = root / artifact
        if path.is_file():
            result[artifact] = _sha256(path)
    return result


def _base_artifacts(stem: str) -> tuple[str, ...]:
    """Return the source receipts for legacy figures by stable stem."""

    if stem == "assembly_cfg":
        return ("output/data/cfg_evidence.json",)
    if stem == "assembly_memory_timeline":
        return ("output/data/memory_evidence.json",)
    if stem in {"scholarship_source_roles", "scholarship_evidence_matrix"}:
        return ("output/data/scholarship_sources.json",)
    if stem == "claim_evidence_map":
        return ("output/data/claim_register.json",)
    if stem == "generative_adequacy":
        return (
            "output/data/generative_adequacy.json",
            "output/data/evaluation_report.json",
        )
    if stem == "typed_observation_matrix":
        return ("output/data/typed_observations.json",)
    if stem == "factorized_observation_comparison":
        return ("output/data/factorized_observation_model.json",)
    if stem == "corpus_coverage":
        return ("output/data/corpus_coverage.json",)
    if stem == "action_sensing":
        return (
            "output/data/action_sensing.json",
            "output/data/factorized_observation_model.json",
        )
    if stem == "parameter_learning":
        return ("output/data/parameter_learning.json",)
    if stem == "representation_invariance_benchmark":
        return ("output/data/representation_invariance_benchmark.json",)
    if stem == "intervention_power":
        return (
            "output/data/intervention_power.json",
            "output/data/representation_invariance_benchmark.json",
        )
    if stem == "cross_architecture_benchmark":
        return ("output/data/cross_architecture_fixtures.json",)
    if stem == "formal_verification_receipt":
        return ("output/data/formal_verification.json",)
    if stem == "active_information_frontier":
        return ("output/data/active_information_frontier.json",)
    if stem == "quotient_classifier":
        return (
            "output/data/quotient_classifier.json",
            "output/data/research_protocol.json",
        )
    if stem == "compiler_transfer":
        return (
            "output/data/compiler_transfer.json",
            "output/data/research_protocol.json",
        )
    if stem == "formal_correspondence":
        return (
            "output/data/formal_trace_correspondence.json",
            "output/data/research_protocol.json",
        )
    if stem == "sensor_costs":
        return (
            "output/data/sensor_cost_receipt.json",
            "output/data/research_protocol.json",
        )
    if stem == "replication_manifest":
        return (
            "output/data/replication_manifest.json",
            "output/replication/public/public_manifest.json",
            "output/replication/private/gold_labels.json",
        )
    if stem == "external_source_provenance":
        return (
            "output/data/external_source_validation.json",
            "configs/external_sources.yaml",
        )
    if stem == "external_label_coverage":
        return (
            "output/data/external_label_contract.json",
            "configs/external_label_contracts.yaml",
        )
    if stem == "artifact_receipt_closure":
        return (
            "output/data/research_protocol.json",
            "output/data/scientific_program.json",
        )
    if stem == "trace_refinement":
        return ("output/data/symbolic_execution.json",)
    if stem == "artifact_hash_manifest":
        return ("output/data/reproducibility_manifest.json",)
    if stem == "validation_summary":
        return ("output/data/validation_report.json",)
    artifacts = ["output/data/trace_bundle.json", "output/data/evaluation_report.json"]
    if "temporal" in stem or stem in {"segmentation_ribbons", "phase_recovery"}:
        artifacts.append("output/data/temporal_inference.json")
    if "active" in stem or stem == "ambiguity_ablation":
        artifacts.append("output/data/active_selection.json")
    return tuple(artifacts)


def _narrative_artifacts(stem: str) -> tuple[str, ...]:
    mapping = {
        "gnn_sensing": (
            "output/gnn/figure_inputs.json",
            "output/gnn/sensor_model.json",
            "output/gnn/controls.json",
        ),
        "research_architecture": ("output/data/research_protocol.json",),
        "assembly_state_plate": (
            "output/data/assembly_listing.json",
            "output/data/trace_bundle.json",
            "output/data/formal_trace_correspondence.json",
            "output/data/memory_evidence.json",
        ),
        "quotient_construction": (
            "output/data/quotient_invariance_protocol.json",
            "output/data/quotient_classifier.json",
            "output/data/research_protocol.json",
        ),
        "semantic_coverage_matrix": (
            "output/data/formal_trace_correspondence.json",
            "output/data/formal_verification.json",
            "output/data/research_protocol.json",
            "output/data/trace_bundle.json",
        ),
        "statistical_hierarchy": (
            "output/data/quotient_classifier.json",
            "output/data/research_protocol.json",
        ),
        "replay_failure_audit": (
            "output/data/independent_replay_receipt.json",
            "output/data/replication_manifest.json",
            "output/replication/public/public_manifest.json",
        ),
        "policy_certificate_frontier": (
            "output/data/sensor_cost_receipt.json",
            "output/data/active_information_frontier.json",
            "output/data/research_protocol.json",
        ),
        "theorem_status_map": (
            "output/data/theorem_or_countertheorem.json",
            "output/data/formal_verification.json",
            "output/data/intervention_power.json",
            "output/data/sensor_cost_receipt.json",
        ),
        "observation_semantics": (
            "output/data/typed_observations.json",
            "output/data/replication_manifest.json",
            "output/data/research_protocol.json",
        ),
    }
    return mapping[stem]


def _source_note(artifacts: Iterable[str], hashes: Mapping[str, str]) -> str:
    parts = [
        f"{artifact} ({hashes.get(artifact, 'missing')[:12]})"
        for artifact in sorted(set(artifacts))
    ]
    return "Source receipts: " + "; ".join(parts) + "."


def _dynamic_narrative_copy(output_dir: Path, stem: str) -> dict[str, str]:
    """Build source-bound visible and accessibility prose for a new figure."""

    data_dir = output_dir / "data"
    protocol = _load(data_dir / "research_protocol.json")
    formal = _load(data_dir / "formal_trace_correspondence.json")
    formal_verification = _load(data_dir / "formal_verification.json")
    quotient = _load(data_dir / "quotient_invariance_protocol.json")
    classifier = _load(data_dir / "quotient_classifier.json")
    sensor = _load(data_dir / "sensor_cost_receipt.json")
    frontier = _load(data_dir / "active_information_frontier.json")
    replay_path = data_dir / "independent_replay_receipt.json"
    replay = _load(replay_path) if replay_path.is_file() else {}
    theorem = _load(data_dir / "theorem_or_countertheorem.json")
    typed = _load(data_dir / "typed_observations.json")
    replication = _load(data_dir / "replication_manifest.json")
    protocol_hash = _safe_value(protocol.get("protocol_sha256"), "not observed")[:12]
    if stem == "gnn_sensing":
        inputs = _load(output_dir / "gnn/figure_inputs.json")
        model = _load(output_dir / "gnn/sensor_model.json")
        choices = model["actions"][1]
        policy = inputs["sensor_policies"]
        selected = choices[inputs["sensor_actions"][0][1]]
        caption = (
            "Prospective sensing on two transparent fixtures under the same fixed GNN model. "
            f"Panel A shows the first policy probabilities: {', '.join(f'{name} = {value:.3f}' for name, value in zip(choices, policy, strict=True))}. "
            f"The deterministic Agent selects {selected} before acquisition. Panel B shows "
            "the probability of the accumulator hypothesis before acquisition and after "
            "the first dynamic result, separately for the accumulator and bitwise fixtures. "
            "The two runs share the initial prior and first decision; only acquired evidence "
            "separates their subsequent beliefs. Lines connect two recorded belief states, "
            "not a continuous learning trajectory. These probabilities condition on authored "
            "likelihoods and illustrative dimensionless utilities; they are not uncertainty "
            "intervals, discovered semantics, or an external performance estimate "
            "[@friedman2026gnn; @heins2022pymdp]."
        )
        alt = "Three first-decision policy probabilities and two fixture-conditioned posterior paths sharing the same prior, then separating after dynamic acquisition."
        long = (
            caption
            + " The horizontal stages in panel B are decision-time posteriors: the first precedes any fixture acquisition, while the second conditions on the preceding dynamic result. The second action does not supply evidence to its own preceding belief. Evaluation fixture names label plotted runs only after decisions. Panel A reports the Agent policy distribution, not repeated-action frequencies; deterministic selection chooses its highest-probability action. The complete source and tensor digests remain in the linked study receipts."
        )
    elif stem == "research_architecture":
        caption = (
            f"The research architecture maps the protocol {protocol.get('protocol_id', 'not observed')} "
            f"from source bytes through Capstone static decoding, normalized instruction representation, "
            "Python reference semantics, bounded Unicorn execution, the Lean formal boundary, typed "
            "observations, finite interventions and quotient labels, the co-primary model portfolio, "
            "sensing and cost evaluation, receipt/claim registration, and template hydration. Static, "
            "dynamic, formal, inferential, and document-production regions are outlined separately; "
            f"the protocol hash prefix is {protocol_hash}. Edge annotations identify whether a relation "
            "is kernel-checked, computationally cross-checked, inferential, or provenance-preserving. "
            "The visual is an evidence map for this bounded technical monograph. It does not make "
            "the pipeline transitive proof, complete x86-64 semantics, universal identifiability, or "
            "application validity."
        )
        alt = "Evidence map from assembly bytes through decoding, bounded semantics, inference, receipts, and template hydration, with proof and claim boundaries shown explicitly."
        long = (
            caption
            + " The source node is the repository-owned transparent fixture surface. Capstone contributes static decoding; it does not execute code. The Python interpreter supplies the reference transition relation, while the Unicorn path is a bounded cross-check on transparent fixtures. Lean covers only the declared mechanized fragment. Typed observations retain missing and unseen values as different states before model fitting. Quotient labels are generated from a finite intervention contract, not inferred by a classifier. The final boxes represent audit and document surfaces whose hashes connect back to producer inputs. The map therefore supports provenance and reproducibility reasoning, while every scientific interpretation remains scoped to the associated receipt status."
        )
    elif stem == "observation_semantics":
        typed_fields = typed.get("fields", [])
        # The visual producer receives the replication manifest as its
        # canonical public/protected boundary.  The controlled-replay receipt
        # records the outcome of a run, but it is not the schema authority and
        # may intentionally omit the public-field vocabulary.  Reading the
        # manifest here keeps the generated caption numerically aligned with
        # the figure and its source table.
        input_contract = replication.get("input_contract", {})
        public_fields = (
            input_contract.get("public_fields", [])
            if isinstance(input_contract, Mapping)
            else []
        )
        forbidden_fields = (
            input_contract.get("forbidden_public_fields", [])
            if isinstance(input_contract, Mapping)
            else []
        )
        typed_field_count = (
            len(typed_fields) if isinstance(typed_fields, list) else "not observed"
        )
        public_field_count = (
            len(public_fields) if isinstance(public_fields, list) else "not observed"
        )
        forbidden_field_count = (
            len(forbidden_fields)
            if isinstance(forbidden_fields, list)
            else "not observed"
        )
        caption = (
            f"The observation-semantics diagram distinguishes the typed schema's {typed_field_count} fields, "
            f"the dynamic missing token {typed.get('missing_token', '__MISSING__')}, and the unseen token "
            f"{typed.get('unseen_token', '__UNKNOWN__')} from a measured numeric zero. It also keeps unsupported "
            "instruction effects and unavailable compiler or target capabilities as explicit statuses rather than "
            f"numeric values. The public replay surface exposes {public_field_count} label-blind fields, while "
            f"{forbidden_field_count} forbidden fields remain behind the protected scoring boundary. This is a "
            "schema and provenance visual: it documents how absence and protection are represented before inference, "
            f"not how well a model performs. The protocol is {protocol.get('protocol_id', 'not observed')} with hash "
            f"prefix {protocol_hash}. It does not establish complete semantic coverage, recoverability of protected "
            "labels, or universal identifiability."
        )
        alt = "Typed observation and replay-boundary diagram distinguishing observed values, missing fields, unseen values, unsupported states, unavailable capabilities, and measured zero."
        long = (
            caption
            + " The left side shows six different states that are often collapsed by convenience encodings. Observed values are ordinary typed fields; missing means that a field was not produced by the observation action; unseen means that a categorical value was outside the observed vocabulary; unsupported means that the operation lies outside the declared semantic contract; unavailable means that a requested capability was not present; and a measured zero is a real receipt value. The central record is label-blind and retains status-bearing fields. On the right, public rows and predictions are separated from protected gold labels, while source-bound protocol and hash receipts provide provenance. The exact table below the visual records the field and boundary vocabulary. This figure is descriptive and computationally constructed from receipts; it supplies no inferential interval and does not transform replay separation into independent replication."
        )
    elif stem == "assembly_state_plate":
        listing = _load(data_dir / "assembly_listing.json")
        memory = _load(data_dir / "memory_evidence.json")
        steps = formal.get("step_rows", [])
        caption = (
            f"The assembly-state plate shows {len(listing.get('instructions', []))} decoded instructions and "
            f"{len(steps)} receipt step rows from the transparent {listing.get('sample_id', 'fixture')} fixture. "
            "The listing panel reports addresses, bytes, decoded operands, operand widths, flag reads and "
            "writes, and next program counters. The state panel shows bounded register and RIP transitions; "
            f"the memory panel contains {len(memory.get('dynamic_memory_events', []))} dynamic hook events. "
            "The boundary panel keeps unsupported instructions, undefined flags, exceptions, and out-of-bounds "
            "memory explicit. Values are exact fixture observations under the declared interventions, with no "
            "inferential confidence interval. The plate documents a normalized state contract and a transparent "
            "assembly listing; it does not establish complete x86-64 semantics, arbitrary alias analysis, ABI "
            "behavior, or execution safety for external binaries."
        )
        alt = "Decoded x86-64 listing beside bounded register, program-counter, flag, and memory transitions, with a visibly hatched unsupported-state boundary."
        long = (
            caption
            + " The listing is not copied from prose: it is read from the assembly-listing receipt. Register values are shown before and after each recorded instruction, and the flag column is tied to the normalized instruction record. Memory events are reported only when the transparent hook observed them inside the declared bounded region. A blank or non-applicable field is represented with an em dash or boundary status rather than an invented zero. The panel makes the distinction between static evidence and dynamic evidence visible to a reader inspecting the figure at page scale."
        )
    elif stem == "quotient_construction":
        pairs = quotient.get("pairs", [])
        intervention_values: set[int] = set()
        for row in pairs:
            if not isinstance(row, Mapping):
                continue
            values = row.get("initial_state_interventions", [])
            if isinstance(values, list):
                intervention_values.update(
                    value for value in values if isinstance(value, int)
                )
        interventions = sorted(intervention_values)
        caption = (
            f"The finite quotient construction starts with {len(pairs)} source-bound pair rows, applies "
            f"{len(interventions)} declared initial-state intervention values, compares normalized traces "
            "and the recorded output relation, and assigns the resulting benchmark label before fitting. "
            f"The quotient classifier receipt contains {classifier.get('family_count', 'not observed')} source "
            f"families and {classifier.get('held_out_family_count', 'not observed')} held-out families, while "
            "equivalent transformations, one-change near misses, and non-separating controls remain distinct "
            "roles. The table is a provenance view of label construction, not a learned semantic taxonomy. "
            "Equality means agreement under the declared intervention, observation, and endpoint contract; it "
            "does not mean unrestricted program equivalence or universal identifiability."
        )
        alt = "Flow diagram from source-bound pair roles through interventions and normalized traces to finite quotient labels and held-out family splits."
        long = (
            caption
            + " Pair rows are nested within their source families, so the visual does not imply that rows are independent samples. The intervention stage is deliberately finite and auditable. The normalized trace stage is where static instruction records and bounded dynamic outcomes are compared; it is not a claim that the observation channel contains every architectural effect. The split stage protects the evaluation from source-family leakage. Negative controls are shown as controls, not erased when they remain unresolved."
        )
    elif stem == "semantic_coverage_matrix":
        supported_count = len(formal.get("supported_instruction_subset", []))
        lean_count = len(formal_verification.get("theorems", []))
        observed_count = len(formal.get("step_rows", []))
        caption = (
            f"The semantic coverage matrix records {supported_count} declared instruction names, {lean_count} "
            f"kernel-checked theorem entries, and {observed_count} formal correspondence step rows across "
            "operation families and evidence layers. Columns distinguish declaration, operand-width handling, "
            "flag effects, bounded memory, control flow, Lean coverage, Capstone availability, and Unicorn "
            "cross-checks. Check marks therefore mean a declared or observed capability at the stated layer; "
            "unresolved, not-observed, and unsupported cells are written as text and hatching rather than zero. "
            "The matrix is a scope receipt for a user-mode integer subset and its current transparent fixtures. "
            "It is not a complete x86-64 ISA table, a proof of all adapter behavior, or a performance result."
        )
        alt = "Categorical coverage matrix for x86-64 operation families across widths, flags, memory, control flow, Lean, Capstone, and Unicorn."
        long = (
            caption
            + " The operation-family rows aggregate related instruction names only for visual navigation; the underlying supported subset remains in the formal receipt. Lean coverage is separated from Python reference and adapter coverage, because a theorem about a normalized fragment is not automatically a theorem about a decoder or emulator. The matrix also preserves the concept of not central and not observed, which prevents readers from confusing an absent test with a failed measurement."
        )
    elif stem == "statistical_hierarchy":
        stats = protocol.get("statistics_contract", {})
        caption = (
            f"The statistical hierarchy contains {classifier.get('family_count', 'not observed')} transformation-family identifiers, "
            f"{classifier.get('row_count', 'not observed')} pair rows, and {len(classifier.get('interventions', []))} "
            f"classifier interventions in the current receipt. Family identifiers group transformation pairs; "
            "pairs are evaluated under interventions, producing traces with dependent instruction steps. "
            "Identifiers are disjoint across splits, but authored template lineages are shared. The resampling unit "
            "is the held-out family identifier, not an independently sampled source lineage. The declared "
            f"cluster-bootstrap contract uses a {stats.get('confidence_level', 'not observed')} confidence level. "
            "Training, development, and test families are fixed before fitting, with selection and stopping "
            "restricted to development data. The visual explains the dependence structure and interval meaning; "
            "it does not license instruction-step pseudoreplication or turn row counts into independent evidence."
        )
        alt = "Authored lineages, split-disjoint family identifiers, pairs and dependent steps; family resampling and shared-lineage limitations are explicit."
        long = (
            caption
            + " The chain describes the generated inventory, not independent draws from a population. No revision or instruction count is invented when that denominator is absent. Family-resampling intervals retain rows within each sampled identifier, but they do not account for dependence between different identifiers with a shared authored template. Conditional randomization separately assumes exchangeability at the swapping unit. The test identifiers are not consulted for model tuning; that selection boundary does not create lineage-disjoint evaluation or prospective registration."
        )
    elif stem == "replay_failure_audit":
        external = replay.get("external_receipt", {})
        score = replay.get("protected_score", {})
        score_observed = any(
            isinstance(score.get(key), (int, float))
            and not isinstance(score.get(key), bool)
            for key in (
                "balanced_accuracy",
                "macro_f1",
                "brier_score",
                "expected_calibration_error",
            )
        )
        if score_observed:
            caption = (
                f"The controlled replay audit follows {score.get('rows', 'not observed')} label-blind rows through a "
                f"{external.get('control_scope', 'not observed')} with network disabled and protected gold-label scoring. "
                f"The protected result is balanced accuracy {_format_number(score.get('balanced_accuracy'))}, macro-F1 "
                f"{_format_number(score.get('macro_f1'))}, Brier {_format_number(score.get('brier_score'))}, and ECE "
                f"{_format_number(score.get('expected_calibration_error'))}; the confusion panel shows the model's "
                "equivalence prediction for both classes. The reliability and selective-risk panels are single-run "
                "audit diagnostics, not population estimates. The receipt confirms a separately controlled guest run, "
                "not an independent physical-machine replication or independent reimplementation. The poor result is "
                "retained as a reproducibility and failure-boundary result rather than replaced by a favorable rerun."
            )
            alt = "Public/private replay boundary and protected confusion, calibration, and selective-risk panels showing chance discrimination in the controlled guest run."
            long = (
                caption
                + " Public rows cross the guest boundary without labels, predictions return with stable identifiers, and scoring occurs only at the protected owner-side boundary. The flow does not imply that the guest saw the gold labels. The negative confusion pattern is the central result: all positive rows are accepted but near misses are not separated. Calibration metrics are shown with their actual direction and units. Because there is one controlled replay receipt, no independent-sample uncertainty is attached to the result."
            )
        else:
            caption = (
                "The controlled replay audit is defined over the frozen public replay bundle: label-blind rows "
                "cross a no-network guest boundary, predictions return through identifier validation, and "
                "protected gold-label scoring remains on the owner side. No independently replayed receipt has "
                "been ingested; protected counts, balanced accuracy, macro-F1, Brier, and ECE are not observed. "
                "The confusion, reliability, and selective-risk panels therefore carry explicit not-observed "
                "statuses rather than measured values, and no substitute score, rerun, or expected value is "
                "imputed. When a receipt is ingested, this caption will report the protected result and retain "
                "it whether favorable or poor; a poor result is kept as a reproducibility and failure-boundary "
                "record rather than replaced by a favorable rerun. The audited boundary is a same-host "
                "controlled guest run, not an independent physical-machine replication or independent "
                "reimplementation."
            )
            alt = (
                "Public/private replay boundary diagram with confusion, reliability, and selective-risk "
                "panels marked not observed; no independently replayed receipt has been ingested."
            )
            long = (
                caption
                + " Public rows cross the guest boundary without labels, predictions return with stable identifiers, and scoring occurs only at the protected owner-side boundary. The flow does not imply that the guest saw the gold labels. Until a receipt is ingested, every protected quantity remains an explicit not-observed status and the pending verification state is preserved rather than replaced by a placeholder value."
            )
    elif stem == "policy_certificate_frontier":
        frontier_rows = frontier.get("frontier", [])
        certs = sensor.get("policy_certificates", [])
        horizon_values: set[int] = set()
        for row in frontier_rows:
            if not isinstance(row, Mapping):
                continue
            value = row.get("horizon")
            if isinstance(value, int):
                horizon_values.add(value)
        horizons = sorted(horizon_values)
        caption = (
            f"The policy frontier contains {len(frontier_rows)} exact dynamic-program rows across horizons "
            f"{', '.join(str(value) for value in horizons) or 'not observed'} and {len(certs)} finite certificate rows. "
            "The left panel plots expected information gain in nats against deterministic count cost; the right "
            "panel summarizes optimality gaps for implemented and comparison policies when present. Exact finite "
            "enumeration has no sampling confidence interval. Optional wall-clock values are local calibration "
            "measurements and remain separate from deterministic counts. The exact oracle is an inadmissible "
            "reference used for regret and gap calculations. Any tolerance certificate applies only to the frozen "
            "sensor family, state space, and horizon, not to arbitrary binaries, hardware, or a universal sensing law."
        )
        alt = "Finite information-versus-cost frontier and policy optimality-gap bars, separating exact enumeration from local timing calibration."
        long = (
            caption
            + " The horizontal axis uses receipt count units rather than nanoseconds, so it remains reproducible across machines. When calibrated timing exists, it is attached to the policy receipt with trial count and spread, but it is not substituted into the exact dynamic program. The figure retains comparison policies even when they are not certificate winners. A zero expected cost at horizon zero is a meaningful exact baseline; an unavailable timing or unsupported action would instead be labelled, not imputed."
        )
    else:
        candidates = theorem.get("candidate_theorems", [])
        status_counts: dict[str, int] = {}
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                value = str(candidate.get("status", "unresolved"))
                status_counts[value] = status_counts.get(value, 0) + 1
        caption = (
            f"The theorem-status map contains {len(candidates)} candidate results: "
            f"{status_counts.get('proved', 0)} proved, {status_counts.get('computationally_validated', 0)} computationally validated, "
            f"{status_counts.get('empirically_supported', 0)} empirically supported, {status_counts.get('falsified_by_counterexample', 0)} "
            f"falsified by counterexample, and {status_counts.get('unresolved', 0)} unresolved. Each row retains its assumptions, state and "
            "memory bounds, supported instruction scope, observation/intervention contract, and proof or counterexample artifact. "
            "Kernel-checked Lean results are separated from Python/Capstone/Unicorn computational checks, while falsified and open "
            "candidates remain visible. The figure therefore controls claim promotion: a theorem-shaped file is not itself a theorem, "
            "and a bounded computational result is not a broad mathematical guarantee."
        )
        alt = "Status table separating proved, computationally validated, empirical, falsified, and unresolved formal candidates with their scopes."
        long = (
            caption
            + " The status colors are reinforced by text labels and table rows so the interpretation does not depend on hue. The shared assumptions are listed below the table. A falsified candidate contributes a preserved bounded counterexample; it does not prove the unrestricted negation. An unresolved row identifies work that remains open rather than a missing value."
        )
    caption_suffix = (
        " Values are reported at the receipt's stated grain. Counts describe this "
        "package and its frozen contract, not a population beyond the recorded "
        "artifacts. Where evidence is unavailable, unsupported, excluded, or not "
        "observed, the registry preserves that status as text. Readers should "
        "interpret the visual together with its linked source hashes and claim "
        "boundary."
    )
    long_suffix = (
        " The registry carries the source hashes, evidence status, and claim IDs "
        "alongside the rendered visual so that a caption cannot silently drift "
        "from its producer."
    )
    long += caption_suffix + long_suffix
    return {"caption": caption, "alt_text": alt, "long_description": long}


def _metadata_for(stem: str) -> tuple[str, str, str, str, str, str, str]:
    """Return section, placement, question, title, subtitle, chart family, status."""

    rows: dict[str, tuple[str, str, str, str, str, str, str]] = {
        "gnn_sensing": (
            "Results",
            "main",
            "Does acquired evidence follow policy selection and separate fixture beliefs?",
            "Prospective GNN sensing",
            "Fixed authored models; action precedes acquisition.",
            "policy bars and posterior trajectories",
            "computational",
        ),
        "research_architecture": (
            "Introduction",
            "main",
            "How are evidence objects connected without collapsing their proof roles?",
            "Research architecture and evidence boundaries",
            "Protocol map; static, dynamic, formal, inferential, and document evidence remain distinct.",
            "evidence architecture",
            "descriptive",
        ),
        "assembly_state_plate": (
            "Methodology",
            "main",
            "What state and instruction fields are actually modeled and cross-checked?",
            "Assembly state plate: listing → state transition → bounded effects",
            "Transparent fixture evidence; widths, flags, registers, RIP, and memory effects are receipt-backed.",
            "annotated formal plate",
            "formal",
        ),
        "quotient_construction": (
            "Methodology",
            "main",
            "How are finite quotient labels constructed before model fitting?",
            "Finite quotient construction and source-family holdout",
            "Source roles, interventions, normalized traces, and family splits are explicit.",
            "provenance flow diagram",
            "computational",
        ),
        "semantic_coverage_matrix": (
            "Methodology",
            "appendix",
            "Which semantic families are declared, checked, or outside the contract?",
            "Declared semantic coverage by evidence layer",
            "Categorical scope receipt; unsupported and unobserved cells are not numeric failures.",
            "coverage matrix",
            "formal",
        ),
        "statistical_hierarchy": (
            "Methodology",
            "main",
            "What is the independent sampling unit and where can leakage occur?",
            "Statistical hierarchy and leakage control",
            "Nested source-bound evaluation; family-clustered uncertainty and fixed splits.",
            "hierarchy diagram",
            "inferential",
        ),
        "replay_failure_audit": (
            "Results",
            "main",
            "What did the separately controlled replay reproduce, including failure?",
            "Controlled replay failure audit",
            "Same-host separately controlled guest; protected score and negative result retained.",
            "boundary diagram plus audit panels",
            "descriptive",
        ),
        "policy_certificate_frontier": (
            "Results",
            "main",
            "How do finite policy information and auditable cost trade off?",
            "Finite sensing policy certificates and cost frontier",
            "Exact finite enumeration and optional local calibration are separate evidence layers.",
            "frontier and gap plot",
            "computational",
        ),
        "theorem_status_map": (
            "Discussion",
            "main",
            "Which formal candidates are actually proved, falsified, or unresolved?",
            "Theorem-or-countertheorem status map",
            "Formal status is generated from checked artifacts and preserved counterexamples.",
            "status map",
            "formal",
        ),
        "observation_semantics": (
            "Methodology",
            "main",
            "How are observed, absent, unsupported, unavailable, and protected values kept distinct?",
            "Observation semantics: values, absence, and protection",
            "Typed observations preserve missing versus unseen values and keep unavailable evidence distinct from measured zero.",
            "schema and evidence-boundary diagram",
            "descriptive",
        ),
    }
    return rows[stem]


# Verified manuscript placements for legacy receipt figures. Rows outside this
# map keep the long-standing heuristic placement in ``build_figure_specs``.
_LEGACY_SECTION_OVERRIDES: dict[str, str] = {
    "rxinfer_free_energy": "Results",
    "posterior_beliefs": "Results",
    "inference_diagnostics": "Results",
    "parameter_learning": "Results",
    "calibration_curve": "Results",
    "decoder_calibration": "Results",
    "confusion_matrix": "Results",
    "model_evidence": "Results",
    "ambiguity_ablation": "Results",
    "segmentation_ribbons": "Results",
    "transition_ablation": "Results",
    "observation_informativeness": "Results",
    "phase_recovery": "Results",
    "corpus_stability": "Results",
    "active_efe": "Results",
    "action_sensing": "Results",
    "counterexample_search": "Results",
    "sensing_stress": "Results",
    "generative_adequacy": "Results",
    "quotient_classifier": "Methodology",
    "compiler_transfer": "Methodology",
    "formal_correspondence": "Methodology",
    "sensor_costs": "Methodology",
    "replication_manifest": "Methodology",
    "artifact_receipt_closure": "Validation",
}

# Evidence-status overrides for legacy rows whose receipts carry resampled or
# interval-bearing uncertainty; the legacy default remains "descriptive".
_EVIDENCE_STATUS_OVERRIDES: dict[str, str] = {
    "quotient_classifier": "inferential",
    "model_comparison": "inferential",
    "calibration_curve": "inferential",
    # Family-level descriptive 95% Wilson whiskers are plotted in the
    # representation-invariance pilot (src.figures), so it carries intervals.
    "representation_invariance_benchmark": "inferential",
}

# Concrete uncertainty descriptions mirroring each caption's interval type.
_UNCERTAINTY_OVERRIDES: dict[str, str] = {
    "calibration_curve": (
        "95% Wilson intervals around per-bin hit counts (descriptive, fixture corpus)"
    ),
    "generative_adequacy": (
        "central 95% simulation intervals from the fixed-parameter HMM "
        "(descriptive model checking)"
    ),
    "model_comparison": (
        "descriptive Wilson intervals around pooled hit counts at the declared "
        "confidence level"
    ),
}


def _visual_qa(path: Path, vector_dir: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "missing",
            "bytes": 0,
            "sha256": None,
            "pixel_variance": 0.0,
            "vector_svg": False,
            "vector_pdf": False,
        }
    core = _png_qa_core(path)
    svg_path = vector_dir / f"{path.stem}.svg"
    pdf_path = vector_dir / f"{path.stem}.pdf"
    svg_ok = svg_path.is_file() and b"<svg" in svg_path.read_bytes()[:2048].lower()
    pdf_ok = pdf_path.is_file() and pdf_path.read_bytes()[:4] == b"%PDF"
    return {
        **core,
        "vector_svg": svg_ok,
        "vector_pdf": pdf_ok,
    }


def build_figure_specs(output_dir: Path) -> list[FigureSpec]:
    """Build all canonical specifications from current source artifacts."""

    root = _root_for(output_dir)
    output_dir = root / "output"
    context = _caption_context(output_dir)
    legacy_rows = {
        row["figure_id"]: row for row in _figure_registry_rows(output_dir, context)
    }
    specs: list[FigureSpec] = []
    for stem in FIGURE_CAPTIONS:
        claims: tuple[str, ...]
        checks: tuple[str, ...]
        if stem in NARRATIVE_FIGURE_IDS:
            (
                section,
                placement,
                question,
                title,
                subtitle,
                chart_family,
                evidence_status,
            ) = _metadata_for(stem)
            artifacts = tuple(
                artifact
                for artifact in _narrative_artifacts(stem)
                if (root / artifact).is_file()
                or artifact != "output/data/independent_replay_receipt.json"
            )
            copy = _dynamic_narrative_copy(output_dir, stem)
            generator = "src.visual_atlas"
            claims = ("CLM-009",)
            checks = (
                f"figure_{stem}",
                "figure_registry_contract",
                "figure_accessibility_contract",
            )
            citation_keys = tuple(FIGURE_CITATIONS.get(stem, ()))
            primary_artifact = root / artifacts[0]
            receipt_status = (
                _safe_value(_load(primary_artifact).get("status"))
                if primary_artifact.is_file()
                else "not observed"
            )
            population = "source-bound protocol and generated receipts for the declared narrative evidence layer"
            hierarchy = "artifact-level visual; nested sample hierarchy is reported in the caption and source receipt"
            split = (
                "not applicable to architecture/status plate"
                if stem
                in {
                    "research_architecture",
                    "assembly_state_plate",
                    "semantic_coverage_matrix",
                    "theorem_status_map",
                }
                else "source-bound split as recorded in receipt"
            )
            metrics = "counts, statuses, or receipt-backed metrics shown in the visual"
            uncertainty = "exact/status visual; inferential intervals are not added unless the source receipt supplies them"
            if stem == "gnn_sensing":
                population = (
                    "two transparent owned fixtures under the same GNN sensor POMDP"
                )
                hierarchy = "model → fixture run → decision → selected acquisition → next posterior"
                split = "not a fitted benchmark; fixture identity withheld from policy selection"
                metrics = "policy and posterior probabilities; recorded sensor actions"
                uncertainty = "conditional probabilities under fixed authored tensors; no sampling interval"
            elif stem == "statistical_hierarchy":
                population = "frozen quotient family identifiers, pair rows, interventions, traces, and steps"
                hierarchy = "shared authored template lineage → family identifier → pair → intervention → trace → instruction step"
                split = "disjoint family identifiers with shared authored lineages; development-only selection"
                metrics = "family, row, intervention, and step counts"
                uncertainty = "conditional family-bootstrap percentiles; shared-lineage dependence is not resolved"
            elif stem == "replay_failure_audit":
                population = "one protected score over the frozen public replay bundle"
                hierarchy = "bundle → public row → prediction → protected score"
                split = "public label-blind input; protected owner-side gold labels"
                metrics = "balanced accuracy, macro-F1, Brier, ECE, confusion counts, risk/coverage"
                uncertainty = "single controlled replay; no population interval claimed"
            elif stem == "policy_certificate_frontier":
                population = "finite sensor/action/horizon space in the sensing and information receipts"
                hierarchy = "finite state/action/horizon enumeration → policy evaluation → certificate"
                split = "finite contract; no train/test split"
                metrics = (
                    "nats, count cost, calibrated nanoseconds, regret, optimality gap"
                )
                uncertainty = "exact enumeration has no sampling interval; timing has local calibration spread"
            elif stem == "quotient_construction":
                population = "source-bound transformation pairs and declared intervention catalog"
                hierarchy = "family → pair role → intervention → normalized trace → endpoint relation"
                split = "source-family-disjoint training/development/test allocation"
                metrics = "pair and intervention counts; endpoint equality/separation"
                uncertainty = "descriptive receipt counts; model uncertainty remains in classifier receipt"
            elif stem == "semantic_coverage_matrix":
                population = "declared normalized x86-64 operation-family subset and observed formal fixtures"
                hierarchy = "operation family → evidence layer → status cell"
                split = "not applicable; coverage contract"
                metrics = (
                    "categorical declared/checked/unsupported/not-observed statuses"
                )
                uncertainty = "not applicable to deterministic coverage status"
            elif stem == "assembly_state_plate":
                population = "one transparent x86-64 fixture, bounded state transitions, and memory-hook rows"
                hierarchy = "fixture → instruction → state transition / memory effect"
                split = "not applicable; transparent fixture audit"
                metrics = (
                    "addresses, bytes, widths, flags, registers, RIP, memory effects"
                )
                uncertainty = "not applicable to exact fixture rows"
            elif stem == "research_architecture":
                population = (
                    "protocol-declared evidence and document-production components"
                )
                hierarchy = (
                    "source → semantics → observation → inference → receipt → document"
                )
                split = "not applicable; architecture map"
                metrics = "boundary roles and protocol status"
                uncertainty = "not applicable to architecture map"
            elif stem == "theorem_status_map":
                population = "candidate theorem and countertheorem rows in theorem_or_countertheorem.json"
                hierarchy = "candidate → assumptions → proof/counterexample artifact → allowed claim"
                split = "not applicable; theorem-status ledger"
                metrics = "status counts and candidate scope"
                uncertainty = "not applicable to status taxonomy"
            elif stem == "observation_semantics":
                population = "typed-observation schema, public replay interface, and protected receipt boundary"
                hierarchy = "field vocabulary → typed record → public/protected interface → source receipt"
                split = "label-blind public fields separated from protected labels"
                metrics = "field, status, and protected-boundary counts"
                uncertainty = "not applicable to a descriptive schema visual"
            missing_semantics = "Unavailable, unsupported, excluded, underpowered, not observed, and unresolved values remain textual statuses; no zero substitution."
            long_description = copy["long_description"]
            alt_text = copy["alt_text"]
            caption = copy["caption"]
        else:
            row = legacy_rows[stem]
            generator = "src.figures"
            section = _LEGACY_SECTION_OVERRIDES.get(stem) or (
                "Methodology"
                if stem
                not in {
                    "model_comparison",
                    "quotient_classifier",
                    "compiler_transfer",
                    "formal_correspondence",
                    "sensor_costs",
                    "replication_manifest",
                    "external_source_provenance",
                    "external_label_coverage",
                    "artifact_receipt_closure",
                }
                else "Results"
            )
            placement = (
                "appendix"
                if stem
                in {
                    "scholarship_source_roles",
                    "scholarship_evidence_matrix",
                    "claim_evidence_map",
                    "artifact_hash_manifest",
                    "validation_summary",
                }
                else "main"
            )
            question = (
                f"What does the source-bound receipt for {stem.replace('_', ' ')} show?"
            )
            title = stem.replace("_", " ").title()
            subtitle = f"Receipt-backed {stem.replace('_', ' ')} visual; see the generated caption for population and evidence limits."
            chart_family = "diagnostic or audit visual"
            evidence_status = _EVIDENCE_STATUS_OVERRIDES.get(stem, "descriptive")
            artifacts = _base_artifacts(stem)
            caption = str(row["caption"])
            accessible_context = caption
            if len(accessible_context.split()) < 120:
                accessible_context += (
                    " The visual is generated from the named receipt and is intended to expose "
                    "the measurement contract, not to enlarge the scientific population beyond "
                    "the recorded fixture or split. Its marks, axes, and annotations retain the "
                    "declared observation grain and units; deterministic counts are not population "
                    "uncertainty, and an interval is shown only when the producer supplies one. "
                    "Missing, unseen, unsupported, unavailable, excluded, underpowered, and "
                    "unresolved cases remain status-bearing values rather than invented zeros. "
                    "The visual therefore supports the bounded claim in its receipt, not a stronger "
                    "semantic, deployment, or generalization claim."
                )
            alt_text = str(row["alt_text"])
            long_description = (
                accessible_context
                + " The source-bound registry records the exact input paths and hashes, the population, the sampling unit, and the interpretation boundary. Missing, unseen, unsupported, unavailable, excluded, and underpowered states remain explicit. A diagnostic visual can document a method or failure without establishing a population-level performance claim."
            )
            claims = tuple(str(value) for value in row.get("claim_ids", ("CLM-009",)))
            checks = tuple(
                str(value)
                for value in row.get("validation_checks", (f"figure_{stem}",))
            )
            citation_keys = tuple(str(value) for value in row.get("citation_keys", ()))
            receipt_status = "implemented_local"
            population = str(
                row.get(
                    "analysis_population", "source-bound transparent fixture corpus"
                )
            )
            hierarchy = "as declared by the source receipt; instruction steps are not independent unless explicitly stated"
            split = "as declared by the source receipt; unavailable when not applicable"
            metrics = str(row.get("statistics", "descriptive receipt-backed values"))
            uncertainty = _UNCERTAINTY_OVERRIDES.get(
                stem,
                "as stated by the source receipt; deterministic visuals do not "
                "receive invented intervals",
            )
            missing_semantics = "Unavailable and unsupported cells remain labelled; no unavailable evidence is plotted as zero."
        png_path = root / "output" / "figures" / f"{stem}.png"
        svg_path = root / "output" / "figures" / "vector" / f"{stem}.svg"
        pdf_path = root / "output" / "figures" / "vector" / f"{stem}.pdf"
        artifacts = tuple(sorted(set(artifacts)))
        hashes = _hash_inputs(root, artifacts)
        generator_path = root / (
            "src/visual_atlas.py" if stem in NARRATIVE_FIGURE_IDS else "src/figures.py"
        )
        generator_revision = (
            _sha256(generator_path) if generator_path.is_file() else "missing"
        )
        visual_qa = _visual_qa(png_path, svg_path.parent)
        source_note = _source_note(artifacts, hashes)
        caption_word_count = len(caption.split())
        alt_text_word_count = len(alt_text.split())
        long_description_word_count = len(long_description.split())
        table_companion_path = (
            _relative(root / "output" / "figure_atlas" / "index.html", root)
            + f"#{stem}-data-table"
        )
        palette_contract = "Shared high-contrast palette with redundant status encodings: text labels, hatching, marker/line distinctions, and grayscale-readable keylines; no green/red dependence."
        slide_caption = " ".join(caption.split())
        if len(slide_caption.split()) > 40:
            slide_caption = " ".join(slide_caption.split()[:39]) + " …"
        specs.append(
            FigureSpec(
                figure_id=stem,
                label=f"fig:{stem.replace('_', '-')}",
                section=section,
                placement=placement,
                analytical_question=question,
                title=title,
                subtitle=subtitle,
                chart_family=chart_family,
                generator=generator,
                generator_revision=generator_revision,
                input_artifacts=artifacts,
                input_hashes=hashes,
                population=population,
                sample_hierarchy=hierarchy,
                split=split,
                metrics=metrics,
                uncertainty=uncertainty,
                missing_semantics=missing_semantics,
                evidence_status=evidence_status,
                receipt_status=receipt_status,
                claim_ids=claims,
                validation_checks=checks,
                scholarship_keys=citation_keys,
                caption=caption,
                alt_text=alt_text,
                long_description=long_description,
                slide_caption=slide_caption,
                png_path=_relative(png_path, root),
                svg_path=_relative(svg_path, root),
                pdf_path=_relative(pdf_path, root),
                atlas_path=_relative(
                    root / "output" / "figure_atlas" / "index.html", root
                ),
                visual_qa=visual_qa,
                caption_word_count=caption_word_count,
                alt_text_word_count=alt_text_word_count,
                long_description_word_count=long_description_word_count,
                palette_contract=palette_contract,
                table_companion_path=table_companion_path,
                source_note=source_note,
                visual_qa_status=str(visual_qa.get("status", "not observed")),
            )
        )
    return specs


def _write_raster_svg(png_path: Path, svg_path: Path) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    try:
        image = read_png_image(png_path)
        height, width = image.shape[:2]
    except (OSError, SyntaxError, ValueError):
        width, height = 2400, 1600
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><title>{escape(png_path.stem)}</title>'
        f'<image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/></svg>\n'
    )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(content, encoding="utf-8")


def _write_raster_pdf(png_path: Path, pdf_path: Path) -> None:
    image = read_png_image(png_path)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(max(width / 300.0, 1.0), max(height / 300.0, 1.0)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    ax.axis("off")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)


def _write_registry_readme(output_dir: Path, specs: list[FigureSpec]) -> Path:
    rows = [
        "# Figure captions and accessibility registry",
        "",
        "This file is generated from `src.figure_contract`. The canonical machine-readable registry is `output/data/figure_registry.json`; the sibling template consumes `output/figures/figure_registry.json`.",
        "",
        "| Figure | Placement | Caption | Status | Population | Statistics | Source receipts | Alternative text | Visual QA |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for spec in specs:
        status = (
            "present" if (output_dir.parent / spec.png_path).is_file() else "missing"
        )
        rows.append(
            f"| `{spec.figure_id}.png` ({status}) | {spec.placement} | {spec.caption} | `{spec.evidence_status}` / `{spec.receipt_status}` | {spec.population} | {spec.metrics}; {spec.uncertainty} | {', '.join(spec.input_artifacts)} | {spec.alt_text} | `{spec.visual_qa.get('status')}` |"
        )
    path = output_dir / "figures" / "README.md"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _registry_envelope(root: Path, specs: list[FigureSpec]) -> dict[str, Any]:
    """Return common protocol and provenance fields for registry projections."""

    protocol_path = root / "output" / "data" / "research_protocol.json"
    protocol = _load(protocol_path) if protocol_path.is_file() else {}
    artifacts = sorted(
        {
            artifact
            for spec in specs
            for artifact in spec.input_artifacts
            if artifact not in _VALIDATOR_PRODUCED_INPUTS
        }
    )
    return {
        "schema_version": FIGURE_REGISTRY_SCHEMA_VERSION,
        "compatible_schema_versions": list(FIGURE_REGISTRY_COMPATIBLE_SCHEMA_VERSIONS),
        "protocol_id": str(protocol.get("protocol_id", "")),
        "protocol_sha256": str(protocol.get("protocol_sha256", "")),
        "generator_revision": _sha256(Path(__file__)),
        "input_hashes": _hash_inputs(root, artifacts),
        "status": "implemented_local",
        "limitations": [
            "Figure registry evidence is source-bound to the listed receipts and visual producers; it is not a scientific promotion by itself."
        ],
        "unsupported_cases": [],
        "underpowered_cases": [],
        "generated_at": datetime.now(UTC).isoformat(),
        "figure_count": len(specs),
        "figures": [spec.to_dict() for spec in specs],
    }


def write_figure_atlas(output_dir: Path, specs: list[FigureSpec] | None = None) -> Path:
    """Write the offline semantic HTML atlas and its registry projection."""

    root = _root_for(output_dir)
    output_dir = root / "output"
    specs = build_figure_specs(output_dir) if specs is None else specs
    atlas_dir = output_dir / "figure_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for spec in specs:
        input_rows = "".join(
            f'<tr><th scope="row">{escape(path)}</th><td><code>{escape(spec.input_hashes.get(path, "missing"))}</code></td></tr>'
            for path in spec.input_artifacts
        )
        data_table = (
            f'<table id="{escape(spec.figure_id)}-data-table" class="source-table"><caption>Source artifacts and SHA-256 digests</caption>'
            '<thead><tr><th scope="col">Artifact</th><th scope="col">Digest</th></tr></thead>'
            f"<tbody>{input_rows}</tbody></table>"
        )
        cards.append(
            f'<section id="{escape(spec.figure_id)}"><h2>{escape(spec.title)}</h2>'
            f'<p class="subtitle">{escape(spec.subtitle)}</p>'
            f'<figure><img src="../figures/{escape(Path(spec.png_path).name)}" alt="{escape(spec.alt_text)}">'
            f"<figcaption><strong>{escape(spec.caption)}</strong></figcaption></figure>"
            f"<p><strong>Evidence status:</strong> <code>{escape(spec.evidence_status)}</code>; receipt status: <code>{escape(spec.receipt_status)}</code>.</p>"
            f"<p><strong>Population:</strong> {escape(spec.population)}<br><strong>Hierarchy:</strong> {escape(spec.sample_hierarchy)}<br><strong>Split:</strong> {escape(spec.split)}<br><strong>Metrics:</strong> {escape(spec.metrics)}<br><strong>Uncertainty:</strong> {escape(spec.uncertainty)}</p>"
            f"<p><strong>Source note:</strong> {escape(spec.source_note)}</p>"
            f"<p><strong>Quality contract:</strong> caption {spec.caption_word_count} words; alternative text {spec.alt_text_word_count} words; long description {spec.long_description_word_count} words; {escape(spec.palette_contract)} Visual QA: <code>{escape(spec.visual_qa_status)}</code>.</p>"
            f"<details><summary>Extended description</summary><p>{escape(spec.long_description)}</p></details>"
            f'<p class="downloads"><a href="../figures/vector/{escape(Path(spec.svg_path).name)}">SVG</a> · <a href="../figures/vector/{escape(Path(spec.pdf_path).name)}">PDF</a></p>'
            f"{data_table}</section>"
        )
    registry = _registry_envelope(root, specs)
    write_json_atomic(atlas_dir / "figure_registry.json", registry)
    index = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Active Assembly Theory and Practice — figure atlas</title><style>"
        'body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45;color:#243447;max-width:1200px;margin:0 auto;padding:2rem}'
        "h1{font-size:2rem}h2{margin-bottom:.25rem;color:#17324d}.subtitle{color:#52606d;margin-top:0}"
        "section{border-top:3px solid #d9e2ec;padding:1.5rem 0}figure{margin:1rem 0}figure img{display:block;max-width:100%;height:auto;border:1px solid #9aa5b1}figcaption{margin-top:.75rem;max-width:100ch}"
        "details{background:#f5f7fa;padding:.8rem;margin:.8rem 0}table{border-collapse:collapse;width:100%;font-size:.86rem}th,td{border:1px solid #d9e2ec;padding:.35rem;text-align:left;vertical-align:top}th{background:#e9eef3}.source-table{margin-top:1rem}code{overflow-wrap:anywhere}.downloads{font-weight:600}"
        "</style></head><body><header><h1>Source-bound visual atlas</h1>"
        f"<p>Technical-monograph figures generated from the frozen protocol and receipt surface. {len(specs)} substantive visuals are indexed; the cover is a separate document asset. The atlas is offline, semantic HTML with visible captions, short alternatives, expandable long descriptions, and exact source digests.</p>"
        '<nav aria-label="Figure index"><ol>'
        + "".join(
            f'<li><a href="#{escape(spec.figure_id)}">{escape(spec.title)}</a></li>'
            for spec in specs
        )
        + "</ol></nav></header>"
        + "".join(cards)
        + "</body></html>\n"
    )
    index_path = atlas_dir / "index.html"
    index_path.write_text(index, encoding="utf-8")
    return index_path


def write_figure_bundle(
    output_dir: Path,
    formats: Iterable[str] = ("png", "svg", "pdf", "atlas"),
) -> dict[str, str]:
    """Generate narrative visuals, vector companions, registries, and atlas."""

    root = _root_for(output_dir)
    output_dir = root / "output"
    requested = {str(value).lower() for value in formats}
    write_visual_atlas_figures(output_dir)
    missing = [
        stem
        for stem in FIGURE_CAPTIONS
        if not (output_dir / "figures" / f"{stem}.png").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"figure bundle missing PNG producers: {missing}")
    if "svg" in requested or "pdf" in requested:
        vector_dir = output_dir / "figures" / "vector"
        vector_dir.mkdir(parents=True, exist_ok=True)
        for stem in FIGURE_CAPTIONS:
            # Narrative figures write native vector output in
            # ``write_visual_atlas_figures``.  Legacy figures are raster
            # producers, so their compatibility SVG/PDF companions must be
            # rewritten on every bundle build; retaining an older embedded
            # raster would silently desynchronise the vector surface from the
            # current PNG and its receipt hash.
            if stem in NARRATIVE_FIGURE_IDS:
                continue
            png_path = output_dir / "figures" / f"{stem}.png"
            svg_path = vector_dir / f"{stem}.svg"
            pdf_path = vector_dir / f"{stem}.pdf"
            if "svg" in requested:
                _write_raster_svg(png_path, svg_path)
            if "pdf" in requested:
                _write_raster_pdf(png_path, pdf_path)
    # The legacy writer remains part of the compatibility boundary; the
    # canonical rows below replace its shorter metadata projection atomically.
    write_figure_captions(output_dir)
    specs = build_figure_specs(output_dir)
    readme_path = _write_registry_readme(output_dir, specs)
    registry_payload = _registry_envelope(root, specs)
    write_json_artifact(
        output_dir / "data" / "figure_registry.json",
        registry_payload,
        project_root=root,
        producer_path=Path(__file__),
    )
    template_registry = {
        "schema_version": "figure-registry-v1",
        "figures": [
            {
                "label": spec.label,
                "filename": Path(spec.png_path).name,
                "alt": spec.alt_text,
            }
            for spec in specs
        ],
    }
    write_json_artifact(
        output_dir / "figures" / "figure_registry.json",
        template_registry,
        project_root=root,
        producer_path=Path(__file__),
    )
    atlas_path = ""
    if "atlas" in requested:
        atlas_path = str(write_figure_atlas(output_dir, specs).relative_to(root))
    return {
        "registry": str(
            (output_dir / "data" / "figure_registry.json").relative_to(root)
        ),
        "template_registry": str(
            (output_dir / "figures" / "figure_registry.json").relative_to(root)
        ),
        "caption_ledger": str(readme_path.relative_to(root)),
        "atlas": atlas_path,
        **{stem: f"output/figures/{stem}.png" for stem in NARRATIVE_FIGURE_IDS},
    }


def validate_figure_bundle(output_dir: Path) -> list[str]:
    """Return fail-closed diagnostics for the canonical figure bundle."""

    root = _root_for(output_dir)
    output_dir = root / "output"
    errors: list[str] = []
    try:
        specs = build_figure_specs(output_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return [f"figure_spec_build: {exc}"]
    expected_count = len(FIGURE_CAPTIONS)
    if len(specs) != expected_count:
        errors.append(f"figure_count={len(specs)} expected={expected_count}")
    ids = [spec.figure_id for spec in specs]
    labels = [spec.label for spec in specs]
    if len(ids) != len(set(ids)):
        errors.append("duplicate figure IDs")
    if len(labels) != len(set(labels)):
        errors.append("duplicate figure labels")
    for spec in specs:
        png = root / spec.png_path
        if not png.is_file():
            errors.append(f"{spec.figure_id}: missing PNG")
        else:
            actual = _sha256(png)
            if spec.visual_qa.get("sha256") != actual:
                errors.append(f"{spec.figure_id}: stale PNG hash")
            if spec.visual_qa.get("status") != "programmatic_nonblank":
                errors.append(f"{spec.figure_id}: blank or unreadable PNG")
        for artifact in spec.input_artifacts:
            path = root / artifact
            if path.is_file():
                if spec.input_hashes.get(artifact) != _sha256(path):
                    errors.append(f"{spec.figure_id}: stale source hash {artifact}")
            elif artifact == "output/data/independent_replay_receipt.json":
                # The separately controlled replay receipt is an optional
                # promotion artifact (required=False in the reproducibility
                # manifest); its absence is the documented not-observed
                # state, not a broken figure producer.
                continue
            else:
                errors.append(f"{spec.figure_id}: missing source artifact {artifact}")
        if not spec.alt_text.strip():
            errors.append(f"{spec.figure_id}: missing alternative text")
        if len(spec.alt_text.split()) > 45:
            errors.append(f"{spec.figure_id}: alternative text exceeds 45 words")
        if not spec.long_description.strip():
            errors.append(f"{spec.figure_id}: missing long description")
        if not spec.caption.strip():
            errors.append(f"{spec.figure_id}: missing caption")
        if not 30 <= len(spec.caption.split()) <= 260:
            errors.append(
                f"{spec.figure_id}: caption word count={len(spec.caption.split())}"
            )
        if "The visual is generated from the named receipt" in spec.caption:
            errors.append(f"{spec.figure_id}: repeated reader-caption boilerplate")
        if not 150 <= len(spec.long_description.split()) <= 360:
            errors.append(
                f"{spec.figure_id}: long description word count={len(spec.long_description.split())}"
            )
        if spec.caption_word_count != len(spec.caption.split()):
            errors.append(f"{spec.figure_id}: stale caption word count")
        if spec.alt_text_word_count != len(spec.alt_text.split()):
            errors.append(f"{spec.figure_id}: stale alternative-text word count")
        if spec.long_description_word_count != len(spec.long_description.split()):
            errors.append(f"{spec.figure_id}: stale long-description word count")
        if not spec.palette_contract.strip():
            errors.append(f"{spec.figure_id}: missing palette contract")
        if not spec.table_companion_path.strip():
            errors.append(f"{spec.figure_id}: missing table companion path")
        if not spec.source_note.strip():
            errors.append(f"{spec.figure_id}: missing source note")
        if re.search(
            r"\b(unavailable|unsupported|not observed|excluded|underpowered)\b[^.]{0,30}\b0\b",
            spec.caption,
            re.IGNORECASE,
        ):
            errors.append(f"{spec.figure_id}: possible status-to-zero substitution")
        if "http://" in spec.caption or "https://" in spec.caption:
            errors.append(f"{spec.figure_id}: caption contains remote URL")
        svg = root / spec.svg_path
        pdf = root / spec.pdf_path
        if svg.is_file() and b"<svg" not in svg.read_bytes()[:2048].lower():
            errors.append(f"{spec.figure_id}: SVG is not parseable")
        if pdf.is_file() and pdf.read_bytes()[:4] != b"%PDF":
            errors.append(f"{spec.figure_id}: PDF vector export is not parseable")
    registry_path = output_dir / "data" / "figure_registry.json"
    try:
        payload = _load(registry_path)
        if payload.get("schema_version") != FIGURE_REGISTRY_SCHEMA_VERSION:
            errors.append("canonical registry schema mismatch")
        if (
            payload.get("figure_count") != expected_count
            or len(payload.get("figures", [])) != expected_count
        ):
            errors.append("canonical registry count mismatch")
        compatible = payload.get("compatible_schema_versions", [])
        if not isinstance(compatible, list) or "2.0" not in compatible:
            errors.append("canonical registry compatibility declaration missing")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"canonical registry unreadable: {exc}")
    atlas_index = output_dir / "figure_atlas" / "index.html"
    atlas_registry = output_dir / "figure_atlas" / "figure_registry.json"
    if not atlas_index.is_file() or not atlas_registry.is_file():
        errors.append("offline atlas is incomplete")
    else:
        html = atlas_index.read_text(encoding="utf-8")
        if (
            html.count("<figure>") != expected_count
            or html.count("<figcaption>") != expected_count
        ):
            errors.append("offline atlas figure/figcaption parity failure")
        if any(
            remote in html for remote in ("http://", "https://", "<script", "<link")
        ):
            errors.append("offline atlas contains network or script dependency")
        if (
            html.count('alt="') != expected_count
            or html.count("<details>") != expected_count
        ):
            errors.append("offline atlas accessibility metadata incomplete")
    return errors


__all__ = [
    "FIGURE_REGISTRY_ARTIFACT",
    "FIGURE_REGISTRY_COMPATIBLE_SCHEMA_VERSIONS",
    "FIGURE_REGISTRY_SCHEMA_VERSION",
    "FigureSpec",
    "build_figure_specs",
    "validate_figure_bundle",
    "write_figure_atlas",
    "write_figure_bundle",
]
