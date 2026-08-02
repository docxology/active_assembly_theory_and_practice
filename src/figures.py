"""Publication figures for the assembly Active Inference scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .active_inference_bridge import InferenceResult
from .features import AssemblyObservation
from .generative_model import GenerativeModelSpec

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")


def plot_observation_pipeline(
    observations: Sequence[AssemblyObservation],
    output_path: Path,
) -> Path:
    """Plot instruction observations as a categorical pipeline."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(observations))
    y = [observation.observation_id for observation in observations]

    # Deterministic observation_id -> category map (first occurrence wins). Ties
    # the y-axis tick to the category vocabulary rather than to whichever step
    # happened to exhibit a given id first.
    category_by_id: dict[int, str] = {}
    for observation in observations:
        category_by_id.setdefault(observation.observation_id, observation.category)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(x, y, marker="o", color=PALETTE[0], linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([obs.mnemonic for obs in observations])
    ax.set_yticks(sorted(set(y)))
    ax.set_yticklabels([category_by_id[value] for value in sorted(set(y))])
    ax.set_xlabel("Instruction order")
    ax.set_ylabel("Observation category")
    ax.set_title("Assembly observations from decoded x86-64 bytes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_posterior_beliefs(
    inference: InferenceResult,
    model: GenerativeModelSpec,
    output_path: Path,
) -> Path:
    """Plot posterior beliefs over latent program states."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(inference.posterior_by_step, dtype=float)
    x = np.arange(matrix.shape[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for state_index, state_label in enumerate(model.hidden_state_labels):
        ax.plot(
            x,
            matrix[:, state_index],
            marker="o",
            linewidth=2,
            label=state_label.replace("_", " "),
            color=PALETTE[state_index % len(PALETTE)],
        )
    ax.set_xticks(x)
    ax.set_xlabel("Observation step")
    ax.set_ylabel("Posterior probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Variational state beliefs induced by assembly observations")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_rxinfer_free_energy(records: list[dict[str, Any]], output_path: Path) -> Path:
    """Plot RxInfer Bethe Free Energy values emitted by the Julia backend."""

    if not records:
        raise ValueError("free-energy records must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = [int(record["iteration"]) for record in records]
    values = [float(record["value"]) for record in records]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(iterations, values, marker="o", linewidth=2, color=PALETTE[2])
    ax.set_xlabel("RxInfer iteration")
    ax.set_ylabel("Bethe Free Energy")
    ax.set_title("RxInfer Bethe Free Energy trajectory")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_package_provenance(
    provenance_rows: list[dict[str, object]], output_path: Path
) -> Path:
    """Plot required and optional assembly package adapter availability."""

    if not provenance_rows:
        raise ValueError("package provenance rows must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(row["name"]) for row in provenance_rows]
    available = [1 if row["available"] else 0 for row in provenance_rows]
    colors = [PALETTE[2] if row["available"] else "#999999" for row in provenance_rows]
    hatch = ["" if not row["optional"] else "//" for row in provenance_rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    bars = ax.bar(names, available, color=colors, edgecolor="#333333", linewidth=0.7)
    for bar, pattern in zip(bars, hatch, strict=True):
        bar.set_hatch(pattern)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["unavailable", "available"])
    ax.set_ylabel("Adapter status")
    ax.set_title("Assembly package provenance and optional adapter probes")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_trace_feature_matrix(
    observations: list[AssemblyObservation],
    output_path: Path,
) -> Path:
    """Plot trace-derived feature semantics for each decoded instruction."""

    if not observations:
        raise ValueError("observations must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    categories = sorted({observation.category for observation in observations})
    category_index = {category: index for index, category in enumerate(categories)}
    matrix = np.array(
        [
            [
                1.0 if observation.executed else 0.0,
                float(observation.observation_id),
                1.0
                if any(char.isdigit() for char in observation.operand_text)
                else 0.0,
                1.0
                if observation.operand_text.lower().startswith(("rax", "eax"))
                else 0.0,
                float(category_index[observation.category]),
            ]
            for observation in observations
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(observations)))
    ax.set_yticklabels(
        [
            f"{observation.index}: {observation.mnemonic}"
            for observation in observations
        ],
        fontsize=8,
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        ["executed", "obs id", "immediate", "accumulator", "category"],
        rotation=30,
        ha="right",
    )
    ax.set_title("Trace feature matrix used for observation semantics")
    ax.set_xlabel("Feature channel")
    ax.set_ylabel("Instruction event")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_inference_diagnostics(
    pymdp_free_energy: list[float],
    rxinfer_free_energy: list[float],
    output_path: Path,
) -> Path:
    """Plot pymdp comparison diagnostics beside RxInfer Bethe Free Energy."""

    if not pymdp_free_energy or not rxinfer_free_energy:
        raise ValueError("both inference diagnostic series must be non-empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pymdp_steps = np.arange(1, len(pymdp_free_energy) + 1)
    rxinfer_steps = np.arange(1, len(rxinfer_free_energy) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.8), sharex=False)
    axes[0].plot(
        pymdp_steps,
        pymdp_free_energy,
        marker="o",
        linewidth=2,
        color=PALETTE[0],
    )
    axes[0].set_title("pymdp comparison variational diagnostic")
    axes[0].set_xlabel("Observation step")
    axes[0].set_ylabel("Diagnostic value")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(
        rxinfer_steps,
        rxinfer_free_energy,
        marker="o",
        linewidth=2,
        color=PALETTE[2],
    )
    axes[1].set_title("RxInfer Bethe Free Energy")
    axes[1].set_xlabel("RxInfer iteration")
    axes[1].set_ylabel("Bethe Free Energy")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Inference diagnostics are recorded as separate objectives", y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_validation_summary(checks: list[dict[str, object]], output_path: Path) -> Path:
    """Plot pass/fail status for analysis validation checks."""

    if not checks:
        raise ValueError("validation checks must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(check["name"]).replace("_", " ") for check in checks]
    status = [1 if check["passed"] else 0 for check in checks]
    colors = [PALETTE[0] if check["passed"] else "#D55E00" for check in checks]

    fig, ax = plt.subplots(figsize=(8.0, max(3.8, 0.36 * len(checks))))
    y = np.arange(len(checks))
    ax.barh(y, status, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fail", "pass"])
    ax.set_xlabel("Validation status")
    ax.set_title("Analysis validation report")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_artifact_hash_manifest(rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Plot tracked artifact bytes by artifact class."""

    if not rows:
        raise ValueError("artifact manifest rows must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_by_type: dict[str, int] = {}
    count_by_type: dict[str, int] = {}
    missing_by_type: dict[str, int] = {}
    for row in rows:
        artifact_type = str(row["artifact_type"])
        bytes_by_type[artifact_type] = bytes_by_type.get(artifact_type, 0) + int(
            row["bytes"]
        )
        count_by_type[artifact_type] = count_by_type.get(artifact_type, 0) + 1
        if row["status"] != "present":
            missing_by_type[artifact_type] = missing_by_type.get(artifact_type, 0) + 1

    labels = sorted(bytes_by_type)
    values_kib = [bytes_by_type[label] / 1024.0 for label in labels]
    colors = [
        "#D55E00" if missing_by_type.get(label, 0) else PALETTE[index % len(PALETTE)]
        for index, label in enumerate(labels)
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values_kib, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label.replace("_", " ") for label in labels], rotation=28, ha="right"
    )
    ax.set_ylabel("Tracked bytes (KiB)")
    ax.set_title("Reproducibility manifest coverage by artifact class")
    ax.grid(axis="y", alpha=0.2)
    for bar, label in zip(bars, labels, strict=True):
        text = f"{count_by_type[label]} files"
        if missing_by_type.get(label, 0):
            text = f"{text}; {missing_by_type[label]} missing"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values_kib) * 0.02,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_scholarship_source_roles(
    sources: list[dict[str, object]],
    output_path: Path,
) -> Path:
    """Plot scholarship source types used by the manuscript."""

    if not sources:
        raise ValueError("scholarship sources must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for source in sources:
        source_type = str(source["source_type"])
        counts[source_type] = counts.get(source_type, 0) + 1
    labels = list(counts)
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(labels))
    ax.bar(
        x, values, color=[PALETTE[index % len(PALETTE)] for index in range(len(labels))]
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label.replace("_", " ") for label in labels], rotation=30, ha="right"
    )
    ax.set_ylabel("Source count")
    ax.set_title("Scholarship source roles and evidence tiers")
    ax.grid(axis="y", alpha=0.2)
    for index, value in enumerate(values):
        ax.text(index, value + 0.05, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_scholarship_evidence_matrix(
    sources: list[dict[str, object]],
    output_path: Path,
) -> Path:
    """Plot source counts by project stage and evidence tier."""

    if not sources:
        raise ValueError("scholarship sources must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stages = ["implemented", "comparison", "roadmap", "context"]
    tiers = sorted({str(source["evidence_tier"]) for source in sources})
    tier_index = {tier: index for index, tier in enumerate(tiers)}
    stage_index = {stage: index for index, stage in enumerate(stages)}
    matrix = np.zeros((len(tiers), len(stages)), dtype=float)
    for source in sources:
        stage = str(source["project_stage"])
        tier = str(source["evidence_tier"])
        if stage in stage_index and tier in tier_index:
            matrix[tier_index[tier], stage_index[stage]] += 1.0

    fig, ax = plt.subplots(figsize=(8.2, max(4.2, 0.38 * len(tiers))))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(tiers)))
    ax.set_yticklabels([tier.replace("_", " ") for tier in tiers], fontsize=8)
    ax.set_title("Scholarship evidence tiers by implementation stage")
    ax.set_xlabel("Project stage")
    ax.set_ylabel("Evidence tier")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            if value:
                ax.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_claim_evidence_map(
    claims: list[dict[str, object]],
    output_path: Path,
) -> Path:
    """Plot claim coverage by source, artifact, and validation-link counts."""

    if not claims:
        raise ValueError("claim records must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(claim["claim_id"]) for claim in claims]
    source_counts = [len(cast(list, claim["support_keys"])) for claim in claims]
    artifact_counts = [len(cast(list, claim["artifact_paths"])) for claim in claims]
    validation_counts = [
        len(cast(list, claim["validation_checks"])) for claim in claims
    ]
    x = np.arange(len(claims))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x - width, source_counts, width=width, label="sources", color=PALETTE[0])
    ax.bar(x, artifact_counts, width=width, label="artifacts", color=PALETTE[2])
    ax.bar(x + width, validation_counts, width=width, label="checks", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Linked item count")
    ax.set_title("Claim register evidence coverage")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_redteam_risk_register(
    findings: list[dict[str, str]], output_path: Path
) -> Path:
    """Plot RedTeam finding status by severity."""

    if not findings:
        raise ValueError("RedTeam findings must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{finding['finding_id']} {finding['vector']}".replace("_", " ")
        for finding in findings
    ]
    severity_score = {"critical": 3, "investigate": 2, "clean": 1}
    values = [severity_score.get(str(finding["severity"]), 1) for finding in findings]
    colors = [
        PALETTE[3] if finding["status"] != "remediated" else PALETTE[2]
        for finding in findings
    ]

    fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.48 * len(findings))))
    y = np.arange(len(findings))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["clean", "investigate", "critical"])
    ax.set_xlabel("Finding severity")
    ax.set_title("RedTeam risk register and remediation status")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# v0.2 temporal / active / evaluation visualizations
# ---------------------------------------------------------------------------


def plot_temporal_beliefs_heatmap(
    fixture_name: str,
    observation_ids: list[int],
    filtered_posteriors: list[list[float]],
    hidden_state_labels: list[str],
    output_path: Path,
) -> Path:
    """Plot a single fixture's forward-filter belief trajectory as a heatmap.

    Rows are observation steps (labelled by decoded observation category); columns
    are the five latent program-phase beliefs. Each row is a normalized belief, so
    columns sum to one row-wise.
    """

    if not filtered_posteriors:
        raise ValueError("filtered_posteriors must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(filtered_posteriors, dtype=float)
    step_labels = [
        f"{index + 1}: obs {obs_id}" for index, obs_id in enumerate(observation_ids)
    ]

    fig, ax = plt.subplots(figsize=(7.6, max(2.6, 0.42 * len(step_labels) + 1.2)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(step_labels)))
    ax.set_yticklabels(step_labels, fontsize=8)
    ax.set_xticks(np.arange(len(hidden_state_labels)))
    ax.set_xticklabels(
        [label.replace("_", " ") for label in hidden_state_labels],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Observation step")
    ax.set_xlabel("Latent program-phase belief")
    ax.set_title(f"Temporal HMM forward-filter beliefs: {fixture_name}")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Posterior probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_model_comparison_bars(
    evaluation_report: dict[str, object],
    output_path: Path,
) -> Path:
    """Plot per-fixture decoding accuracy and pooled accuracy with 95% Wilson CIs.

    Left panel: sticky vs memoryless vs temporal accuracy per corpus fixture.
    Right panel: the pooled (micro) accuracy of each decoder with its 95% Wilson
    confidence interval, so the headline uncertainty is visible rather than just
    a point estimate.
    """

    compare = cast(
        dict[str, dict[str, float]], evaluation_report.get("compare_models", {})
    )
    if not compare:
        raise ValueError("evaluation_report must contain compare_models rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_names = list(compare.keys())
    sticky = [compare[name].get("static_accuracy", 0.0) for name in fixture_names]
    memoryless = [
        compare[name].get("memoryless_accuracy", 0.0) for name in fixture_names
    ]
    temporal = [compare[name].get("temporal_accuracy", 0.0) for name in fixture_names]
    x = np.arange(len(fixture_names))
    width = 0.26

    summary = cast(dict[str, object], evaluation_report.get("summary", {}))
    pooled_accs = [
        cast(float, summary.get("pooled_static_accuracy", 0.0)),
        cast(float, summary.get("pooled_memoryless_accuracy", 0.0)),
        cast(float, summary.get("pooled_temporal_accuracy", 0.0)),
    ]
    ci_lo = [
        cast(list[float], summary.get("static_ci95", [0.0, 0.0]))[0],
        cast(list[float], summary.get("memoryless_ci95", [0.0, 0.0]))[0],
        cast(list[float], summary.get("temporal_ci95", [0.0, 0.0]))[0],
    ]
    ci_hi = [
        cast(list[float], summary.get("static_ci95", [0.0, 0.0]))[1],
        cast(list[float], summary.get("memoryless_ci95", [0.0, 0.0]))[1],
        cast(list[float], summary.get("temporal_ci95", [0.0, 0.0]))[1],
    ]
    decoder_labels = ["sticky (v0.1)", "memoryless", "temporal HMM"]
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), sharey=True)
    ax = axes[0]
    ax.bar(x - width, sticky, width=width, label="sticky (v0.1)", color=PALETTE[0])
    ax.bar(x, memoryless, width=width, label="memoryless", color=PALETTE[1])
    ax.bar(x + width, temporal, width=width, label="temporal HMM", color=PALETTE[2])
    ax.axhline(
        0.2, color="#333333", linewidth=0.8, linestyle="--", label="chance (0.20)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [name.replace("x86_", "") for name in fixture_names], rotation=15, ha="right"
    )
    ax.set_ylabel("Phase-decoding accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-fixture accuracy")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax2 = axes[1]
    ypos = np.arange(len(decoder_labels))
    ax2.errorbar(
        pooled_accs,
        ypos,
        xerr=np.array(
            [
                [p - lo for p, lo in zip(pooled_accs, ci_lo)],
                [hi - p for p, hi in zip(pooled_accs, ci_hi)],
            ]
        ),
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1.5,
    )
    for yi, (p, color) in enumerate(zip(pooled_accs, colors)):
        ax2.plot(p, yi, "o", color=color, markersize=8)
    ax2.axvline(
        0.2, color="#333333", linewidth=0.8, linestyle="--", label="chance (0.20)"
    )
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(decoder_labels)
    ax2.set_xlabel("Pooled accuracy (95% Wilson CI)")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_title("Pooled accuracy with 95% CI")
    ax2.grid(axis="x", alpha=0.2)

    fig.suptitle(
        "Decoder comparison: per-fixture and pooled accuracy with confidence",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_active_efe_trajectory(
    selection: dict[str, object],
    output_path: Path,
) -> Path:
    """Plot the greedy active-disassembly expected-free-energy and belief-entropy trajectory."""

    efe_by_step = cast(list[float], selection.get("efe_by_step"))
    entropy_by_step = cast(list[float], selection.get("belief_entropy_by_step"))
    if not efe_by_step or not entropy_by_step:
        raise ValueError(
            "selection must contain efe_by_step and belief_entropy_by_step"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, len(efe_by_step) + 1)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(
        steps,
        efe_by_step,
        marker="o",
        linewidth=2,
        color=PALETTE[0],
        label="Expected free energy",
    )
    ax.plot(
        steps,
        entropy_by_step,
        marker="s",
        linewidth=2,
        color=PALETTE[2],
        label="Belief entropy (nats)",
    )
    ax.set_xticks(steps)
    ax.set_xlabel("Active-selection step")
    ax.set_ylabel("Value")
    ax.set_title("Active-disassembly EFE and belief-entropy trajectory")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_calibration_curve(bins: list[dict[str, object]], output_path: Path) -> Path:
    """Plot a reliability (calibration) diagram from per-bin statistics.

    Each bin's mean confidence (max posterior) is drawn against its observed
    accuracy, with the identity line for perfect calibration. Point labels show
    the number of steps in each bin.
    """

    if not bins:
        raise ValueError("calibration bins must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    confidences = [cast(float, bin_["confidence"]) for bin_ in bins]
    accuracies = [cast(float, bin_["accuracy"]) for bin_ in bins]
    counts = [cast(int, bin_["count"]) for bin_ in bins]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label="Perfect calibration",
    )
    ax.plot(
        confidences,
        accuracies,
        marker="o",
        linewidth=2,
        color=PALETTE[0],
        label="Forward filter",
    )
    for cx, cy, count in zip(confidences, accuracies, counts, strict=True):
        ax.annotate(
            str(count),
            (cx, cy),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence (mean max posterior)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability diagram: forward-filter calibration")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(pooled: dict[str, object], output_path: Path) -> Path:
    """Plot a pooled phase-decoding confusion matrix as an annotated heatmap."""

    labels = cast(list[str], pooled.get("labels"))
    matrix = np.asarray(cast(list[list[int]], pooled.get("matrix")), dtype=float)
    if matrix.size == 0 or matrix.shape != (len(labels), len(labels)):
        raise ValueError("confusion matrix must be square and match label count")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(
        [label.replace("_", " ") for label in labels], rotation=30, ha="right"
    )
    ax.set_yticklabels([label.replace("_", " ") for label in labels])
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
                fontsize=9,
            )
    ax.set_xlabel("Predicted phase")
    ax.set_ylabel("True phase")
    ax.set_title("Pooled confusion matrix (temporal HMM, all fixtures)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_ambiguity_ablation(ablation: dict[str, object], output_path: Path) -> Path:
    """Plot decoder accuracy, segmentation F1, and EFE-active advantage vs ambiguity.

    Three panels. Left: mean step-decoding accuracy of the temporal, memoryless,
    sticky, and Viterbi decoders as the likelihood is blended toward uniform —
    per-step decoding degrades and the temporal transition structure carries the
    information. Middle: pooled phase-boundary F1 for the temporal (smoothed) and
    memoryless segmentations. Right: how many fixtures reach confidence in fewer
    steps under the EFE-active policy than under a fixed scan — an advantage that
    appears only in an intermediate-ambiguity band where observations become
    differentially informative. The crossover (temporal first beating memoryless)
    is annotated on every panel, and the figure shows all three inference layers
    gaining their mechanism exactly under observation ambiguity.
    """

    rows = cast(list[dict[str, object]], ablation.get("rows"))
    if not rows:
        raise ValueError("ambiguity ablation must contain rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blends = [cast(float, row["blend"]) for row in rows]
    crossover: float | None = cast(float | None, ablation.get("crossover_blend"))

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.6), sharex=True)
    ax_left, ax_mid, ax_right = axes

    accuracy_series = {
        "temporal HMM": ("temporal_accuracy", PALETTE[2], "-"),
        "memoryless": ("memoryless_accuracy", PALETTE[1], "--"),
        "sticky (v0.1)": ("static_accuracy", PALETTE[0], ":"),
        "viterbi": ("viterbi_accuracy", "#CC79A7", "-."),
    }
    for label, (key, color, linestyle) in accuracy_series.items():
        ax_left.plot(
            blends,
            [cast(float, row[key]) for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    if crossover is not None:
        ax_left.axvline(
            crossover,
            color="#333333",
            linewidth=1.0,
            linestyle=":",
            label=f"crossover {crossover:.1f}",
        )
    ax_left.set_xlabel("Observation-ambiguity blend (0 = authored, 1 = uniform)")
    ax_left.set_ylabel("Mean decoding accuracy")
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_title("Step decoding accuracy")
    ax_left.legend(loc="lower left", fontsize=8)
    ax_left.grid(axis="both", alpha=0.2)

    ax_mid.plot(
        blends,
        [cast(float, row.get("temporal_segmentation_f1", 0.0)) for row in rows],
        marker="s",
        linewidth=2,
        color=PALETTE[2],
        linestyle="-",
        label="temporal segmentation",
    )
    ax_mid.plot(
        blends,
        [cast(float, row.get("memoryless_segmentation_f1", 0.0)) for row in rows],
        marker="s",
        linewidth=2,
        color="#555555",
        linestyle="--",
        label="memoryless segmentation",
    )
    if crossover is not None:
        ax_mid.axvline(crossover, color="#333333", linewidth=1.0, linestyle=":")
    ax_mid.set_xlabel("Observation-ambiguity blend (0 = authored, 1 = uniform)")
    ax_mid.set_ylabel("Pooled boundary F1")
    ax_mid.set_ylim(0.0, 1.0)
    ax_mid.set_title("Phase-segmentation boundary F1")
    ax_mid.legend(loc="lower left", fontsize=8)
    ax_mid.grid(axis="both", alpha=0.2)

    active_counts = [cast(int, row.get("active_advantage_count", 0)) for row in rows]
    step = (
        (blends[-1] - blends[0]) / max(1, len(blends) - 1) if len(blends) > 1 else 0.05
    )
    ax_right.bar(
        blends,
        active_counts,
        width=0.7 * step,
        color=PALETTE[2],
        alpha=0.85,
        label="# fixtures where EFE active reaches confidence first",
    )
    if crossover is not None:
        ax_right.axvline(crossover, color="#333333", linewidth=1.0, linestyle=":")
    ax_right.set_xlabel("Observation-ambiguity blend (0 = authored, 1 = uniform)")
    ax_right.set_ylabel("Fixtures with EFE-active advantage")
    ax_right.set_ylim(0.0, max(max(active_counts), 1) + 0.5)
    ax_right.set_yticks(range(0, max(max(active_counts), 1) + 1))
    ax_right.set_title("Active (EFE) advantage under ambiguity")
    ax_right.legend(loc="upper right", fontsize=8)
    ax_right.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _draw_segment_runs(
    ax: Any, y: float, sequence: Sequence[str], phase_colors: dict[str, Any]
) -> None:
    """Draw a phase sequence as contiguous colored runs on one ribbon row."""

    i = 0
    while i < len(sequence):
        j = i
        while j < len(sequence) and sequence[j] == sequence[i]:
            j += 1
        ax.barh(
            y,
            j - i,
            left=i,
            height=0.8,
            color=phase_colors.get(sequence[i], "#cccccc"),
            edgecolor="white",
            linewidth=0.5,
        )
        i = j


def plot_segmentation_ribbons(
    evaluation_report: dict[str, object], output_path: Path
) -> Path:
    """Plot per-fixture ground-truth vs predicted phase-segmentation ribbons.

    For each fixture two horizontal ribbons are drawn: the hand-authored
    ground-truth phase sequence (top) and the phase sequence decoded from the
    smoothed forward-backward beliefs (bottom). Contiguous segments sharing a
    phase are merged into single colored runs, so phase boundaries appear as
    color changes. This makes the measured segmentation (CLM-015) visual.
    """

    segmented = cast(dict[str, object], evaluation_report.get("segmentation"))
    per_fixture = cast(dict[str, dict[str, object]], segmented.get("per_fixture"))
    if not per_fixture:
        raise ValueError("segmentation per_fixture must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    for entry in per_fixture.values():
        for label in cast(list[str], entry.get("truth_labels", [])) + cast(
            list[str], entry.get("predicted_labels", [])
        ):
            if label not in labels:
                labels.append(label)
    cmap = plt.get_cmap("tab10")
    phase_colors = {label: cmap(i % 10) for i, label in enumerate(labels)}

    max_len = max(
        len(cast(list[str], entry.get("truth_labels", [])))
        for entry in per_fixture.values()
    )
    n_fixtures = len(per_fixture)
    fig, ax = plt.subplots(figsize=(0.55 * max_len + 2.0, 1.2 * n_fixtures + 1.5))

    y = n_fixtures * 2 - 1
    yticks: list[int] = []
    yticklabels: list[str] = []
    for name, entry in per_fixture.items():
        truth = cast(list[str], entry.get("truth_labels", []))
        predicted = cast(list[str], entry.get("predicted_labels", []))
        _draw_segment_runs(ax, y, truth, phase_colors)
        yticks.append(y)
        yticklabels.append(f"{name.replace('x86_', '')} · truth")
        y -= 1
        _draw_segment_runs(ax, y, predicted, phase_colors)
        yticks.append(y)
        yticklabels.append(f"{name.replace('x86_', '')} · predicted")
        y -= 1

    pooled = cast(dict[str, object], segmented.get("pooled", {}))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlim(0, max_len)
    ax.set_xticks(range(0, max_len + 1))
    ax.set_xlabel("Instruction index")
    ax.set_title(
        "Phase segmentation: ground truth vs smoothed prediction "
        f"(pooled boundary F1 = {float(cast(float, pooled.get('boundary_f1', 0.0))):.2f})"
    )
    for i in range(1, max_len):
        ax.axvline(i, color="#dddddd", linewidth=0.5, zorder=0)
    proxies = [
        Rectangle((0, 0), 1, 1, color=phase_colors[label], label=label)
        for label in labels
    ]
    ax.legend(handles=proxies, loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_ylim(-1, n_fixtures * 2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_corpus_stability(stability: dict[str, object], output_path: Path) -> Path:
    """Plot leave-one-out pooled temporal accuracy to quantify corpus stability.

    Bars show the pooled temporal decoding accuracy on the full corpus and after
    holding out each single fixture; the dashed line is the pooled static (sticky)
    accuracy and the dotted line is chance. The headline temporal result surviving
    every single-fixture removal is made visual, and the span is annotated so the
    small-corpus fragility is explicit rather than hidden.
    """

    reference = cast(dict[str, object], stability.get("reference", {}))
    per_holdout = cast(list[dict[str, object]], stability.get("per_holdout", []))
    if not per_holdout:
        raise ValueError("corpus stability must contain per_holdout rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["full corpus"] + [
        cast(str, row["held_out_fixture"]).replace("x86_", "") for row in per_holdout
    ]
    temporal = [cast(float, reference.get("temporal", 0.0))] + [
        cast(float, row["temporal"]) for row in per_holdout
    ]
    static = [cast(float, reference.get("static", 0.0))] + [
        cast(float, row["static"]) for row in per_holdout
    ]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.bar(
        x, temporal, width=0.55, color=PALETTE[2], label="temporal HMM pooled accuracy"
    )
    ax.plot(
        x,
        static,
        marker="x",
        linewidth=1.5,
        linestyle="--",
        color=PALETTE[0],
        label="static (sticky) pooled",
    )
    chance = float(cast(float, stability.get("chance_floor", 0.2)))
    ax.axhline(
        chance,
        color="#555555",
        linewidth=1.0,
        linestyle=":",
        label=f"chance ({chance:.1f})",
    )
    pooled_range = cast(dict[str, object], stability.get("temporal_pooled_range", {}))
    span = float(cast(float, pooled_range.get("span", 0.0)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Pooled decoding accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        f"Corpus stability: leave-one-out pooled temporal accuracy (span {span:.2f})"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_phase_recovery(phase_recovery: dict[str, object], output_path: Path) -> Path:
    """Plot per-phase recall and F1 to show which phases are recoverable.

    Phases are ordered by F1 so the hardest (underdetermined) phases are on the
    left; recall and F1 are grouped bars, and the worst phase is highlighted. This
    makes quantitative — rather than asserted — which phases the temporal forward
    filter recovers (e.g. that ``exit`` and especially ``guard_check`` are
    observation-underdetermined).
    """

    per_phase = cast(dict[str, dict[str, object]], phase_recovery.get("per_phase"))
    if not per_phase:
        raise ValueError("phase recovery must contain per_phase rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(
        list(per_phase.keys()), key=lambda p: float(cast(float, per_phase[p]["f1"]))
    )
    f1s = [float(cast(float, per_phase[p]["f1"])) for p in labels]
    recalls = [float(cast(float, per_phase[p]["recall"])) for p in labels]
    x = np.arange(len(labels))
    hardest = cast(str, phase_recovery.get("hardest_phase", labels[0]))

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(x - 0.18, recalls, width=0.36, color="#9ecae1", label="recall")
    f1_colors = ["#d62728" if p == hardest else PALETTE[2] for p in labels]
    ax.bar(x + 0.18, f1s, width=0.36, color=f1_colors, label="F1")
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Recovery score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        f"Per-phase recovery (macro F1 = "
        f"{float(cast(float, phase_recovery.get('macro_f1', 0.0))):.2f}; "
        f"hardest: {hardest})"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_decoder_calibration(
    calibration_payload: dict[str, object], output_path: Path
) -> Path:
    """Plot ECE and Brier for each decoder to compare calibration quality.

    Shows expected calibration error (ECE) and Brier score for the temporal,
    static, and memoryless decoders (ordered by ECE). This makes the honest
    result visual: on this corpus calibration tracks accuracy — the memoryless
    decoder is best calibrated and the sticky static filter worst.
    """

    decoders = cast(dict[str, dict[str, object]], calibration_payload.get("decoders"))
    if not decoders:
        raise ValueError("decoder calibration must contain decoders")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = ["memoryless", "temporal", "static"]
    eces = [float(cast(float, decoders[name]["ece"])) for name in names]
    briers = [float(cast(float, decoders[name]["brier"])) for name in names]
    best = cast(str, calibration_payload.get("best_calibrated", names[0]))
    worst = cast(str, calibration_payload.get("worst_calibrated", names[-1]))
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x - 0.18, eces, width=0.36, color=PALETTE[2], label="ECE")
    ax.bar(x + 0.18, briers, width=0.36, color="#9ecae1", label="Brier")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Calibration error")
    ax.set_ylim(0.0, 0.8)
    ax.set_title(f"Decoder calibration (best: {best}, worst: {worst})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_generative_model(output_path: Path) -> Path:
    """Bayesian-graph (plate) diagram of the generative model.

    Schematic of the discrete generative model rendered from constants (no data),
    so it is deterministic. Hidden phase states ``s_t`` form a left-to-right
    Markov chain (transition ``B``) that emits categorical observations ``o_t``
    (likelihood ``A``); ``D`` is the initial prior. The repeating ``s``-``o``
    block is drawn as a plate. An active layer (policy ``pi``, expected free
    energy ``G(pi)``, and prior preference ``C``) extends the model to
    observation selection.
    """

    from matplotlib.patches import FancyArrowPatch, Rectangle

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    def node(x: float, y: float, label: str, fill: str = "#EAF1FB") -> None:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc=fill, ec="#333333", lw=1.0),
        )

    def edge(
        x1: float, y1: float, x2: float, y2: float, label: str | None = None
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", color="#333333", lw=1.2
            )
        )
        if label:
            ax.text(
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0 + 0.03,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="#333333",
            )

    s_x = [0.28, 0.60, 0.92]
    o_y = 0.26
    s_y = 0.62

    # hidden states and their observations
    for x in s_x:
        node(x, s_y, "")
    node(s_x[0], s_y, "$s_{t-1}$")
    node(s_x[1], s_y, "$s_t$")
    node(s_x[2], s_y, "$s_{t+1}$")
    node(s_x[0], o_y, "$o_{t-1}$", "#FEF3D9")
    node(s_x[1], o_y, "$o_t$", "#FEF3D9")
    node(s_x[2], o_y, "$o_{t+1}$", "#FEF3D9")
    # transition B between consecutive states
    edge(s_x[0], s_y, s_x[1], s_y, "B")
    edge(s_x[1], s_y, s_x[2], s_y, "B")
    # likelihood A from each state to its observation
    for x in s_x:
        edge(x, s_y, x, o_y, "A")
    # initial prior D
    node(0.10, s_y, "$D$", "#E8F5E9")
    edge(0.10, s_y, s_x[0], s_y)
    ax.text(0.20, s_y + 0.08, "prior", fontsize=8, ha="center", color="#333333")
    # plate over the repeating s-o block
    ax.add_patch(
        Rectangle(
            (s_x[1] - 0.14, o_y - 0.11),
            s_x[2] - s_x[1] + 0.30,
            s_y - o_y + 0.22,
            fill=False,
            edgecolor="#777777",
            linewidth=1.0,
            linestyle="--",
        )
    )
    ax.text(
        s_x[2] - 0.04,
        s_y + 0.11,
        "t = 1, …, T",
        fontsize=8,
        ha="right",
        color="#555555",
    )
    # --- active / EFE layer ---
    ax.axvline(1.24, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax.text(1.56, 0.94, "active layer", fontsize=9, style="italic", color="#444444")
    node(1.56, 0.80, "policy $\\pi$", "#F3E5F5")
    node(1.56, 0.48, "preference $C$", "#E8F5E9")
    node(1.56, 0.26, "select next $o$", "#FEF3D9")
    edge(1.56, 0.72, 1.56, 0.58, "G($\\pi$)")
    edge(1.56, 0.40, 1.56, 0.34)
    ax.annotate(
        "beliefs",
        xy=(s_x[2], s_y),
        xytext=(1.32, 0.80),
        arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0, linestyle="--"),
        fontsize=7,
        color="#555555",
    )
    ax.text(
        1.56,
        0.68,
        "expected free\nenergy, G($\\pi$)",
        fontsize=6.5,
        ha="center",
        va="center",
        color="#333333",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure caption registry
# ---------------------------------------------------------------------------
# Captions are written next to every output figure so the rendered manuscript and
# any figure inventory stay consistent with what each PNG actually shows. Keys
# match the generated PNG stem under ``output/figures/``.

FIGURE_CAPTIONS: dict[str, str] = {
    "generative_model": (
        "Bayesian-graph (plate) diagram of the generative model. Hidden phase states s_t form "
        "a left-to-right Markov chain (transition B) emitting categorical observations o_t "
        "(likelihood A), with prior D; the repeating s-o block is the plate. An active layer "
        "(policy pi, expected free energy G(pi), preference C) extends the model to observation "
        "selection. Rendered from constants, so it is deterministic."
    ),
    "observation_pipeline": (
        "Assembly observation pipeline. Each decoded x86-64 instruction (labelled by mnemonic) "
        "is mapped to a categorical observation id; these ids feed the generative model."
    ),
    "posterior_beliefs": (
        "Variational posterior beliefs over the five latent program-state classes at each of the "
        "four observation steps. Beliefs are produced by the v0.1 sticky filter (explicit "
        "likelihood update, implicit identity transition)."
    ),
    "package_provenance": (
        "Adapter provenance for the required (Capstone, Unicorn) and optional assembly-analysis "
        "packages. Hatched bars are optional roadmap slots; grey bars were probed but are unavailable."
    ),
    "trace_feature_matrix": (
        "Trace feature matrix for every decoded instruction: whether it executed, its observation "
        "id, whether it carries an immediate operand, whether it touches the accumulator, and its "
        "categorical channel."
    ),
    "rxinfer_free_energy": (
        "Bethe Free Energy trajectory emitted by the native Julia/RxInfer backend (Gaussian "
        "event-code model) for the v0.1 fixture."
    ),
    "inference_diagnostics": (
        "Separate diagnostic traces: the pymdp comparison variational diagnostic (upper) and the "
        "RxInfer Bethe Free Energy (lower). The panels are distinct objectives, not one series."
    ),
    "scholarship_source_roles": (
        "Scholarship source roles used to support theory, software, documentation, and binary-analysis context."
    ),
    "scholarship_evidence_matrix": (
        "Scholarship evidence tiers partitioned by implementation stage (implemented, comparison, "
        "roadmap, context)."
    ),
    "claim_evidence_map": (
        "Claim-register evidence coverage: the number of source keys, generated artifacts, and "
        "validation checks linked to each bounded claim."
    ),
    "redteam_risk_register": (
        "RedTeam risk register. Every finding is remediated and guarded by a validation check; "
        "severity reflects the false-certification path, not residual risk."
    ),
    "artifact_hash_manifest": (
        "Reproducibility manifest coverage by artifact class (tracked bytes). The manifest recomputes "
        "each file's SHA-256 and byte size so stale or replaced artifacts fail validation."
    ),
    "validation_summary": (
        "Analysis validation report: every generated-output oracle that the validation gate enforces."
    ),
    "temporal_beliefs_x86_accumulator": (
        "Forward-filter belief trajectory of the temporal HMM for the x86_accumulator corpus "
        "fixture. Each row is a normalized belief over the five program phases at one observation step."
    ),
    "temporal_beliefs_x86_bitwise": (
        "Forward-filter belief trajectory of the temporal HMM for the x86_bitwise corpus fixture. "
        "Each row is a normalized belief over the five program phases at one observation step."
    ),
    "temporal_beliefs_x86_guard": (
        "Forward-filter belief trajectory of the temporal HMM for the x86_guard corpus fixture. "
        "Each row is a normalized belief over the five program phases at one observation step."
    ),
    "temporal_beliefs_x86_long_accumulator": (
        "Forward-filter belief trajectory of the temporal HMM for the x86_long_accumulator corpus "
        "fixture. Each row is a normalized belief over the five program phases at one observation step."
    ),
    "model_comparison": (
        "Decoder comparison. Left: per-fixture phase-decoding accuracy for sticky, memoryless, and "
        "temporal decoders — the temporal HMM beats the v0.1 sticky filter but not the memoryless "
        "baseline on this short-program corpus. Right: pooled accuracy with 95% Wilson confidence "
        "intervals, making the measured uncertainty around each headline explicit."
    ),
    "active_efe": (
        "Expected-free-energy and belief-entropy trajectory of the greedy active-disassembly policy. "
        "On the short-program corpus the trajectory is honest: the policy is functional but ties the "
        "fixed scan."
    ),
    "calibration_curve": (
        "Reliability diagram of the temporal-HMM forward filter over the labelled corpus: mean "
        "max-posterior confidence per bin versus observed accuracy, with the identity line for "
        "perfect calibration and per-bin step counts. Points below the line indicate overconfidence."
    ),
    "confusion_matrix": (
        "Pooled phase-decoding confusion matrix of the temporal forward filter across all corpus "
        "fixtures. Diagonal cells are correct phase recoveries; off-diagonal mass shows which "
        "phases are confusable (e.g. the exit phase sharing observation categories with earlier "
        "phases)."
    ),
    "ambiguity_ablation": (
        "Three panels of decoder accuracy, segmentation F1, and EFE-active advantage versus "
        "observation-ambiguity blend. Left: per-step decoding degrades as the authored likelihood is "
        "blended toward uniform, and the left-to-right temporal transition structure carries the "
        "information — the temporal line stays high while memoryless and sticky fall, with the "
        "crossover (temporal first beating memoryless) marked. Middle: pooled phase-boundary F1 for "
        "the temporal (smoothed) and memoryless segmentations, showing the same mechanism in the "
        "segmentation dimension. Right: the number of fixtures where the EFE-active policy reaches "
        "confidence in fewer steps than a fixed scan — an advantage that appears only in an "
        "intermediate-ambiguity band, where observations become differentially informative, and is "
        "never worse. Together the panels show all three inference layers (temporal, segmentation, "
        "active) gaining their mechanism exactly under observation ambiguity."
    ),
    "segmentation_ribbons": (
        "Per-fixture phase-segmentation ribbons. For each fixture the top row is the hand-authored "
        "ground-truth phase sequence and the bottom row is the phase sequence decoded from the "
        "smoothed forward-backward beliefs; contiguous same-phase runs share a color so phase "
        "boundaries appear as color changes. The pooled boundary F1 is shown in the title, making "
        "the measured phase segmentation (CLM-015) visual."
    ),
    "corpus_stability": (
        "Leave-one-out corpus stability. Bars are the pooled temporal forward-filter accuracy on "
        "the full corpus and after holding out each single fixture; the dashed line is the pooled "
        "static (sticky) accuracy and the dotted line is chance. The headline temporal result "
        "survives every single-fixture removal, and the annotated span quantifies the small-corpus "
        "fragility explicitly."
    ),
    "phase_recovery": (
        "Per-phase recovery of the temporal forward filter, ordered by F1. Grouped bars show recall "
        "and F1 for each of the five program phases (worst phase highlighted); the macro F1 is in "
        "the title. The setup and accumulator_update phases are fully recovered, while the exit (and "
        "especially guard_check) phase is observation-underdetermined — quantifying the recovery "
        "limitation the manuscript cites qualitatively."
    ),
    "decoder_calibration": (
        "Calibration of the temporal, static (sticky), and memoryless decoders, showing ECE and "
        "Brier for each (ordered by ECE). On this corpus calibration tracks accuracy: the memoryless "
        "decoder is best calibrated, the temporal model middle, and the sticky static filter worst — "
        "an honest statement that the temporal model is neither the most accurate nor the best "
        "calibrated."
    ),
}


def write_figure_captions(output_dir: Path) -> Path:
    """Write ``output/figures/README.md`` with one caption per generated figure."""

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = ["# Figure Captions", "", "| Figure | Caption |", "| --- | --- |"]
    for stem, caption in FIGURE_CAPTIONS.items():
        path = figure_dir / f"{stem}.png"
        status = "present" if path.exists() else "missing"
        rows.append(f"| `{stem}.png` ({status}) | {caption} |")
    caption_path = figure_dir / "README.md"
    caption_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return caption_path
