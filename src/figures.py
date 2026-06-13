"""Publication figures for the assembly Active Inference scaffold."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .active_inference_bridge import InferenceResult
from .features import AssemblyObservation
from .generative_model import GenerativeModelSpec

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")


def plot_observation_pipeline(
    observations: list[AssemblyObservation],
    output_path: Path,
) -> Path:
    """Plot instruction observations as a categorical pipeline."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(observations))
    y = [observation.observation_id for observation in observations]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(x, y, marker="o", color=PALETTE[0], linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([obs.mnemonic for obs in observations])
    ax.set_yticks(sorted(set(y)))
    ax.set_yticklabels([observations[y.index(value)].category for value in sorted(set(y))])
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


def plot_rxinfer_free_energy(records: list[dict[str, object]], output_path: Path) -> Path:
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


def plot_package_provenance(provenance_rows: list[dict[str, object]], output_path: Path) -> Path:
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
                1.0 if any(char.isdigit() for char in observation.operand_text) else 0.0,
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
        [f"{observation.index}: {observation.mnemonic}" for observation in observations],
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


def plot_artifact_hash_manifest(rows: list[dict[str, object]], output_path: Path) -> Path:
    """Plot tracked artifact bytes by artifact class."""

    if not rows:
        raise ValueError("artifact manifest rows must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_by_type: dict[str, int] = {}
    count_by_type: dict[str, int] = {}
    missing_by_type: dict[str, int] = {}
    for row in rows:
        artifact_type = str(row["artifact_type"])
        bytes_by_type[artifact_type] = bytes_by_type.get(artifact_type, 0) + int(row["bytes"])
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
    ax.set_xticklabels([label.replace("_", " ") for label in labels], rotation=28, ha="right")
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
    ax.bar(x, values, color=[PALETTE[index % len(PALETTE)] for index in range(len(labels))])
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", " ") for label in labels], rotation=30, ha="right")
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
                ax.text(column_index, row_index, str(value), ha="center", va="center", fontsize=8)
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
    source_counts = [len(claim["support_keys"]) for claim in claims]
    artifact_counts = [len(claim["artifact_paths"]) for claim in claims]
    validation_counts = [len(claim["validation_checks"]) for claim in claims]
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


def plot_redteam_risk_register(findings: list[dict[str, object]], output_path: Path) -> Path:
    """Plot RedTeam finding status by severity."""

    if not findings:
        raise ValueError("RedTeam findings must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{finding['finding_id']} {finding['vector']}".replace("_", " ") for finding in findings]
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
