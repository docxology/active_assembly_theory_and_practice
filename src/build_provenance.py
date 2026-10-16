"""Build-lineage receipt for source, generated, hydrated, and rendered outputs."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

from .artifact_contract import write_json_artifact
from .manuscript_registry import _git_diff_sha256, _git_value, _template_root

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_PROVENANCE_PATH = "output/data/build_provenance.json"
SCHEMA_VERSION = "1.1"

_TEMPLATE_IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "infrastructure/core",
    "infrastructure/rendering",
    "scripts/pipeline/stage_03_render.py",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_paths(project_root: Path, paths: list[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        digest = _sha256(path)
        if digest:
            hashes[str(path.relative_to(project_root))] = digest
    return dict(sorted(hashes.items()))


def _template_implementation_hashes(template_root: Path | None) -> dict[str, str]:
    """Hash the template code that can affect hydration and rendering.

    The sibling template checkout may carry unrelated dirty documentation or
    generated state.  The complete worktree status/diff is retained as an
    audit snapshot, while strict freshness compares this functional surface so
    a renderer change cannot be hidden by unrelated template churn.
    """

    if template_root is None:
        return {}
    paths: list[Path] = []
    for relative in _TEMPLATE_IMPLEMENTATION_PATHS:
        candidate = template_root / relative
        if candidate.is_dir():
            paths.extend(
                path
                for path in sorted(candidate.rglob("*"))
                if path.suffix in {".py", ".lua"} and "__pycache__" not in path.parts
            )
        elif candidate.is_file():
            paths.append(candidate)
    return _hash_paths(template_root, paths)


def build_provenance_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return a current, self-excluding lineage payload."""

    from .reproducibility import MANIFEST_TARGETS

    source_paths = [
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        project_root / "formal" / "lean-toolchain",
        project_root / "formal" / "lakefile.toml",
        project_root / "formal" / "lake-manifest.json",
        project_root / "formal" / "README.md",
        project_root / "manuscript" / "config.yaml",
        project_root / "manuscript" / "references.bib",
        project_root / "manuscript" / "assets" / "active_assembly_cover.png",
        *sorted((project_root / "configs/gnn").rglob("*.md")),
        project_root / "julia/ActiveAssemblyRxInfer/Project.toml",
        project_root / "julia/ActiveAssemblyRxInfer/Manifest.toml",
        project_root / "julia/ActiveAssemblyRxInfer/src/ActiveAssemblyRxInfer.jl",
        *sorted(
            path
            for path in (project_root / "src").rglob("*")
            if path.suffix in {".py", ".lua"} and "__pycache__" not in path.parts
        ),
        *sorted(
            path
            for path in (project_root / "formal").rglob("*.lean")
            if ".lake" not in path.parts
        ),
        *sorted(
            path
            for path in (project_root / "manuscript").glob("*.md")
            if path.name not in {"AGENTS.md", "README.md", "SYNTAX.md"}
        ),
    ]
    generated_paths = [project_root / target.path for target in MANIFEST_TARGETS] + [
        project_root / "output/data/manuscript_variables.json",
        project_root / "output/data/manuscript_variable_registry.json",
        project_root / "output/data/figure_registry.json",
        project_root / "output/figures/figure_registry.json",
        project_root / "output/data/validation_report.json",
        project_root / "output/data/reproducibility_manifest.json",
        project_root / "output/data/reproducibility_manifest.md",
        project_root / "output/data/docx_render_receipt.json",
        project_root / "output/data/evaluation_report.json",
        project_root / "output/data/active_selection.json",
        project_root / "output/data/temporal_inference.json",
        project_root / "output/data/symbolic_execution.json",
        project_root / "output/data/cfg_evidence.json",
        project_root / "output/data/held_out_model.json",
        project_root / "output/figures/README.md",
    ]
    generated_paths.extend(
        path
        for path in sorted((project_root / "output/gnn").rglob("*"))
        if path.is_file() and path.suffix in {".json", ".md", ".png", ".svg", ".pdf"}
    )
    generated_paths.extend(
        path
        for path in sorted((project_root / "output/manuscript").glob("*"))
        if path.is_file()
    )
    generated_paths.extend(sorted((project_root / "output/figures").glob("*.png")))
    rendered_paths = [
        *sorted((project_root / "output/pdf").glob("*.pdf")),
        *sorted((project_root / "output/web").glob("*.html")),
        *sorted((project_root / "output/docx").glob("*.docx")),
        *sorted((project_root / "output/slides").glob("*.pdf")),
    ]
    template_root = _template_root(project_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "target_repository": {
            "root": str(project_root),
            "revision": _git_value(project_root, "rev-parse", "HEAD"),
            "worktree_status": _git_value(project_root, "status", "--short"),
            "worktree_diff_sha256": _git_diff_sha256(project_root),
        },
        "template_repository": {
            "root": str(template_root) if template_root else "unavailable",
            "revision": (
                _git_value(template_root, "rev-parse", "HEAD")
                if template_root is not None
                else "unavailable"
            ),
            "worktree_status": (
                _git_value(template_root, "status", "--short")
                if template_root is not None
                else "unavailable"
            ),
            "worktree_diff_sha256": (
                _git_diff_sha256(template_root) if template_root is not None else None
            ),
            "implementation_hashes": _template_implementation_hashes(template_root),
            "contract": (
                "infrastructure.rendering.manuscript_injection."
                "write_resolved_manuscript_tree"
            ),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "lineage": {
            "source_hashes": _hash_paths(project_root, source_paths),
            "generated_hashes": _hash_paths(project_root, generated_paths),
            "rendered_hashes": _hash_paths(project_root, rendered_paths),
            "rendered_surface_status": "present" if rendered_paths else "not_rendered",
        },
        "excluded_self_referential_path": BUILD_PROVENANCE_PATH,
    }


def write_build_provenance(project_root: Path = PROJECT_ROOT) -> Path:
    """Write the current build-lineage receipt and return its path."""

    path = project_root / BUILD_PROVENANCE_PATH
    write_json_artifact(
        path,
        build_provenance_payload(project_root),
        project_root=project_root,
        producer_path=Path(__file__),
    )
    return path


def verify_build_provenance(
    project_root: Path = PROJECT_ROOT, *, strict: bool = True
) -> list[str]:
    """Return current-lineage errors for the stored receipt.

    ``strict=False`` is used while the validation report is itself being
    regenerated, because that report and its summary figure are downstream of
    the check. The final post-render receipt is always verified strictly.
    """

    path = project_root / BUILD_PROVENANCE_PATH
    if not path.exists():
        return [f"missing {BUILD_PROVENANCE_PATH}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [str(exc)]
    current = build_provenance_payload(project_root)
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema version mismatch")
    if (
        payload.get("target_repository", {}).get("revision")
        != current["target_repository"]["revision"]
    ):
        errors.append("target revision mismatch")
    if (
        payload.get("target_repository", {}).get("worktree_diff_sha256")
        != current["target_repository"]["worktree_diff_sha256"]
    ):
        errors.append("target worktree diff mismatch")
    if payload.get("template_repository", {}).get("revision") != current[
        "template_repository"
    ].get("revision"):
        errors.append("template revision mismatch")
    recorded_template_hashes = payload.get("template_repository", {}).get(
        "implementation_hashes", {}
    )
    current_template_hashes = current["template_repository"].get(
        "implementation_hashes", {}
    )
    if recorded_template_hashes != current_template_hashes:
        errors.append("template implementation hashes mismatch")
    sections = (
        ("source_hashes", "generated_hashes", "rendered_hashes")
        if strict
        else ("source_hashes",)
    )
    for section in sections:
        recorded = payload.get("lineage", {}).get(section, {})
        current_hashes = current["lineage"][section]
        if set(recorded) != set(current_hashes):
            errors.append(
                f"{section} path-set mismatch: "
                f"missing={sorted(set(current_hashes) - set(recorded))} "
                f"extra={sorted(set(recorded) - set(current_hashes))}"
            )
        for relative_path, digest in recorded.items():
            if current_hashes.get(relative_path) != digest:
                errors.append(f"{section} stale={relative_path}")
    return errors
