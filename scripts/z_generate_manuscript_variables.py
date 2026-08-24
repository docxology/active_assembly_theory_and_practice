"""Generate manuscript variables for template rendering."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REEXEC_ENV = "ACTIVE_ASSEMBLY_PROJECT_UV"
TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
SOURCE_EXCLUDES = {"AGENTS.md", "README.md", "SYNTAX.md"}


def _resolve_template_root() -> Path:
    """Resolve the configured template checkout without selecting a duplicate."""

    explicit = os.environ.get("TEMPLATE_REPO_ROOT")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not (
            candidate / "infrastructure/rendering/manuscript_injection.py"
        ).is_file():
            raise RuntimeError(
                "TEMPLATE_REPO_ROOT does not contain the template rendering "
                f"implementation: {candidate}"
            )
        return candidate

    candidates: list[Path] = []
    for parent in (PROJECT_ROOT.parent, *PROJECT_ROOT.parents):
        candidates.append(parent / "template")
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "infrastructure/rendering/manuscript_injection.py").is_file():
            return resolved
    raise RuntimeError(
        "could not locate the template renderer; set TEMPLATE_REPO_ROOT to "
        "the configured docxology/template checkout"
    )


def _load_template_injection():
    template_root = _resolve_template_root()
    if str(template_root) not in sys.path:
        sys.path.insert(0, str(template_root))
    from infrastructure.rendering.manuscript_injection import (  # type: ignore[import-untyped]
        write_resolved_manuscript_tree,
    )

    return write_resolved_manuscript_tree


def _validate_hydrated_tree(output_dir: Path, variables: dict[str, str]) -> None:
    """Fail closed if template hydration omits files or changes substitution."""

    source_dir = PROJECT_ROOT / "manuscript"
    expected = {
        path.name
        for path in source_dir.glob("*.md")
        if path.name not in SOURCE_EXCLUDES
    }
    observed = {path.name for path in output_dir.glob("*.md")}
    unresolved = {
        path.name
        for path in output_dir.glob("*.md")
        if TOKEN_RE.search(path.read_text(encoding="utf-8"))
    }
    mismatched: list[str] = []
    for source_path in source_dir.glob("*.md"):
        if source_path.name in SOURCE_EXCLUDES:
            continue
        expected_text = TOKEN_RE.sub(
            lambda match: str(variables[match.group(1)]),
            source_path.read_text(encoding="utf-8"),
        )
        hydrated_path = output_dir / source_path.name
        if (
            hydrated_path.is_file()
            and hydrated_path.read_text(encoding="utf-8") != expected_text
        ):
            mismatched.append(source_path.name)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra or unresolved or mismatched:
        raise RuntimeError(
            "template manuscript hydration contract failed: "
            f"missing={missing} extra={extra} unresolved={sorted(unresolved)} "
            f"mismatched={mismatched}"
        )


def _missing_project_dependencies() -> list[str]:
    return [
        package
        for package in ("capstone", "pymdp", "unicorn")
        if importlib.util.find_spec(package) is None
    ]


def _reexec_in_project_environment_if_needed() -> None:
    missing = _missing_project_dependencies()
    if not missing or os.environ.get(REEXEC_ENV) == "1":
        return

    env = dict(os.environ)
    env[REEXEC_ENV] = "1"
    command = [
        "uv",
        "run",
        "--project",
        str(PROJECT_ROOT),
        "python",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command, env=env))


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _reexec_in_project_environment_if_needed()
    from src.manuscript_variables import generate_variables
    from src.manuscript_registry import write_variable_registry
    from src.build_provenance import write_build_provenance
    from src.validation import write_validation_report

    variables = generate_variables()
    write_resolved_manuscript_tree = _load_template_injection()
    out_dir = write_resolved_manuscript_tree(PROJECT_ROOT, variables)
    _validate_hydrated_tree(out_dir, variables)

    # Refresh lineage before the first validation pass. A source-only change
    # can alter the worktree hash without changing the validation counts; in
    # that case validating first would reject a stale receipt and the loop
    # below would break before reaching its old refresh branch.
    write_build_provenance(PROJECT_ROOT)

    # Hydration changes one validation oracle from "not hydrated yet" to an
    # actual rendered-tree check. Synchronize the injected validation summary
    # with that final report, then rehydrate so the manuscript reports the same
    # counts that the gate just inspected.
    previous_docx_mode = os.environ.get("ACTIVE_ASSEMBLY_DOCX_PRE_RENDER")
    os.environ["ACTIVE_ASSEMBLY_DOCX_PRE_RENDER"] = "1"
    try:
        for _ in range(3):
            report = write_validation_report()
            validation_values = {
                "VALIDATION_PASS_COUNT": str(report.summary["passed"]),
                "VALIDATION_FAIL_COUNT": str(report.summary["failed"]),
                "VALIDATION_TOTAL_COUNT": str(report.summary["total"]),
                "VALIDATION_ALL_PASSED": str(report.summary["all_passed"]).lower(),
                "VALIDATION_REPORT": "../output/data/validation_report.json",
                "VALIDATION_FIGURE": "../output/figures/validation_summary.png",
            }
            if all(
                variables.get(key) == value for key, value in validation_values.items()
            ):
                break
            variables.update(validation_values)
            variables_path = PROJECT_ROOT / "output/data/manuscript_variables.json"
            variables_path.write_text(
                json.dumps(variables, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            write_variable_registry(variables, PROJECT_ROOT)
            out_dir = write_resolved_manuscript_tree(PROJECT_ROOT, variables)
            _validate_hydrated_tree(out_dir, variables)
            write_build_provenance(PROJECT_ROOT)
    finally:
        if previous_docx_mode is None:
            os.environ.pop("ACTIVE_ASSEMBLY_DOCX_PRE_RENDER", None)
        else:
            os.environ["ACTIVE_ASSEMBLY_DOCX_PRE_RENDER"] = previous_docx_mode
    if not report.summary["all_passed"]:
        failed = [check.name for check in report.checks if not check.passed]
        raise RuntimeError(f"analysis validation failed after hydration: {failed}")
    write_variable_registry(variables, PROJECT_ROOT)
    write_build_provenance(PROJECT_ROOT)
    for key in sorted(variables):
        print(f"{key}={variables[key]}")
    print(f"resolved_manuscript={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
