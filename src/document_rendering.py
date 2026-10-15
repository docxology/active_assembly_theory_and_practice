"""Project-owned DOCX cover projection through the canonical Pandoc renderer."""

from importlib import import_module
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import time

import yaml

from .docx_validation import (
    DOCX_ARTIFACT,
    PROJECT_ROOT,
    HYDRATED_SOURCE_BINDING_PREFIX,
    _configured_cover,
    _hydrated_manuscript_hash,
)
from .template_hydration import (
    load_template_injection,
    resolve_template_root,
    validate_hydrated_tree,
)


def render_docx_with_cover(project_root: Path = PROJECT_ROOT) -> Path:
    """Render the shared manuscript with its configured cover and alternative.

    The template currently adds the cover only to PDF. This project adapter
    uses its same native Office renderer, crossref/citation filters, resource
    paths and metadata. The final cover filter runs after figure numbering so
    the uncaptioned cover cannot change the scientific figure sequence.
    A failed conversion leaves the previous DOCX intact.
    """
    project_root = Path(project_root)
    source = project_root / "output/web/_combined_manuscript.md"
    if not source.is_file():
        raise FileNotFoundError(f"render the shared manuscript first: {source}")
    cover, alt = _configured_cover(project_root)
    if cover is None or not cover.is_file() or not alt:
        raise ValueError(
            "DOCX projection requires a readable configured cover and alternative text"
        )
    if cover.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("DOCX cover must use a supported raster format")
    config_path = project_root / "manuscript/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    load_template_injection(project_root)
    from .project_rendering import manuscript_sections

    composition = import_module("infrastructure.rendering.manuscript_composition")
    try:
        recorded = composition.read_manuscript_composition(
            project_root / "output/reports/manuscript_composition.json"
        )
        inputs = manuscript_sections(project_root / "output/manuscript")
        current = composition.build_manuscript_composition(
            project_root, recorded.project, inputs, source, algorithm=recorded.algorithm
        )
        if recorded.input_root_kind != "hydrated" or current != recorded:
            raise ValueError("shared manuscript composition is stale")
    except (OSError, ValueError) as exc:
        raise ValueError(
            "render the current hydrated manuscript through the template before DOCX projection"
        ) from exc
    combined_pandoc_args = import_module(
        "infrastructure.rendering._pandoc_args"
    ).combined_pandoc_args
    render_docx = import_module("infrastructure.rendering.docx_renderer").render_docx
    office_pandoc_metadata = import_module(
        "infrastructure.rendering.office_native_projection"
    ).office_pandoc_metadata

    output = project_root / DOCX_ARTIFACT
    output.parent.mkdir(parents=True, exist_ok=True)
    bibliography = project_root / "manuscript/references.bib"
    with tempfile.TemporaryDirectory(
        prefix="cover-render-", dir=output.parent
    ) as staging:
        metadata = Path(staging) / "metadata.yaml"
        metadata.write_text(
            yaml.safe_dump(office_pandoc_metadata(config), allow_unicode=True)
        )
        args = combined_pandoc_args(
            [
                project_root / "output/manuscript",
                project_root / "output/figures",
                project_root / "output",
            ],
            [bibliography] if bibliography.is_file() else [],
            edition="DOCX",
        )
        args.extend(
            [
                f"--metadata-file={metadata}",
                f"--metadata=description:{HYDRATED_SOURCE_BINDING_PREFIX}{_hydrated_manuscript_hash(project_root)}",
                f"--metadata=assembly-figure-root:{(project_root / 'output/figures').resolve()}",
                f"--metadata=assembly-cover-path:{cover.resolve()}",
                f"--metadata=assembly-cover-alt:{alt}",
                "--lua-filter",
                str(Path(__file__).with_name("docx_cover.lua")),
            ]
        )
        staged = Path(staging) / "covered.docx"
        render_docx(source, staged, extra_args=args)
        staged.replace(output)
    return output


def render_manuscript(project_root: Path = PROJECT_ROOT) -> None:
    project_root = Path(project_root).resolve()
    variables_path = project_root / "output/data/manuscript_variables.json"
    variables_bytes = variables_path.read_bytes()
    variables = json.loads(variables_bytes)
    validate_hydrated_tree(
        project_root / "output/manuscript", variables, project_root=project_root
    )
    template_root = resolve_template_root(project_root)
    runtime = os.environ.get("TEMPLATE_PYTHON")
    executable = (
        Path(runtime).expanduser().absolute()
        if runtime
        else (
            template_root
            / ".venv"
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
    )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(
            "prepare the template Python environment or set TEMPLATE_PYTHON: "
            f"{executable}"
        )
    from .project_rendering import _hashes, manuscript_sections

    sources = sorted(
        path for path in (project_root / "manuscript").glob("*") if path.is_file()
    )
    hydrated = sorted(
        path
        for path in (project_root / "output/manuscript").glob("*")
        if path.is_file()
    )
    inputs = {
        "source_hashes": _hashes(sources, project_root),
        "hydrated_hashes": _hashes(hydrated, project_root),
    }
    expected = [
        project_root / "output/pdf" / f"{Path(DOCX_ARTIFACT).stem}.pdf",
        project_root / "output/web/index.html",
        project_root / DOCX_ARTIFACT,
        *(
            project_root / "output/slides" / f"{path.stem}_slides.pdf"
            for path in manuscript_sections(project_root / "output/manuscript")
        ),
    ]
    previous = {
        path: path.stat().st_mtime_ns if path.is_file() else None for path in expected
    }
    report_path = project_root / "output/reports/project_render.json"
    previous_report = report_path.stat().st_mtime_ns if report_path.is_file() else None
    started = time.time_ns()
    subprocess.run(
        [
            str(executable),
            "-I",
            "-B",
            str(
                Path(__file__).resolve().parents[1] / "scripts/render_project_worker.py"
            ),
            "--project-root",
            str(project_root),
            "--skip-manuscript-hydration",
        ],
        cwd=template_root,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    from .artifact_contract import write_json_artifact

    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or report_path.stat().st_mtime_ns < started
        or report_path.stat().st_mtime_ns == previous_report
    ):
        raise RuntimeError("RENDER_REPORT_NOT_FRESH")
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict) or report.get("project_root") != str(project_root):
        raise RuntimeError("RENDER_PROJECT_ROOT_MISMATCH")
    if (
        report.get("template_root") != str(template_root)
        or report.get("formats") != ["pdf", "html", "docx", "slides"]
        or report.get("skip_manuscript_hydration") is not True
        or report.get("variables_sha256") != hashlib.sha256(variables_bytes).hexdigest()
        or variables_path.read_bytes() != variables_bytes
        or any(report.get(key) != value for key, value in inputs.items())
        or _hashes(sources, project_root) != inputs["source_hashes"]
        or _hashes(hydrated, project_root) != inputs["hydrated_hashes"]
    ):
        raise RuntimeError("RENDER_REPORT_INPUT_MISMATCH")
    for path in expected:
        if (
            not path.resolve().is_relative_to(project_root)
            or not path.is_file()
            or not path.stat().st_size
        ):
            raise RuntimeError("RENDER_OUTPUT_MISSING")
        if (
            path.stat().st_mtime_ns < started
            or path.stat().st_mtime_ns == previous[path]
        ):
            raise RuntimeError("RENDER_OUTPUT_NOT_FRESH")
    if report.get("rendered_hashes") != _hashes(expected, project_root):
        raise RuntimeError("RENDER_REPORT_OUTPUT_MISMATCH")
    validate_hydrated_tree(
        project_root / "output/manuscript", variables, project_root=project_root
    )
    write_json_artifact(
        project_root / "output/data/project_render_receipt.json",
        report,
        project_root=project_root,
        producer_path=Path(__file__),
        input_paths=[report_path],
    )
