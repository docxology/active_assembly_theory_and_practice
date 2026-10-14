"""Project-owned DOCX cover projection through the canonical Pandoc renderer."""

from importlib import import_module
from pathlib import Path
import json
import os
import subprocess
import tempfile

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
    composition = import_module("infrastructure.rendering.manuscript_composition")
    try:
        recorded = composition.read_manuscript_composition(
            project_root / "output/reports/manuscript_composition.json"
        )
        inputs = sorted((project_root / "output/manuscript").glob("[0-9][0-9]_*.md"))
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
    """Export current hydrated sources through the registered native template.

    Registration and environment preparation are explicit prerequisites. Never
    select a different checkout or silently hydrate under the template runtime.
    """
    project_root = Path(project_root).resolve()
    variables = json.loads(
        (project_root / "output/data/manuscript_variables.json").read_text()
    )
    validate_hydrated_tree(
        project_root / "output/manuscript", variables, project_root=project_root
    )
    template_root = resolve_template_root(project_root)
    project_name = project_root.name
    working_slot = template_root / "projects/working" / project_name
    if not working_slot.is_dir() or working_slot.resolve() != project_root:
        raise RuntimeError(
            f"register the canonical project working slot first: {working_slot}"
        )
    runtime = os.environ.get("TEMPLATE_PYTHON")
    executable = (
        Path(runtime).expanduser().resolve()
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
    subprocess.run(
        [
            str(executable),
            str(template_root / "scripts/pipeline/stage_03_render.py"),
            "--project",
            f"working/{project_name}",
            "--skip-manuscript-hydration",
        ],
        cwd=template_root,
        check=True,
        stdin=subprocess.DEVNULL,
    )
