#!/usr/bin/env python3
"""Stage verified v0.1.0 manuscript evidence for the pinned Template replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from infrastructure.rendering.manuscript_injection import (
    substitute_manuscript_text,
    write_resolved_manuscript_tree,
)

REVIEW_SHA256 = "7783a3796c79abfbbcbe705a81313601a54dfe8828bb12efcc592f25d2f7c891"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verified_bytes(archive: ZipFile, files: dict[str, str], name: str) -> bytes:
    data = archive.read(name)
    if hashlib.sha256(data).hexdigest() != files[name]:
        raise ValueError(f"Published evidence hash differs: {name}")
    return data


def prepare(archive_path: Path) -> None:
    """Verify the archive and reuse its immutable scientific data and figures."""
    with archive_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != REVIEW_SHA256:
            raise ValueError("Expected the published v0.1.0 Zenodo review archive")
    source = PROJECT_ROOT / "docs/manuscript"
    sections = sorted(source.glob("[0-9][0-9]_*.md"))
    with ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("RELEASE_MANIFEST.json"))
        files = manifest["files"]
        for path in [*sections, source / "preamble.md", source / "references.bib"]:
            if path.read_bytes() != verified_bytes(
                archive, files, f"manuscript/{path.name}"
            ):
                raise ValueError(
                    f"Canonical manuscript differs from the published text: {path.name}"
                )
        variable_file = "output/data/manuscript_variables.json"
        variables = {
            name: value.replace("../output/", "../../output/").replace(
                "manuscript/assets/active_assembly_cover.png",
                "docs/manuscript/assets/active_assembly_cover.png",
            )
            for name, value in json.loads(
                verified_bytes(archive, files, variable_file)
            ).items()
            if name.isupper() and isinstance(value, str)
        }
        for path in sections:
            _, missing = substitute_manuscript_text(
                path.read_text(encoding="utf-8"), variables
            )
            if missing:
                raise ValueError(
                    f"{path.name}: unresolved variables {sorted(set(missing))}"
                )
        for name in files:
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix
                not in {".json", ".md", ".png", ".pdf", ".svg", ".html"}
                or not name.startswith(
                    (
                        "output/data/",
                        "output/figures/",
                        "output/rxinfer_categorical/",
                        "output/gnn/inference/",
                    )
                )
            ):
                continue
            target = PROJECT_ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(verified_bytes(archive, files, name))
        registry_path = PROJECT_ROOT / "output/figures/figure_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for figure in registry["figures"]:
            figure["alt_text"] = figure["alt"]
        registry_path.write_text(
            json.dumps(registry, indent=2) + "\n", encoding="utf-8"
        )
    write_resolved_manuscript_tree(PROJECT_ROOT, variables)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    prepare(parser.parse_args().archive)


if __name__ == "__main__":
    main()
