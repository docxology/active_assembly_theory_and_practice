from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import TypedDict

import numpy as np

from .gnn_model import CompiledModel, GNNError
from .gnn_parser import compile_gnn


class UpstreamReceipt(TypedDict):
    repository: str
    revision: str
    import_hashes: dict[str, str]
    source_sha256: str
    model_digest: str
    parser: str


@dataclass(frozen=True)
class QualifiedGNN:
    model: CompiledModel
    receipt: UpstreamReceipt


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise GNNError(f"GNN_UPSTREAM_SOURCE: {result.stderr.strip()}")
    return result.stdout.strip()


def compile_upstream(path: Path, gnn_root: Path) -> QualifiedGNN:
    """Consume upstream parser tensors, admitting only the bounded numeric contract."""
    model = compile_gnn(path)
    root = gnn_root.resolve()
    origin = _git(root, "remote", "get-url", "origin")
    if origin.removesuffix(".git") not in (
        "https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation",
        "git@github.com:ActiveInferenceInstitute/Generalized_Notation_Notation",
    ):
        raise GNNError("GNN_UPSTREAM_SOURCE: unexpected repository")
    revision = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain", "--untracked-files=all", "--", "src/gnn"):
        raise GNNError("GNN_UPSTREAM_SOURCE: parser source is dirty")
    worker = Path(__file__).with_name("gnn_upstream_worker.py")
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(worker), str(root), str(path.resolve())],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        raise GNNError(f"GNN_UPSTREAM_PARSE: {result.stderr.strip()}")
    try:
        parsed = json.loads(result.stdout)
        if (
            parsed["success"] is not True
            or parsed["errors"]
            or parsed["warnings"]
            or parsed["source_sha256"] != model.source_sha256
            or parsed["name"] != model.name
        ):
            raise GNNError("GNN_UPSTREAM_PARSE: incomplete or inconsistent parse")
        parameters = {p["name"]: p["value"] for p in parsed["parameters"]}
        variables = {v["name"]: v["dimensions"] for v in parsed["variables"]}
        expected = {
            f"{group}{i}": array
            for group in ("A", "B", "C", "D")
            for i, array in enumerate(getattr(model, group))
        } | {"E": model.E}
        if (
            set(parameters) != set(expected)
            or set(variables) != set(expected)
            or len(parameters) != len(parsed["parameters"])
            or len(variables) != len(parsed["variables"])
        ):
            raise GNNError("GNN_UPSTREAM_TENSOR: variable inventory mismatch")
        arrays = {}
        for name, array in expected.items():
            value = np.asarray(parameters[name])
            if (
                value.dtype.kind not in "fiu"
                or variables[name] != list(array.shape)
                or value.shape != array.shape
                or not np.array_equal(value, array)
            ):
                raise GNNError(f"GNN_UPSTREAM_TENSOR: {name} mismatch")
            arrays[name] = value
        hashes = parsed["import_hashes"]
        if "src/gnn/parsers/markdown_parser.py" not in hashes:
            raise GNNError("GNN_UPSTREAM_SOURCE: parser not witnessed")
        for relative, digest in hashes.items():
            source = (root / relative).resolve()
            source.relative_to(root / "src/gnn")
            committed = subprocess.run(
                ["git", "-C", str(root), "show", f"{revision}:{relative}"],
                capture_output=True,
                timeout=30,
                check=True,
            ).stdout
            if (
                hashlib.sha256(committed).hexdigest() != digest
                or hashlib.sha256(source.read_bytes()).hexdigest() != digest
            ):
                raise GNNError("GNN_UPSTREAM_SOURCE: imported source drift")
        if _git(root, "rev-parse", "HEAD") != revision:
            raise GNNError("GNN_UPSTREAM_SOURCE: revision changed during parse")
        compiled = replace(
            model,
            A=tuple(arrays[f"A{i}"] for i in range(len(model.A))),
            B=tuple(arrays[f"B{i}"] for i in range(len(model.B))),
            C=tuple(arrays[f"C{i}"] for i in range(len(model.C))),
            D=tuple(arrays[f"D{i}"] for i in range(len(model.D))),
            E=arrays["E"],
        )
        receipt = UpstreamReceipt(
            repository=origin,
            revision=revision,
            import_hashes=hashes,
            source_sha256=model.source_sha256,
            model_digest=compiled.tensor_digest,
            parser="gnn.parsers.markdown_parser.MarkdownGNNParser",
        )
        return QualifiedGNN(compiled, receipt)
    except (KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, GNNError):
            raise
        raise GNNError("GNN_UPSTREAM_PARSE: invalid parser output") from exc
