"""Strict, bounded parser for the documented numerical GNN subset."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from .gnn_model import CompiledModel, GNNError

SECTIONS = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Metadata",
    "ModelAnnotation",
)
META = {
    "state_labels",
    "factors",
    "modalities",
    "actions",
    "outcomes",
    "A_dependencies",
    "horizon",
    "gamma",
    "seed",
}


def compile_gnn(path: Path) -> CompiledModel:
    """Compile literal tensors without evaluating Python or symbolic content."""
    if path.stat().st_size > 65536:
        raise GNNError(f"GNN-E103: {path}: text exceeds 64 KiB")
    raw = path.read_bytes()
    if len(raw) > 65536:
        raise GNNError(f"GNN-E103: {path}: text exceeds 64 KiB")
    sections: dict[str, list[tuple[int, str]]] = {}
    section = ""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GNNError("GNN-E100: invalid UTF-8") from exc
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            section = line[3:].strip()
            if section not in SECTIONS or section in sections:
                raise GNNError(
                    f"GNN-E100: {path}:{lineno}: unsupported/duplicate section"
                )
            sections[section] = []
        elif line.strip() and not line.lstrip().startswith("#"):
            if not section:
                raise GNNError(f"GNN-E100: {path}:{lineno}: content outside section")
            sections[section].append((lineno, line.split("#", 1)[0].strip()))
    if any(s not in sections for s in SECTIONS[:7]):
        raise GNNError(f"GNN-E001: {path}: missing required section")
    if list(sections) != [s for s in SECTIONS if s in sections]:
        raise GNNError(f"GNN-E100: {path}: section order")
    if [v for _, v in sections["GNNSection"]] != ["ActInfPOMDP"]:
        raise GNNError("GNN-E100: only ActInfPOMDP is supported")
    if [v for _, v in sections["GNNVersionAndFlags"]] != ["GNN v1.1"]:
        raise GNNError("GNN-E100: only GNN v1.1 is supported")
    shapes = {}
    for lineno, line in sections["StateSpaceBlock"]:
        match = re.fullmatch(r"([A-E]\d*)\[([0-9, ]+),\s*type=float\]", line)
        if not match:
            raise GNNError(f"GNN-E100: {path}:{lineno}: unsupported declaration")
        name = match[1]
        dims = tuple(int(v.strip()) for v in match[2].split(","))
        if len(dims) > 4 or any(d < 1 for d in dims) or math.prod(dims) > 4096:
            raise GNNError("GNN-E103: declaration resource limit")
        if name in shapes:
            raise GNNError("GNN-E004: duplicate declaration")
        shapes[name] = dims
    connections = set()
    for lineno, line in sections["Connections"]:
        match = re.fullmatch(r"([A-E]\d*)>([A-E]\d*)", line)
        if not match:
            raise GNNError(f"GNN-E005: {path}:{lineno}: unsupported connection")
        if any(n not in shapes for n in match.groups()):
            raise GNNError("GNN-E003: unknown connection variable")
        if match.groups() in connections:
            raise GNNError("GNN-E004: duplicate connection")
        connections.add(match.groups())
    arrays = {}
    for lineno, line in sections["InitialParameterization"]:
        match = re.fullmatch(r"([A-E]\d*)\s*=\s*(\[[0-9eE+.,\[\] \-]+\])", line)
        if not match:
            raise GNNError(f"GNN-E100: {path}:{lineno}: unsupported numeric literal")
        name = match[1]
        if name not in shapes or name in arrays:
            raise GNNError("GNN-E102: unknown/duplicate parameter")
        try:
            value = json.loads(match[2])
            a = np.asarray(value, dtype=float)
        except (ValueError, TypeError, RecursionError) as exc:
            raise GNNError("GNN-E100: invalid numeric tensor") from exc
        if a.shape != shapes[name]:
            raise GNNError(f"GNN-E002: {name}: parameter shape mismatch")
        arrays[name] = a
    metadata = {}
    for lineno, line in sections["Metadata"]:
        key, sep, val = line.partition(":")
        if not sep or key not in META or key in metadata:
            raise GNNError(f"GNN-E100: {path}:{lineno}: unsupported/duplicate metadata")
        try:
            metadata[key] = json.loads(val)
        except ValueError as exc:
            raise GNNError("GNN-E100: invalid metadata literal") from exc
    if set(metadata) != META or set(arrays) != set(shapes):
        raise GNNError("GNN-E102: incomplete executable contract")
    try:
        f, m = len(metadata["factors"]), len(metadata["modalities"])
        expected_connections = {
            (f"D{d}", f"A{i}")
            for i, deps in enumerate(metadata["A_dependencies"])
            for d in deps
        }
        if connections != expected_connections:
            raise GNNError(
                "GNN-E102: connections must declare exactly the likelihood factor dependencies"
            )
        expected = {
            f"{g}{i}"
            for g, n in [("A", m), ("B", f), ("C", m), ("D", f)]
            for i in range(n)
        } | {"E"}
        if set(arrays) != expected:
            raise GNNError("GNN-E102: unexpected tensor inventory")
        return CompiledModel(
            name=" ".join(v for _, v in sections["ModelName"]),
            factors=tuple(metadata["factors"]),
            modalities=tuple(metadata["modalities"]),
            state_labels=tuple(tuple(v) for v in metadata["state_labels"]),
            actions=tuple(tuple(a) for a in metadata["actions"]),
            outcomes=tuple(tuple(a) for a in metadata["outcomes"]),
            A=tuple(arrays[f"A{i}"] for i in range(m)),
            B=tuple(arrays[f"B{i}"] for i in range(f)),
            C=tuple(arrays[f"C{i}"] for i in range(m)),
            D=tuple(arrays[f"D{i}"] for i in range(f)),
            E=arrays["E"],
            A_dependencies=tuple(tuple(a) for a in metadata["A_dependencies"]),
            horizon=metadata["horizon"],
            gamma=metadata["gamma"],
            seed=metadata["seed"],
            source_sha256=hashlib.sha256(raw).hexdigest(),
            annotations="\n".join(v for _, v in sections.get("ModelAnnotation", [])),
        )
    except (TypeError, IndexError, KeyError, ValueError) as exc:
        if isinstance(exc, GNNError):
            raise
        raise GNNError("GNN-E102: invalid metadata types") from exc
