from __future__ import annotations

from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import matplotlib.pyplot as plt
import networkx as nx

from .artifact_contract import write_json_artifact
from .gnn_model import GNNError
from .gnn_numeric import (
    DEFINITIONS,
    NumericBinding,
    admit_numeric_source,
    canonical_model,
    semantic_graph,
    upstream_parser,
)
from .gnn_upstream import _git
from .provenance import sha256_path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = "output/data/inference_model_catalog.json"
ADAPTERS = {
    "phase_hmm": ("temporal_model.py", "model_fitting.py", "parameter_learning.py"),
    "identity_phase": ("active_inference_bridge.py",),
    "gaussian_event_code": ("rxinfer_bridge.py",),
    "factorized_fields": ("factorized_model.py",),
    "hierarchical_fields": ("factorized_model.py",),
    "finite_sensor": ("information_policy.py", "action_sensing.py", "sensor_costs.py"),
    "retrospective_ordering": ("active_selection.py",),
    "quotient_logistic": ("quotient_classifier.py",),
    "quotient_markov": ("quotient_classifier.py",),
    "quotient_recurrent": ("quotient_classifier.py",),
    "platt_calibration": ("quotient_classifier.py",),
    "linear_regression": ("sensor_costs.py",),
}


def _source_hashes() -> dict[str, str]:
    paths = [
        ROOT / "src/gnn_numeric.py",
        ROOT / "src/gnn_catalog.py",
        ROOT / "src/gnn_upstream.py",
        ROOT / "src/gnn_upstream_worker.py",
        ROOT / "src/gnn_parser.py",
        ROOT / "src/gnn_model.py",
        ROOT / "src/pymdp_runtime.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "julia/ActiveAssemblyRxInfer/src/ActiveAssemblyRxInfer.jl",
        *(ROOT / "src" / name for names in ADAPTERS.values() for name in names),
        *sorted((ROOT / "configs/gnn").rglob("*.md")),
    ]
    return {
        str(path.relative_to(ROOT)): sha256_path(path) for path in sorted(set(paths))
    }


def _upstream_sources(root: Path) -> dict[str, object]:
    revision = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain", "--untracked-files=all", "--", "src/gnn"):
        raise GNNError("GNN_UPSTREAM_SOURCE: imported source tree is dirty")
    paths = sorted(
        {
            Path(module.__file__).resolve()
            for name, module in tuple(sys.modules.items())
            if (name == "gnn" or name.startswith("gnn."))
            and module.__file__ is not None
        }
    )
    names = [str(path.relative_to(root)) for path in paths]
    tree = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "-z", revision, "--", "src/gnn"],
        capture_output=True,
        check=True,
    ).stdout
    committed = {}
    for entry in tree.split(b"\0"):
        if entry:
            info, name = entry.split(b"\t", 1)
            committed[name.decode()] = info.split()[-1].decode()
    blobs = subprocess.run(
        ["git", "-C", str(root), "hash-object", "--", *names],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if len(blobs) != len(names) or any(
        committed.get(name) != blob for name, blob in zip(names, blobs, strict=True)
    ):
        raise GNNError(
            "GNN_UPSTREAM_SOURCE: imported bytes differ from committed sources"
        )
    return {
        "revision": revision,
        "repository": _git(root, "remote", "get-url", "origin"),
        "import_hashes": {
            name: sha256_path(path) for name, path in zip(names, paths, strict=True)
        },
    }


def _readable_graph(
    graph: Mapping[str, Any], parameters: set[str], name: str, directory: Path
) -> list[Path]:
    variables = graph["variables"]
    network: nx.DiGraph[str] = nx.DiGraph()
    for variable in variables:
        network.add_node(variable["name"], dimensions=variable["dimensions"])
    for connection in graph["connections"]:
        for source in connection["source_variables"]:
            for target in connection["target_variables"]:
                if source not in network or target not in network:
                    raise GNNError("GNN_RENDER_STRUCTURE: unknown connection endpoint")
                network.add_edge(source, target)
    if not nx.is_directed_acyclic_graph(network):
        raise GNNError("GNN_RENDER_STRUCTURE: unsupported cyclic adapter topology")
    layers = list(nx.topological_generations(network))
    for layer, nodes in enumerate(layers):
        for node in nodes:
            network.nodes[node]["layer"] = layer
    position = nx.multipartite_layout(network, subset_key="layer", align="vertical")
    height = max(4.2, 0.9 * max(map(len, layers)) + 1.5)
    with plt.rc_context(
        {"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"}
    ):
        fig, ax = plt.subplots(figsize=(max(7.5, len(layers) * 1.9), height))
        nx.draw_networkx_edges(
            network,
            position,
            ax=ax,
            arrows=True,
            arrowsize=22,
            node_size=4800,
            edge_color="#52606D",
            width=1.7,
        )
        nx.draw_networkx_nodes(
            network,
            position,
            ax=ax,
            node_size=4800,
            node_color=[
                "#BFDCEC" if node in parameters else "#D9EBCB" for node in network
            ],
            edgecolors="#243447",
            linewidths=1.2,
        )
        labels = {
            node: node
            + "\n["
            + " × ".join(map(str, network.nodes[node]["dimensions"]))
            + "]"
            for node in network
        }
        nx.draw_networkx_labels(
            network, position, labels, ax=ax, font_size=12, font_color="#243447"
        )
        ax.set_title(name.replace("_", " "), fontsize=15, pad=18)
        ax.margins(0.2)
        ax.axis("off")
        fig.text(
            0.03,
            0.025,
            "Declared GNN topology; blue: parameter tensors, green: state/data nodes.\nDimensions belong to this actual admitted instance; arrows do not establish scientific validity.",
            fontsize=10,
        )
        fig.subplots_adjust(top=0.85, bottom=0.14, left=0.05, right=0.95)
        paths = [
            directory / f"{name}_readable.{suffix}" for suffix in ("png", "svg", "pdf")
        ]
        try:
            for path in paths:
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)
    return paths


def write_inference_catalog(
    project_root: Path, bindings: Mapping[str, NumericBinding]
) -> Path:
    expected = {path.stem for path in DEFINITIONS.glob("*.md")}
    if expected != set(ADAPTERS):
        raise GNNError("GNN_CATALOG_COVERAGE: definition/adapter inventory differs")
    grouped: dict[str, list[NumericBinding]] = {name: [] for name in expected}
    for _, binding in sorted(bindings.items()):
        if binding.model_name not in grouped:
            raise GNNError("GNN_CATALOG_COVERAGE: unknown executed adapter")
        grouped[binding.model_name].append(binding)
    missing = sorted(name for name, values in grouped.items() if not values)
    if missing:
        raise GNNError(f"GNN_CATALOG_COVERAGE: no runtime binding for {missing}")
    directory = project_root / "output/gnn/inference"
    instances = directory / "instances"
    instances.mkdir(parents=True, exist_ok=True)
    render = import_module(
        "gnn.visualization.graph.network_visualizations"
    ).generate_network_visualizations
    rows: list[dict[str, Any]] = []
    for name, values in sorted(grouped.items()):
        paths = []
        for binding in values:
            path = instances / f"{binding.instance_sha256}.md"
            if path.exists() and path.read_text() != binding.source:
                raise GNNError("GNN_CATALOG_SOURCE: immutable instance was changed")
            if not path.exists():
                path.write_text(binding.source)
            paths.append(path)
        representative = values[0]
        rows.append(
            {
                "name": name,
                "kind": "project_numeric_adapter",
                "definition": str((DEFINITIONS / f"{name}.md").relative_to(ROOT)),
                "definition_sha256": representative.definition_sha256,
                "instance_paths": [
                    str(path.relative_to(project_root)) for path in paths
                ],
                "instance_sha256": [binding.instance_sha256 for binding in values],
                "graph": dict(representative.graph),
                "parameter_names": sorted(representative.parameters),
                "adapter_sources": [f"src/{source}" for source in ADAPTERS[name]],
            }
        )
    for name in ("passive_phase", "fixture_sensor"):
        path = ROOT / f"configs/gnn/{name}.md"
        canonical_model(path)
        result = upstream_parser().parse_string(path.read_text())
        if not result.success or result.errors or result.warnings:
            raise GNNError("GNN_CATALOG_SOURCE: canonical parse failed")
        model = result.model.to_dict()
        rows.append(
            {
                "name": name,
                "kind": "canonical_pomdp",
                "definition": str(path.relative_to(ROOT)),
                "definition_sha256": sha256_path(path),
                "instance_paths": [str(path.relative_to(ROOT))],
                "instance_sha256": [sha256_path(path)],
                "graph": semantic_graph(model),
                "parameter_names": [
                    parameter.name for parameter in result.model.parameters
                ],
                "adapter_sources": ["src/gnn_experiments.py", "src/pymdp_runtime.py"],
            }
        )
    for row in rows:
        target = directory / row["name"]
        target.mkdir(exist_ok=True)
        render(row["graph"], target, row["name"])
        native = [
            target / f"{row['name']}_network_graph.png",
            target / f"{row['name']}_network_stats.json",
        ]
        if any(not path.is_file() or not path.stat().st_size for path in native):
            raise GNNError(f"GNN_RENDER_FAILED: upstream did not render {row['name']}")
        readable = _readable_graph(
            row["graph"], set(row["parameter_names"]), row["name"], target
        )
        row["rendered_artifacts"] = {
            str(path.relative_to(project_root)): sha256_path(path)
            for path in [*native, *readable]
        }
    upstream = _upstream_sources(Path(os.environ["GNN_ROOT"]).resolve())
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "validated_bounded",
        "model_profile_count": len(rows),
        "numeric_instance_count": len(bindings),
        "models": rows,
        "source_hashes": _source_hashes(),
        "upstream": upstream,
        "boundary": "Actual upstream parsing and topology rendering; explicit project adapters, not universal upstream code generation. Statistical comparators remain comparators.",
    }
    destination = project_root / ARTIFACT
    write_json_artifact(
        destination,
        payload,
        project_root=project_root,
        producer_path=Path(__file__),
        input_paths=[ROOT / name for name in _source_hashes()],
    )
    readme: list[str] = [
        "# Executed GNN inference model profiles",
        "",
        payload["boundary"],
        "",
        "Every numeric instance below was admitted during the producer or restored from a source-bound fitted-model cache. Graphs show a representative actual instance of each profile. Fitted instances share architecture; their parameters and training contexts remain separate.",
        "",
    ]
    for row in rows:
        name = row["name"]
        readme.extend(
            [
                f"## {name}",
                "",
                f"Kind: {row['kind']}. Instances: {len(row['instance_paths'])}.",
                f"Definition: `{row['definition']}`; SHA256 `{row['definition_sha256']}`.",
                "",
                f"![Declared GNN topology for {name}]({name}/{name}_readable.png)",
                "",
                f"[Upstream rendering]({name}/{name}_network_graph.png) · [SVG]({name}/{name}_readable.svg) · [PDF]({name}/{name}_readable.pdf)",
                "",
            ]
        )
    (directory / "README.md").write_text("\n".join(readme))
    return destination


def validate_inference_catalog(project_root: Path) -> list[str]:
    errors = []
    try:
        payload = json.loads((project_root / ARTIFACT).read_text())
        rows = payload["models"]
        expected = set(ADAPTERS) | {"passive_phase", "fixture_sensor"}
        if (
            len(rows) != len(expected)
            or {row["name"] for row in rows} != expected
            or payload["model_profile_count"] != len(expected)
            or payload["source_hashes"] != _source_hashes()
        ):
            raise GNNError("GNN_CATALOG_COVERAGE: model or source inventory mismatch")
        root = Path(os.environ["GNN_ROOT"]).resolve()
        if payload["upstream"]["revision"] != _git(root, "rev-parse", "HEAD"):
            raise GNNError("GNN_CATALOG_SOURCE: upstream revision mismatch")
        required_imports = {
            "src/gnn/parsers/markdown_parser.py",
            "src/gnn/visualization/graph/network_visualizations.py",
        }
        if not required_imports.issubset(payload["upstream"]["import_hashes"]):
            raise GNNError("GNN_CATALOG_SOURCE: parser/renderer witness is missing")
        if _git(
            root, "status", "--porcelain", "--untracked-files=all", "--", "src/gnn"
        ):
            raise GNNError("GNN_CATALOG_SOURCE: upstream source tree changed")
        for relative, digest in payload["upstream"]["import_hashes"].items():
            path = (root / relative).resolve()
            path.relative_to(root / "src/gnn")
            if sha256_path(path) != digest:
                raise GNNError("GNN_CATALOG_SOURCE: imported source bytes changed")
        instance_count = 0
        for row in rows:
            canonical = row["name"] in {"passive_phase", "fixture_sensor"}
            expected_definition = (
                f"configs/gnn/{row['name']}.md"
                if canonical
                else f"configs/gnn/inference/{row['name']}.md"
            )
            if row["definition"] != expected_definition or row["kind"] != (
                "canonical_pomdp" if canonical else "project_numeric_adapter"
            ):
                raise GNNError(
                    "GNN_CATALOG_SOURCE: definition/adapter association mismatch"
                )
            definition = ROOT / row["definition"]
            if sha256_path(definition) != row["definition_sha256"]:
                raise GNNError(
                    f"GNN_CATALOG_SOURCE: definition changed for {row['name']}"
                )
            if not row["instance_paths"] or len(row["instance_paths"]) != len(
                row["instance_sha256"]
            ):
                raise GNNError("GNN_CATALOG_SOURCE: missing runtime instances")
            for relative, digest in zip(
                row["instance_paths"], row["instance_sha256"], strict=True
            ):
                path = (
                    ROOT if row["kind"] == "canonical_pomdp" else project_root
                ) / relative
                if not canonical:
                    path.resolve().relative_to(
                        project_root.resolve() / "output/gnn/inference/instances"
                    )
                if sha256_path(path) != digest:
                    raise GNNError("GNN_CATALOG_SOURCE: instance bytes changed")
                if row["kind"] != "canonical_pomdp":
                    binding = admit_numeric_source(path.read_text())
                    if (
                        binding.model_name != row["name"]
                        or binding.definition_sha256 != row["definition_sha256"]
                    ):
                        raise GNNError("GNN_CATALOG_SOURCE: instance profile mismatch")
                    if (
                        relative == row["instance_paths"][0]
                        and dict(binding.graph) != row["graph"]
                    ):
                        raise GNNError(
                            "GNN_CATALOG_RENDER: graph differs from GNN source"
                        )
                    instance_count += 1
            if canonical:
                result = upstream_parser().parse_string(definition.read_text())
                if (
                    not result.success
                    or semantic_graph(result.model.to_dict()) != row["graph"]
                ):
                    raise GNNError(
                        "GNN_CATALOG_RENDER: canonical graph differs from source"
                    )
            expected_graphs = {
                f"output/gnn/inference/{row['name']}/{row['name']}_{suffix}"
                for suffix in (
                    "network_graph.png",
                    "network_stats.json",
                    "readable.png",
                    "readable.svg",
                    "readable.pdf",
                )
            }
            if set(row["rendered_artifacts"]) != expected_graphs:
                raise GNNError("GNN_CATALOG_RENDER: incomplete graph formats")
            for relative, digest in row["rendered_artifacts"].items():
                path = (project_root / relative).resolve()
                path.relative_to(project_root.resolve() / "output/gnn/inference")
                if sha256_path(path) != digest:
                    raise GNNError("GNN_CATALOG_RENDER: rendered graph bytes changed")
        if instance_count != payload["numeric_instance_count"]:
            raise GNNError("GNN_CATALOG_COVERAGE: runtime instance count mismatch")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        errors.append(str(exc))
    return errors
