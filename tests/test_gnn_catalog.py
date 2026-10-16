import hashlib
import json
import shutil

import pytest

from src.gnn_catalog import (
    ARTIFACT,
    validate_inference_catalog,
    write_inference_catalog,
)
from src.gnn_numeric import DEFINITIONS, bind_numeric, capture_numeric_bindings


@pytest.fixture(scope="module")
def catalog_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("gnn_catalog")
    with capture_numeric_bindings() as bindings:
        for path in sorted(DEFINITIONS.glob("*.md")):
            bind_numeric(path.stem)
    write_inference_catalog(root, bindings)
    return root


def test_every_model_has_actual_upstream_and_readable_graphs(catalog_root):
    assert validate_inference_catalog(catalog_root) == []
    payload = json.loads((catalog_root / ARTIFACT).read_text())
    assert payload["model_profile_count"] == len(payload["models"])
    assert {p.stem for p in DEFINITIONS.glob("*.md")}.issubset(
        {r["name"] for r in payload["models"]}
    )
    assert len(
        list((catalog_root / "output/gnn/inference").glob("*/*_network_graph.png"))
    ) == len(payload["models"])
    assert (catalog_root / "output/gnn/inference/README.md").is_file()


@pytest.mark.parametrize(
    "defect",
    [
        "missing_profile",
        "duplicate_profile",
        "profile_count",
        "instance_count",
        "source_hash",
        "upstream_revision",
        "missing_parser_renderer_witness",
        "definition_association",
        "missing_instances",
        "instance_bytes",
        "malformed_distribution_with_fresh_hash",
        "graph_topology",
        "rendered_bytes",
        "missing_render_format",
        "render_path_association",
        "schema",
        "status",
        "repository",
        "adapter_sources",
        "parameter_names",
        "duplicate_instance",
        "canonical_instance",
        "canonical_parameters",
    ],
)
def test_corrupted_catalog_is_rejected_without_modifying_inputs(
    catalog_root, tmp_path, defect
):
    root = tmp_path / "candidate"
    shutil.copytree(catalog_root, root)
    path = root / ARTIFACT
    payload = json.loads(path.read_text())
    row = next(r for r in payload["models"] if r["name"] == "phase_hmm")
    if defect == "missing_profile":
        payload["models"].pop()
    elif defect == "duplicate_profile":
        payload["models"][-1] = dict(payload["models"][0])
    elif defect == "profile_count":
        payload["model_profile_count"] += 1
    elif defect == "instance_count":
        payload["numeric_instance_count"] = 0
    elif defect == "source_hash":
        payload["source_hashes"]["src/gnn_numeric.py"] = "0" * 64
    elif defect == "upstream_revision":
        payload["upstream"]["revision"] = "0" * 40
    elif defect == "missing_parser_renderer_witness":
        payload["upstream"]["import_hashes"] = {}
    elif defect == "definition_association":
        row["definition"] = "configs/gnn/inference/identity_phase.md"
    elif defect == "missing_instances":
        row["instance_paths"] = []
        row["instance_sha256"] = []
    elif defect in {"instance_bytes", "malformed_distribution_with_fresh_hash"}:
        instance = root / row["instance_paths"][0]
        source = instance.read_text()
        if defect == "instance_bytes":
            source += "\n## Unsupported\ncontent\n"
        else:
            source = source.replace("D=[0.5,0.5]", "D=[-0.5,1.5]")
            assert "D=[-0.5,1.5]" in source
            row["instance_sha256"][0] = hashlib.sha256(source.encode()).hexdigest()
        instance.write_text(source)
    elif defect == "graph_topology":
        row["graph"]["connections"] = []
    elif defect == "rendered_bytes":
        rendered = root / next(iter(row["rendered_artifacts"]))
        rendered.write_bytes(b"invalid image")
    elif defect == "missing_render_format":
        row["rendered_artifacts"].pop(next(iter(row["rendered_artifacts"])))
    elif defect == "render_path_association":
        row["rendered_artifacts"] = {"../outside.png": "0" * 64}
    elif defect == "schema":
        payload["schema_version"] = "99"
    elif defect == "status":
        payload["status"] = "unchecked"
    elif defect == "repository":
        payload["upstream"]["repository"] = "https://example.org/foreign"
    elif defect == "adapter_sources":
        row["adapter_sources"] = ["src/foreign.py"]
    elif defect == "parameter_names":
        row["parameter_names"] = []
    elif defect == "duplicate_instance":
        row["instance_paths"] *= 2
        row["instance_sha256"] *= 2
        payload["numeric_instance_count"] += 1
    else:
        canonical = next(r for r in payload["models"] if r["name"] == "passive_phase")
        if defect == "canonical_instance":
            canonical["instance_paths"] = ["configs/gnn/fixture_sensor.md"]
        else:
            canonical["parameter_names"] = []
    path.write_text(json.dumps(payload))
    before = {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert validate_inference_catalog(root)
    after = {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_writer_rejects_empty_runtime_coverage(tmp_path):
    with pytest.raises(ValueError, match="no runtime binding"):
        write_inference_catalog(tmp_path, {})
