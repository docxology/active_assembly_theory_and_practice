"""Literal GNN parameter bindings for the project's bounded inference adapters.

Fitting remains a backend operation. Before inference, its numeric result is
specialized into a GNN instance, parsed by the real configured upstream parser,
and validated against the source model's tensor rules. Adapters consume the
parsed values, never a parallel unparsed parameter dictionary.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping

import numpy as np

from .gnn_model import CompiledModel, GNNError
from .gnn_upstream import compile_upstream

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT / "configs/gnn/inference"
MAX_ELEMENTS = 2_000_000


@dataclass(frozen=True)
class NumericBinding:
    model_name: str
    parameters: Mapping[str, np.ndarray]
    source: str
    definition_sha256: str
    instance_sha256: str
    graph: Mapping[str, Any]
    context: Mapping[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "dialect": "active-assembly-numeric-gnn-1",
            "model_name": self.model_name,
            "parameters": {
                key: value.tolist() for key, value in self.parameters.items()
            },
            "gnn_source": self.source,
            "source_sha256": self.instance_sha256,
            "definition_sha256": self.definition_sha256,
            "context": dict(self.context),
        }


_capture: ContextVar[dict[str, NumericBinding] | None] = ContextVar(
    "gnn_numeric_capture", default=None
)


@lru_cache(maxsize=128)
def _canonical_model(path: str, source_hash: str, root: str) -> CompiledModel:
    model = compile_upstream(Path(path), Path(root)).model
    if model.source_sha256 != source_hash:
        raise GNNError("GNN_BINDING_SOURCE: model changed during admission")
    return model


def canonical_model(path: Path) -> CompiledModel:
    root = os.environ.get("GNN_ROOT")
    if not root:
        raise GNNError("GNN_UPSTREAM_REQUIRED: configure GNN_ROOT for all inference")
    return _canonical_model(
        str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest(), root
    )


def upstream_parser() -> Any:
    configured = os.environ.get("GNN_ROOT")
    if not configured:
        raise GNNError("GNN_UPSTREAM_REQUIRED: configure GNN_ROOT for all inference")
    root = Path(configured).resolve()
    return _upstream_parser(str(root))


@lru_cache(maxsize=1)
def _upstream_parser(root_path: str) -> Any:
    root = Path(root_path)
    # Establish the actual repository, clean committed parser and imported
    # source hashes through the existing isolated qualification boundary.
    compile_upstream(ROOT / "configs/gnn/passive_phase.md", root)
    sys.path.insert(0, str(root / "src"))
    module = import_module("gnn.parsers.markdown_parser")
    if (
        module.__file__ is None
        or Path(module.__file__).resolve()
        != root / "src/gnn/parsers/markdown_parser.py"
    ):
        raise GNNError("GNN_UPSTREAM_SOURCE: foreign parser module")
    return module.MarkdownGNNParser()


def _sections(source: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    name = ""
    for line in source.splitlines():
        if line.startswith("## "):
            name = line[3:].strip()
            if name in sections:
                raise GNNError("GNN_BINDING_SECTION: duplicate section")
            sections[name] = []
        elif line.strip() and not line.startswith("#"):
            if not name:
                raise GNNError("GNN_BINDING_SECTION: content before section")
            sections[name].append(line)
    required = {
        "GNNSection",
        "GNNVersionAndFlags",
        "ModelName",
        "StateSpaceBlock",
        "Connections",
        "InitialParameterization",
        "ModelAnnotation",
    }
    if set(sections) != required:
        raise GNNError("GNN_BINDING_SECTION: unsupported model definition")
    values = {key: "\n".join(value) for key, value in sections.items()}
    if (
        values["GNNSection"] != "ProbabilisticModel"
        or values["GNNVersionAndFlags"] != "GNN v1.1"
    ):
        raise GNNError("GNN_BINDING_SECTION: unsupported declaration or version")
    return values


def _numeric(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise GNNError(
            f"GNN_BINDING_TYPE: {name} must be real numeric; "
            f"received {type(value).__name__}: {str(value)[:120]}"
        )
    array = np.array(raw, dtype=float, copy=True)
    if array.ndim == 0:
        array = array.reshape(1)
    if not 1 <= array.ndim <= 4 or not 1 <= array.size <= MAX_ELEMENTS:
        raise GNNError(f"GNN_BINDING_SIZE: {name} exceeds bounded tensor profile")
    if not np.isfinite(array).all():
        raise GNNError(f"GNN_BINDING_FINITE: {name}")
    return array


def _validate_rule(name: str, array: np.ndarray, rule: Mapping[str, Any]) -> None:
    if array.ndim != rule["rank"]:
        raise GNNError(f"GNN_BINDING_SHAPE: {name} rank")
    if "positive" in rule and np.any(array <= 0):
        raise GNNError(f"GNN_BINDING_POSITIVE: {name}")
    if "nonnegative" in rule and np.any(array < 0):
        raise GNNError(f"GNN_BINDING_NONNEGATIVE: {name}")
    if "stochastic_axis" in rule:
        axis = rule["stochastic_axis"]
        if np.any(array < 0) or not np.allclose(
            array.sum(axis=axis), 1, atol=1e-10, rtol=0
        ):
            raise GNNError(f"GNN_BINDING_DISTRIBUTION: {name}")


def _validate_axes(name: str, p: Mapping[str, np.ndarray]) -> None:
    if name in {"phase_hmm", "identity_phase"}:
        n = p["D"].size
        valid = p["A"].shape[1] == n
        if name == "phase_hmm":
            valid = valid and p["B"].shape == (n, n)
    elif name == "gaussian_event_code":
        valid = all(value.shape == (1,) for value in p.values())
    elif name in {"factorized_fields", "hierarchical_fields"}:
        f, _, n = p["likelihood"].shape
        valid = (
            p["B"].shape == (n, n)
            and p["D"].shape == (n,)
            and p["divisors"].shape == (f,)
            and p["group_divisor"].shape == (1,)
        )
    elif name == "finite_sensor":
        valid = p["cost"].shape == (1,)
    elif name == "retrospective_ordering":
        valid = p["A"].shape[0] == p["preferences"].size
    elif name == "quotient_logistic":
        valid = p["weights"].shape == p["center"].shape == p["scale"].shape and p[
            "bias"
        ].shape == (1,)
    elif name == "quotient_markov":
        c, v = p["initial"].shape
        valid = (
            c == 2 and p["transition"].shape == (c, v, v) and p["prior"].shape == (c,)
        )
    elif name == "quotient_recurrent":
        h, _ = p["wxh"].shape
        valid = (
            p["whh"].shape == (h, h)
            and p["why"].shape == (h,)
            and p["bias"].shape == (1,)
        )
    elif name == "platt_calibration":
        valid = all(value.shape == (1,) for value in p.values())
    elif name == "linear_regression":
        valid = p["coefficients"].ndim == 1
    else:
        raise GNNError(f"GNN_BINDING_ADAPTER: unsupported {name}")
    if not valid:
        raise GNNError(f"GNN_BINDING_SHAPE: {name} incompatible axes")


def _literal(array: np.ndarray) -> str:
    def number(value: float) -> str:
        literal = repr(float(value))
        if "e" in literal and "." not in literal.split("e")[0]:
            mantissa, exponent = literal.split("e")
            return mantissa + ".0e" + exponent
        return literal

    if array.ndim == 1:
        return "[" + ",".join(number(value) for value in array) + "]"

    def row(value: np.ndarray) -> str:
        if value.ndim == 1:
            return "(" + ",".join(number(v) for v in value) + ")"
        return "(" + ",".join(row(v) for v in value) + ")"

    return "{" + ",".join(row(value) for value in array) + "}"


def semantic_graph(parsed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variables": [
            {
                key: variable[key]
                for key in ("name", "dimensions", "var_type", "data_type")
            }
            for variable in parsed["variables"]
        ],
        "connections": [
            {
                key: connection[key]
                for key in ("source_variables", "target_variables", "connection_type")
            }
            for connection in parsed["connections"]
        ],
    }


@lru_cache(maxsize=512)
def _compile_instance(source: str, definition_hash: str) -> NumericBinding:
    sections = _sections(source)
    annotation = json.loads(sections["ModelAnnotation"])
    result = upstream_parser().parse_string(source)
    if not result.success or result.errors or result.warnings:
        raise GNNError(f"GNN_UPSTREAM_PARSE: {result.errors}; {result.warnings}")
    parsed = result.model.to_dict()
    parameters = {}
    for parameter in result.model.parameters:
        if parameter.name in parameters:
            raise GNNError("GNN_BINDING_PARAMETER: duplicate parsed parameter")
        parameters[parameter.name] = _numeric(parameter.value, parameter.name)
    rules = annotation["parameters"]
    if set(parameters) != set(rules):
        raise GNNError("GNN_BINDING_PARAMETER: parser/definition inventory mismatch")
    variables = {v.name: tuple(v.dimensions) for v in result.model.variables}
    if len(variables) != len(result.model.variables):
        raise GNNError("GNN_BINDING_VARIABLE: duplicate parsed variable")
    if set(variables) != set(parameters) | set(annotation["nodes"]):
        raise GNNError("GNN_BINDING_VARIABLE: unsupported variable inventory")
    for name, value in parameters.items():
        if variables.get(name) != value.shape:
            raise GNNError(f"GNN_BINDING_SHAPE: {name} declaration mismatch")
        _validate_rule(name, value, rules[name])
        value.setflags(write=False)
    _validate_axes(result.model.model_name, parameters)
    for name, rule in annotation["nodes"].items():
        expected = (
            tuple(parameters[rule["from"]].shape[i] for i in rule["axes"])
            if "from" in rule
            else tuple(rule["shape"])
        )
        if variables[name] != expected:
            raise GNNError(f"GNN_BINDING_SHAPE: {name} graph node mismatch")
    return NumericBinding(
        model_name=result.model.model_name,
        parameters=parameters,
        source=source,
        definition_sha256=definition_hash,
        instance_sha256=hashlib.sha256(source.encode()).hexdigest(),
        graph=semantic_graph(parsed),
        context=annotation.get("context", {}),
    )


def bind_numeric(
    model_name: str,
    parameters: Mapping[str, object] | None = None,
    *,
    context: Mapping[str, Any] | None = None,
    definition_path: Path | None = None,
) -> NumericBinding:
    """Admit a specialized model and return its upstream-parsed parameters.

    Scalars have the explicit one-element-vector representation used by these
    GNN definitions. No distribution is normalized or repaired at admission.
    Dynamic tensor dimensions specialize a fixed, declared adapter family;
    they do not enable arbitrary equations or universal GNN code generation.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*", model_name):
        raise GNNError("GNN_BINDING_NAME: invalid model name")
    upstream_parser()
    path = definition_path or DEFINITIONS / f"{model_name}.md"
    raw = path.read_bytes()
    sections = _sections(raw.decode())
    annotation = json.loads(sections["ModelAnnotation"])
    if sections["ModelName"] != model_name or annotation["adapter"] != model_name:
        raise GNNError("GNN_BINDING_NAME: definition/adapter mismatch")
    defaults = {}
    for line in sections["InitialParameterization"].splitlines():
        name, literal = line.split("=", 1)
        defaults[name.strip()] = json.loads(literal)
    supplied = dict(parameters) if parameters is not None else defaults
    rules = annotation["parameters"]
    if set(supplied) != set(rules):
        raise GNNError(f"GNN_BINDING_PARAMETER: {model_name} inventory mismatch")
    arrays = {name: _numeric(value, name) for name, value in supplied.items()}
    for name, array in arrays.items():
        _validate_rule(name, array, rules[name])
    _validate_axes(model_name, arrays)
    declarations = []
    for name, array in arrays.items():
        dimensions_text = ",".join(map(str, array.shape))
        declarations.append(f"{name}[{dimensions_text}, type=float]")
    for name, rule in annotation["nodes"].items():
        shape = (
            [arrays[rule["from"]].shape[i] for i in rule["axes"]]
            if "from" in rule
            else rule["shape"]
        )
        declarations.append(f"{name}[{','.join(map(str, shape))}, type=float]")
    annotation["context"] = json.loads(json.dumps(context or {}, allow_nan=False))
    sections["ModelAnnotation"] = json.dumps(annotation, separators=(",", ":"))
    sections["StateSpaceBlock"] = "\n".join(declarations)
    sections["InitialParameterization"] = "\n".join(
        f"{name}={_literal(value)}" for name, value in arrays.items()
    )
    source = "\n".join(f"## {key}\n{value}" for key, value in sections.items()) + "\n"
    if len(source.encode()) > 8 * 1024 * 1024:
        raise GNNError("GNN_BINDING_SIZE: instance exceeds 8 MiB")
    binding = _compile_instance(source, hashlib.sha256(raw).hexdigest())
    for name, value in arrays.items():
        if not np.array_equal(binding.parameters[name], value):
            raise GNNError(f"GNN_BINDING_VALUE: upstream changed {name}")
    recording = _capture.get()
    if recording is not None:
        recording[binding.instance_sha256] = binding
    return binding


def admit_numeric_source(source: str) -> NumericBinding:
    if not isinstance(source, str) or len(source.encode()) > 8 * 1024 * 1024:
        raise GNNError("GNN_BINDING_SIZE: invalid persisted source")
    upstream_parser()
    sections = _sections(source)
    name = sections["ModelName"]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise GNNError("GNN_BINDING_NAME: invalid persisted model name")
    definition = (DEFINITIONS / f"{name}.md").read_bytes()
    expected = _sections(definition.decode())
    for section in ("GNNSection", "GNNVersionAndFlags", "ModelName", "Connections"):
        if sections[section] != expected[section]:
            raise GNNError(f"GNN_BINDING_STRUCTURE: unsupported {section}")
    annotation = json.loads(sections["ModelAnnotation"])
    context = annotation.pop("context", {})
    if annotation != json.loads(expected["ModelAnnotation"]) or not isinstance(
        context, dict
    ):
        raise GNNError("GNN_BINDING_STRUCTURE: unsupported adapter annotation")
    binding = _compile_instance(source, hashlib.sha256(definition).hexdigest())
    recording = _capture.get()
    if recording is not None:
        recording[binding.instance_sha256] = binding
    return binding


def restore_numeric_sources(sources: object) -> set[str]:
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(s, str) for s in sources)
    ):
        raise GNNError("GNN_BINDING_SOURCE: missing runtime model sources")
    return {admit_numeric_source(source).model_name for source in sources}


@contextmanager
def capture_numeric_bindings() -> Iterator[dict[str, NumericBinding]]:
    """Collect actual admitted runtime instances without writes by validators."""
    bindings: dict[str, NumericBinding] = {}
    parent = _capture.get()
    token = _capture.set(bindings)
    try:
        yield bindings
    finally:
        _capture.reset(token)
        if parent is not None:
            parent.update(bindings)
