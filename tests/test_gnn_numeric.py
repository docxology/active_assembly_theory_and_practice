import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.gnn_model import GNNError
from src.gnn_numeric import (
    DEFINITIONS,
    admit_numeric_source,
    bind_numeric,
    capture_numeric_bindings,
    restore_numeric_sources,
)
from src.quotient_classifier import _apply_platt, _predict_logistic, _rnn_forward
from src.temporal_model import TemporalGenerativeModel, forward_filter


@pytest.mark.parametrize("name", sorted(p.stem for p in DEFINITIONS.glob("*.md")))
def test_actual_upstream_model_instances_preserve_every_parameter(name):
    binding = bind_numeric(name)
    parsed = binding.to_dict()
    assert parsed["source_sha256"] == binding.instance_sha256
    assert binding.graph["variables"] and binding.graph["connections"]
    for value in binding.parameters.values():
        assert np.isfinite(value).all()
        assert not value.flags.writeable


def test_large_recurrent_matrix_uses_supported_upstream_literal_without_value_loss():
    rng = np.random.default_rng(11)
    supplied = {
        "wxh": rng.normal(0, 0.05, (32, 4)),
        "whh": rng.normal(0, 0.05, (32, 32)),
        "why": rng.normal(0, 0.05, 32),
        "bias": -0.125,
    }
    binding = bind_numeric("quotient_recurrent", supplied)
    for name, expected in supplied.items():
        np.testing.assert_array_equal(binding.parameters[name], np.atleast_1d(expected))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, True, "one"])
def test_invalid_literal_parameter_is_rejected(bad):
    with pytest.raises(GNNError):
        bind_numeric("platt_calibration", {"slope": bad, "intercept": 0})


@pytest.mark.parametrize(
    "defect", ["negative", "mass", "shape", "rank", "missing", "extra"]
)
def test_malformed_categorical_model_is_not_repaired(defect):
    parameters = {k: v.copy() for k, v in bind_numeric("phase_hmm").parameters.items()}
    if defect == "negative":
        parameters["D"] = np.array([-0.1, 1.1])
    elif defect == "mass":
        parameters["A"] *= 0.5
    elif defect == "shape":
        parameters["D"] = np.full(3, 1 / 3)
    elif defect == "rank":
        parameters["B"] = np.ones((2, 2, 1)) / 2
    elif defect == "missing":
        parameters.pop("D")
    else:
        parameters["unused"] = np.array([1.0])
    with pytest.raises(GNNError):
        bind_numeric("phase_hmm", parameters)


def test_gnn_bound_hmm_matches_independent_two_state_enumeration():
    model = TemporalGenerativeModel(
        A=np.array([[0.9, 0.2], [0.1, 0.8]]),
        B=np.array([[0.7, 0.3], [0.3, 0.7]]),
        D=np.array([0.6, 0.4]),
        observation_labels=("zero", "one"),
        hidden_state_labels=("left", "right"),
    )
    with capture_numeric_bindings() as recorded:
        result = forward_filter([0, 1], model)
    joints = np.array(
        [
            model.D[first] * model.A[0, first] * model.B[last, first] * model.A[1, last]
            for first in range(2)
            for last in range(2)
        ]
    ).reshape(2, 2)
    np.testing.assert_allclose(
        result.final_posterior, joints.sum(axis=0) / joints.sum()
    )
    assert result.log_evidence == pytest.approx(np.log(joints.sum()))
    assert {b.model_name for b in recorded.values()} == {"phase_hmm"}


def test_logistic_and_calibration_use_nondefault_gnn_parameters():
    x = np.array([[1.0, 2.0], [4.0, -1.0]])
    weights = np.array([0.7, -0.9])
    center = np.array([0.2, 0.8])
    scale = np.array([2.0, 3.0])
    with capture_numeric_bindings() as recorded:
        probability = _predict_logistic(x, (weights, 0.25, center, scale))
        calibrated = _apply_platt(probability, 1.7, -0.3)
    logits = ((x - center) / scale) @ weights + 0.25
    np.testing.assert_allclose(probability, 1 / (1 + np.exp(-logits)))
    np.testing.assert_allclose(calibrated, 1 / (1 + np.exp(-(1.7 * logits - 0.3))))
    assert {b.model_name for b in recorded.values()} == {
        "quotient_logistic",
        "platt_calibration",
    }


def test_recurrent_parameters_retain_nonlinear_state_update():
    parameters = bind_numeric(
        "quotient_recurrent",
        {
            "wxh": [[0.2, -0.3]],
            "whh": [[0.5]],
            "why": [0.7],
            "bias": -0.1,
        },
    ).parameters
    probability, _ = _rnn_forward(
        [0, 1],
        parameters["wxh"],
        parameters["whh"],
        parameters["why"],
        float(parameters["bias"][0]),
    )
    final_hidden = np.tanh(-0.3 + 0.5 * np.tanh(0.2))
    assert probability == pytest.approx(1 / (1 + np.exp(-(0.7 * final_hidden - 0.1))))


def test_missing_upstream_configuration_fails_in_real_process(tmp_path):
    env = dict(os.environ)
    env.pop("GNN_ROOT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from src.gnn_numeric import bind_numeric; bind_numeric('gaussian_event_code')",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "GNN_UPSTREAM_REQUIRED" in result.stderr


def test_capture_has_no_output_side_effects_and_restores_outer_scope(tmp_path):
    before = set(tmp_path.iterdir())
    with capture_numeric_bindings() as outer:
        one = bind_numeric("gaussian_event_code")
        with capture_numeric_bindings() as inner:
            two = bind_numeric("platt_calibration")
        bind_numeric("gaussian_event_code")
    assert set(outer) == {one.instance_sha256, two.instance_sha256}
    assert set(inner) == {two.instance_sha256}
    assert set(tmp_path.iterdir()) == before
    assert json.loads(json.dumps(one.to_dict()))["parameters"]["variance"] == [100.0]


@pytest.mark.parametrize("bad", [[], [1 + 2j], np.zeros((1, 1, 1, 1, 1))])
def test_empty_complex_and_excessive_rank_are_rejected(bad):
    with pytest.raises(GNNError):
        bind_numeric("linear_regression", {"coefficients": bad})


@pytest.mark.parametrize(
    "name,parameter,bad",
    [
        ("gaussian_event_code", "variance", [0]),
        ("gaussian_event_code", "mean", [0, 1]),
        ("finite_sensor", "cost", [1, 2]),
        ("finite_sensor", "cost", [-1]),
        ("quotient_logistic", "center", [0, 1, 2]),
        ("quotient_recurrent", "why", [0, 1, 2]),
        ("quotient_markov", "prior", [1]),
        ("platt_calibration", "slope", [1, 2]),
        ("factorized_fields", "divisors", [1, 2, 3]),
        ("retrospective_ordering", "preferences", [1]),
    ],
)
def test_adapter_axes_and_parameter_domains_are_enforced(name, parameter, bad):
    parameters = dict(bind_numeric(name).parameters)
    parameters[parameter] = bad
    with pytest.raises(GNNError):
        bind_numeric(name, parameters)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_section",
        "leading_content",
        "missing_section",
        "version",
        "model_name",
        "connections",
        "annotation",
        "context",
        "node_shape",
        "parameter_shape",
        "extra_variable",
        "duplicate_variable",
        "missing_parameter",
    ],
)
def test_persisted_sources_cannot_change_the_supported_architecture(defect):
    source = bind_numeric("phase_hmm").source
    if defect == "duplicate_section":
        source += "## ModelName\nphase_hmm\n"
    elif defect == "leading_content":
        source = "unowned content\n" + source
    elif defect == "missing_section":
        source = source.replace("## ModelName\nphase_hmm\n", "")
    elif defect == "version":
        source = source.replace("GNN v1.1", "GNN v9.9")
    elif defect == "model_name":
        source = source.replace("\nphase_hmm\n", "\n../foreign\n")
    elif defect == "connections":
        source = source.replace("D>s_prev", "D>o")
    elif defect == "annotation":
        source = source.replace('"adapter":"phase_hmm"', '"adapter":"foreign"')
    elif defect == "context":
        source = source.replace('"context":{}', '"context":[]')
    elif defect == "node_shape":
        source = source.replace("s_prev[2,", "s_prev[3,")
    elif defect == "parameter_shape":
        source = source.replace("A[2,2,", "A[3,2,")
    elif defect == "extra_variable":
        source = source.replace(
            "## Connections", "extra[2, type=float]\n## Connections"
        )
    elif defect == "duplicate_variable":
        source = source.replace("## Connections", "A[2,2, type=float]\n## Connections")
    else:
        source = source.replace("D=[0.5,0.5]\n", "")
    with pytest.raises(GNNError):
        admit_numeric_source(source)


@pytest.mark.parametrize("sources", [None, [], [1], "source"])
def test_cached_models_require_nonempty_literal_source_capsules(sources):
    with pytest.raises(GNNError, match="GNN_BINDING_SOURCE"):
        restore_numeric_sources(sources)


def test_cached_instance_replays_actual_parameters_and_context():
    original = bind_numeric(
        "platt_calibration",
        {"slope": -0.7, "intercept": 0.2},
        context={"training_fixtures": ["owned_one"]},
    )
    with capture_numeric_bindings() as replay:
        assert restore_numeric_sources([original.source]) == {"platt_calibration"}
    assert replay[original.instance_sha256].context == original.context
    np.testing.assert_array_equal(
        replay[original.instance_sha256].parameters["slope"], [-0.7]
    )
