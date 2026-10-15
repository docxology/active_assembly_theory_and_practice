from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.gnn_parser import compile_gnn
from src.gnn_reference import forward_backward, policy_reference, posterior_reference
from src.pymdp_runtime import PymdpRuntime

ROOT = Path(__file__).resolve().parents[1]


def tiny_model():
    m = compile_gnn(ROOT / "configs/gnn/passive_phase.md")
    return replace(
        m,
        state_labels=(("s0", "s1"),),
        outcomes=(("o0", "o1"),),
        A=(np.array([[0.8, 0.2], [0.2, 0.8]]),),
        B=(np.array([[[0.7], [0.3]], [[0.3], [0.7]]]),),
        C=(np.zeros(2),),
        D=(np.array([0.6, 0.4]),),
    )


def test_independent_worked_bayes_and_smoothing():
    m = tiny_model()
    np.testing.assert_allclose(
        posterior_reference(m, m.D, [0])[0], [6 / 7, 1 / 7], atol=1e-10
    )
    exact = forward_backward(m, [0, 1])
    # Four joint path masses are .0672,.1152,.0048,.0448, total .232.
    np.testing.assert_allclose(
        exact["smoothed"],
        [[0.1824 / 0.232, 0.0496 / 0.232], [0.072 / 0.232, 0.160 / 0.232]],
        atol=1e-10,
    )
    np.testing.assert_allclose(exact["filtered"][0], [6 / 7, 1 / 7], atol=1e-10)
    assert policy_reference(m, m.D)["action"] == [0]


def test_runtime_invalid_and_impossible_observation():
    m = compile_gnn(ROOT / "configs/gnn/fixture_sensor.md")
    runtime = PymdpRuntime(m)
    for observation in ([5, 0], [True, 0], [0]):
        with pytest.raises(ValueError, match="GNN_INVALID_OBSERVATION"):
            runtime.infer(observation)
    with pytest.raises(ValueError, match="GNN_IMPOSSIBLE_OBSERVATION"):
        runtime.infer([4, 0])
    runtime.assert_unchanged()
    with pytest.raises(ValueError, match="GNN_EMPTY"):
        forward_backward(tiny_model(), [])
    with pytest.raises(ValueError, match="GNN_NATIVE_UNSUPPORTED"):
        forward_backward(m, [0])


def test_reference_never_mutates_input_priors():
    m = tiny_model()
    prior = [m.D[0].copy()]
    original = prior[0].copy()
    posterior_reference(m, prior, [0])
    np.testing.assert_array_equal(prior[0], original)
    posterior_reference(m, m.D, [0])


def test_exact_reference_bounds_joint_allocation():
    from src.gnn_reference import joint_distribution

    with pytest.raises(ValueError, match="GNN_REFERENCE_RESOURCE_LIMIT"):
        joint_distribution([np.array([0.5, 0.5]) for _ in range(13)])
