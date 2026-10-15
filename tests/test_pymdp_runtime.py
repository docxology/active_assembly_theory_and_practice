from dataclasses import replace
from pathlib import Path

import numpy as np

from src.fixture_sensor_environment import FixtureSensorEnvironment
from src.gnn_parser import compile_gnn
from src.gnn_reference import policy_reference
from src.pymdp_runtime import run_pymdp

ROOT = Path(__file__).resolve().parents[1]


def test_real_agent_oracle_actions_and_withheld_invariance():
    m = compile_gnn(ROOT / "configs/gnn/fixture_sensor.md")
    digest = m.tensor_digest
    a = run_pymdp(m, FixtureSensorEnvironment("x86_accumulator"), steps=1, seed=0)
    b = run_pymdp(m, FixtureSensorEnvironment("x86_bitwise"), steps=1, seed=0)
    row = a["trajectory"][0]
    ref = policy_reference(m, [np.array([0.5, 0.5]), np.array([1.0, 0.0, 0.0])])
    np.testing.assert_allclose(row["q_pi"], ref["q_pi"], atol=1e-5)
    np.testing.assert_allclose(row["negative_efe"], ref["negative_efe"], atol=1e-5)
    assert row["action"] == [0, 2]
    assert row["q_pi"] == b["trajectory"][0]["q_pi"]
    assert row["revealed_observation"] == [3, 2]
    assert b["trajectory"][0]["revealed_observation"] == [2, 2]
    assert a["final_posterior"][0][0] > 0.999
    assert b["final_posterior"][0][1] > 0.999
    assert m.tensor_digest == digest
    cost = list(m.C)
    cost[1] = np.array([0.0, -0.01, -2.0])
    high = run_pymdp(
        replace(m, C=tuple(cost)),
        FixtureSensorEnvironment("x86_accumulator"),
        steps=1,
        seed=0,
    )
    assert high["trajectory"][0]["action"] == [0, 0]


def test_bounds_and_sensor_semantics_fail_before_acquisition():
    import pytest

    from src.pymdp_runtime import PymdpRuntime

    model = compile_gnn(ROOT / "configs/gnn/fixture_sensor.md")
    env = FixtureSensorEnvironment("x86_accumulator")
    for kwargs in ({"steps": 0, "seed": 0}, {"steps": 1, "seed": -1}):
        with pytest.raises(ValueError, match="GNN_INVALID_RUN_BOUND"):
            run_pymdp(model, env, **kwargs)
    assert not env.acquisition_log
    with pytest.raises(ValueError, match="GNN_SENSOR_CONTRACT_MISMATCH"):
        run_pymdp(
            replace(model, state_labels=(("wrong", "other"), model.state_labels[1])),
            env,
            steps=1,
            seed=0,
        )
    env.available_actions = (0, 1)
    with pytest.raises(ValueError, match="SENSOR_INVALID_ACTION"):
        run_pymdp(model, env, steps=1, seed=0)
    assert not env.acquisition_log
    # Alter the actual Agent using Equinox's real immutable-tree replacement API.
    import equinox as eqx
    import jax.numpy as jnp

    runtime = PymdpRuntime(model)
    runtime.agent = eqx.tree_at(
        lambda agent: agent.A[0], runtime.agent, jnp.zeros_like(runtime.agent.A[0])
    )
    with pytest.raises(RuntimeError, match="GNN_AGENT_TENSOR_MUTATION"):
        runtime.assert_unchanged()


def test_zero_policy_prior_rejected_before_acquisition():
    import pytest

    model = compile_gnn(ROOT / "configs/gnn/fixture_sensor.md")
    env = FixtureSensorEnvironment("x86_accumulator")
    with pytest.raises(ValueError, match="GNN_ZERO_POLICY_PRIOR_UNSUPPORTED"):
        run_pymdp(replace(model, E=np.array([1.0, 0.0, 0.0])), env, steps=1, seed=0)
    assert env.acquisition_log == []


def test_dropped_agent_tensor_is_detected():
    import equinox as eqx
    import pytest

    from src.pymdp_runtime import PymdpRuntime

    model = compile_gnn(ROOT / "configs/gnn/fixture_sensor.md")
    for field in ("A", "B", "C", "D"):
        runtime = PymdpRuntime(model)
        runtime.agent = eqx.tree_at(
            lambda agent: getattr(agent, field),
            runtime.agent,
            getattr(runtime.agent, field)[:-1],
        )
        with pytest.raises(RuntimeError, match="GNN_AGENT_TENSOR_MUTATION"):
            runtime.assert_unchanged()
