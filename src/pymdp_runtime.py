"""Explicit real pymdp 1.0.2 action-perception adapter for compiled GNN models."""

from __future__ import annotations

import importlib.metadata

import jax.numpy as jnp
import numpy as np
from pymdp.agent import Agent

from .gnn_model import CompiledModel
from .gnn_reference import posterior_reference


class PymdpRuntime:
    def __init__(self, model: CompiledModel, action_selection: str = "deterministic"):
        if action_selection not in ("deterministic", "stochastic"):
            raise ValueError("GNN_INVALID_ACTION_SELECTION")
        if np.any(model.E <= 0):
            raise ValueError("GNN_ZERO_POLICY_PRIOR_UNSUPPORTED")
        self.model = model
        self.agent = Agent(
            A=[jnp.array(v.copy()) for v in model.A],
            B=[jnp.array(v.copy()) for v in model.B],
            C=[jnp.array(v.copy()) for v in model.C],
            D=[jnp.array(v.copy()) for v in model.D],
            E=jnp.array(model.E.copy()),
            A_dependencies=[list(d) for d in model.A_dependencies],
            B_dependencies=[[i] for i in range(len(model.B))],
            B_action_dependencies=[[i] for i in range(len(model.B))],
            num_controls=[len(a) for a in model.actions],
            policies=np.array(model.policies),
            policy_len=model.horizon,
            gamma=model.gamma,
            action_selection=action_selection,
            sampling_mode="full",
            use_param_info_gain=False,
            learn_A=False,
            learn_B=False,
            learn_C=False,
            learn_D=False,
            learn_E=False,
        )
        self.prior = self.agent.D
        self._tensors = [
            [np.array(v) for v in getattr(self.agent, k)] for k in ("A", "B", "C", "D")
        ] + [[np.array(self.agent.E)]]

    def infer(self, observation):
        if len(observation) != len(self.model.A) or any(
            type(o) is not int or not 0 <= o < len(self.model.outcomes[i])
            for i, o in enumerate(observation)
        ):
            raise ValueError("GNN_INVALID_OBSERVATION")
        prior = [np.asarray(v)[0] for v in self.prior]
        posterior_reference(
            self.model, prior, observation
        )  # Reject impossible evidence before upstream logarithm flooring.
        qs = self.agent.infer_states(
            [jnp.array([o]) for o in observation], empirical_prior=self.prior
        )
        self.assert_unchanged()
        return qs

    def assert_unchanged(self):
        current = [
            [np.array(v) for v in getattr(self.agent, k)] for k in ("A", "B", "C", "D")
        ] + [[np.array(self.agent.E)]]
        if len(self._tensors) != len(current) or any(
            len(old) != len(new) for old, new in zip(self._tensors, current)
        ):
            raise RuntimeError("GNN_AGENT_TENSOR_MUTATION")
        for old, new in zip(self._tensors, current):
            for a, b in zip(old, new):
                if a.shape != b.shape or not np.array_equal(a, b):
                    raise RuntimeError("GNN_AGENT_TENSOR_MUTATION")


def beliefs(qs):
    return [np.asarray(q)[0, -1].astype(float).tolist() for q in qs]


def run_pymdp(
    model: CompiledModel,
    environment,
    *,
    steps: int,
    seed: int,
    action_selection: str = "deterministic",
):
    if (
        type(steps) is not int
        or not 1 <= steps <= 100
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise ValueError("GNN_INVALID_RUN_BOUND")
    if (
        model.state_labels
        != (("x86_accumulator", "x86_bitwise"), ("idle", "static", "dynamic"))
        or model.actions != (("noop",), ("idle", "static", "dynamic"))
        or model.outcomes
        != (
            ("missing", "load_immediate", "result_3", "result_5", "unseen"),
            ("idle", "static", "dynamic"),
        )
    ):
        raise ValueError("GNN_SENSOR_CONTRACT_MISMATCH")
    runtime = PymdpRuntime(model, action_selection)
    digest = model.tensor_digest
    from jax import random

    key = random.PRNGKey(seed)
    observation = environment.reset()
    qs = runtime.infer(observation)
    trajectory = []
    for _ in range(steps):
        prior = [np.asarray(v)[0].astype(float).tolist() for v in runtime.prior]
        pi, neg = runtime.agent.infer_policies(qs)
        key, subkey = random.split(key)
        action = runtime.agent.sample_action(
            pi, rng_key=subkey[None, :] if action_selection == "stochastic" else None
        )
        chosen = np.asarray(action)[0].astype(int).tolist()
        runtime.prior = runtime.agent.update_empirical_prior(action, qs)
        if chosen[-1] not in environment.available_actions:
            raise ValueError("SENSOR_INVALID_ACTION")
        acquisitions_before = len(environment.acquisition_log)
        revealed = environment.step(chosen[-1])
        acquisitions_after = len(environment.acquisition_log)
        trajectory.append(
            {
                "prior": prior,
                "observation": list(observation),
                "posterior": beliefs(qs),
                "q_pi": np.asarray(pi)[0].astype(float).tolist(),
                "negative_efe": np.asarray(neg)[0].astype(float).tolist(),
                "efe": (-np.asarray(neg)[0]).astype(float).tolist(),
                "action": chosen,
                "revealed_observation": list(revealed),
                "decision_before_acquisition": True,
                "acquisition_count_before": acquisitions_before,
                "acquisition_count_after": acquisitions_after,
            }
        )
        qs = runtime.infer(revealed)
        observation = revealed
        runtime.assert_unchanged()
        if model.tensor_digest != digest:
            raise RuntimeError("GNN_MODEL_TENSOR_MUTATION")
    return {
        "model_digest": digest,
        "model_source_sha256": model.source_sha256,
        "backend": "pymdp.Agent",
        "backend_version": importlib.metadata.version("inferactively-pymdp"),
        "time_convention": "first_observation_D",
        "objective": "negative_efe=expected_utility+state_information_gain; q_pi=softmax(gamma*negative_efe+log(E))",
        "seed": seed,
        "action_selection": action_selection,
        "steps": steps,
        "trajectory": trajectory,
        "final_posterior": beliefs(qs),
        "evaluation": environment.evaluation_provenance(),
        "tensors_unchanged": True,
    }
